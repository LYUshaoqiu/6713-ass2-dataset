import argparse
import csv
import json
import math
import os
import random
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_NAMES = ["Non-hate", "Hate"]
SCORE_COLUMNS = ["difficulty", "workload", "teaching_quality", "recommendation"]


def normalize_header(name: str) -> str:
    return str(name).replace("\ufeff", "").strip().lower()


def map_label(raw_label: int, label_mapping: str) -> int:
    if label_mapping == "hateonly":
        return 1 if raw_label == 2 else 0
    if label_mapping == "binary":
        return 1 if raw_label in (1, 2) else 0
    raise ValueError(f"Unsupported label mapping: {label_mapping}")


def load_rows(dataset_path: str, label_mapping: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(dataset_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in dataset: {dataset_path}")

        header_map = {normalize_header(name): name for name in reader.fieldnames}
        text_key = header_map.get("text") or header_map.get("comment")
        label_key = header_map.get("label")
        score_keys = {}
        for feature_name in SCORE_COLUMNS:
            if feature_name not in header_map:
                raise ValueError(f"Missing required score column '{feature_name}' in {dataset_path}")
            score_keys[feature_name] = header_map[feature_name]

        if text_key is None or label_key is None:
            raise ValueError(f"Dataset must contain text/comment and label columns. Found: {reader.fieldnames}")

        for idx, row in enumerate(reader, start=1):
            text = str(row.get(text_key, "")).strip()
            raw_label_text = str(row.get(label_key, "")).strip()
            if not text or raw_label_text == "":
                continue

            raw_label = int(raw_label_text)
            score_values = [float(row[score_keys[name]]) for name in SCORE_COLUMNS]
            rows.append(
                {
                    "row_id": idx,
                    "text": text,
                    "raw_label": raw_label,
                    "label": map_label(raw_label, label_mapping),
                    "scores": score_values,
                }
            )
    return rows


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device


def attach_model_probabilities(rows: List[Dict[str, object]], model_dir: str, max_length: int) -> None:
    tokenizer, model, device = load_model(model_dir)
    for row in rows:
        encoding = tokenizer(
            str(row["text"]),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = F.softmax(logits, dim=-1).squeeze().detach().cpu().tolist()
        row["prob_non_hate"] = float(probs[0])
        row["prob_hate"] = float(probs[1])
        row["text_pred"] = int(probs[1] >= probs[0])


def stratified_split(
    rows: Sequence[Dict[str, object]],
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rng = random.Random(seed)
    grouped: Dict[int, List[Dict[str, object]]] = {0: [], 1: []}
    for row in rows:
        grouped[int(row["label"])].append(dict(row))

    train_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []
    for label, items in grouped.items():
        rng.shuffle(items)
        test_count = max(1, int(round(len(items) * test_ratio)))
        if test_count >= len(items):
            test_count = max(1, len(items) - 1)
        test_rows.extend(items[:test_count])
        train_rows.extend(items[test_count:])

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def compute_feature_stats(rows: Sequence[Dict[str, object]]) -> Tuple[List[float], List[float]]:
    feature_count = 1 + len(SCORE_COLUMNS)
    means = [0.0] * feature_count
    stds = [0.0] * feature_count

    for row in rows:
        values = [float(row["prob_hate"])] + [float(v) for v in row["scores"]]
        for i, value in enumerate(values):
            means[i] += value
    for i in range(feature_count):
        means[i] /= max(len(rows), 1)

    for row in rows:
        values = [float(row["prob_hate"])] + [float(v) for v in row["scores"]]
        for i, value in enumerate(values):
            stds[i] += (value - means[i]) ** 2
    for i in range(feature_count):
        stds[i] = math.sqrt(stds[i] / max(len(rows), 1))
        if stds[i] == 0:
            stds[i] = 1.0
    return means, stds


def build_tensors(
    rows: Sequence[Dict[str, object]],
    means: Sequence[float],
    stds: Sequence[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    for row in rows:
        raw_features = [float(row["prob_hate"])] + [float(v) for v in row["scores"]]
        normalized = [(value - mean) / std for value, mean, std in zip(raw_features, means, stds)]
        features.append(normalized)
        labels.append(float(row["label"]))
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return x, y


class LogisticStacker(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def train_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float,
) -> LogisticStacker:
    model = LogisticStacker(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
    return model


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, object]:
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)

    def per_class(target: int) -> Dict[str, float]:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == target and p == target)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != target and p == target)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == target and p != target)
        support = sum(1 for t in y_true if t == target)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    metrics_0 = per_class(0)
    metrics_1 = per_class(1)
    macro_f1 = (metrics_0["f1"] + metrics_1["f1"]) / 2
    weighted_f1 = safe_div(
        metrics_0["f1"] * metrics_0["support"] + metrics_1["f1"] * metrics_1["support"],
        total,
    )

    confusion = {
        "true_0_pred_0": sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0),
        "true_0_pred_1": sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1),
        "true_1_pred_0": sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0),
        "true_1_pred_1": sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1),
    }

    return {
        "num_samples": total,
        "accuracy": safe_div(correct, total),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "non_hate_precision": metrics_0["precision"],
        "non_hate_recall": metrics_0["recall"],
        "non_hate_f1": metrics_0["f1"],
        "non_hate_support": metrics_0["support"],
        "hate_precision": metrics_1["precision"],
        "hate_recall": metrics_1["recall"],
        "hate_f1": metrics_1["f1"],
        "hate_support": metrics_1["support"],
        "confusion_matrix": confusion,
    }


def evaluate_text_baseline(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    y_true = [int(row["label"]) for row in rows]
    y_pred = [int(row["text_pred"]) for row in rows]
    return compute_binary_metrics(y_true, y_pred)


def evaluate_stacker(model: LogisticStacker, x: torch.Tensor, rows: Sequence[Dict[str, object]]) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).squeeze(1).tolist()
    y_true = [int(row["label"]) for row in rows]
    y_pred = [1 if prob >= 0.5 else 0 for prob in probs]
    metrics = compute_binary_metrics(y_true, y_pred)

    predictions: List[Dict[str, object]] = []
    for row, prob, pred in zip(rows, probs, y_pred):
        predictions.append(
            {
                "row_id": row["row_id"],
                "text": row["text"],
                "raw_label": row["raw_label"],
                "mapped_label": row["label"],
                "text_prob_hate": row["prob_hate"],
                "text_pred": row["text_pred"],
                "difficulty": row["scores"][0],
                "workload": row["scores"][1],
                "teaching_quality": row["scores"][2],
                "recommendation": row["scores"][3],
                "stack_prob_hate": float(prob),
                "stack_pred": int(pred),
            }
        )
    return metrics, predictions


def save_json(data: Dict[str, object], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_csv(rows: List[Dict[str, object]], output_path: str) -> None:
    if not rows:
        return
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_for_model(
    dataset_path: str,
    model_dir: str,
    output_dir: str,
    label_mapping: str,
    max_length: int,
    test_ratio: float,
    seed: int,
    epochs: int,
    lr: float,
) -> Dict[str, object]:
    rows = load_rows(dataset_path, label_mapping=label_mapping)
    attach_model_probabilities(rows, model_dir=model_dir, max_length=max_length)

    train_rows, test_rows = stratified_split(rows, test_ratio=test_ratio, seed=seed)
    means, stds = compute_feature_stats(train_rows)
    x_train, y_train = build_tensors(train_rows, means, stds)
    x_test, _ = build_tensors(test_rows, means, stds)

    stacker = train_model(x_train=x_train, y_train=y_train, epochs=epochs, lr=lr)
    text_metrics = evaluate_text_baseline(test_rows)
    multimodal_metrics, prediction_rows = evaluate_stacker(stacker, x_test, test_rows)

    model_name = os.path.basename(os.path.normpath(model_dir))
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    save_json(text_metrics, os.path.join(model_output_dir, "text_only_metrics.json"))
    save_json(multimodal_metrics, os.path.join(model_output_dir, "text_plus_scores_metrics.json"))
    save_csv(prediction_rows, os.path.join(model_output_dir, "test_predictions.csv"))

    coefficients = stacker.linear.weight.detach().cpu().squeeze(0).tolist()
    intercept = float(stacker.linear.bias.detach().cpu().squeeze())
    feature_names = ["text_prob_hate"] + SCORE_COLUMNS
    weights_path = os.path.join(model_output_dir, "stacker_weights.json")
    save_json(
        {
            "feature_names": feature_names,
            "coefficients": {name: float(value) for name, value in zip(feature_names, coefficients)},
            "intercept": intercept,
            "normalization_means": {name: float(value) for name, value in zip(feature_names, means)},
            "normalization_stds": {name: float(value) for name, value in zip(feature_names, stds)},
            "split": {
                "seed": seed,
                "test_ratio": test_ratio,
                "train_size": len(train_rows),
                "test_size": len(test_rows),
            },
        },
        weights_path,
    )

    return {
        "model_name": model_name,
        "label_mapping": label_mapping,
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "text_accuracy": f"{text_metrics['accuracy']:.6f}",
        "text_macro_f1": f"{text_metrics['macro_f1']:.6f}",
        "text_weighted_f1": f"{text_metrics['weighted_f1']:.6f}",
        "text_hate_recall": f"{text_metrics['hate_recall']:.6f}",
        "multimodal_accuracy": f"{multimodal_metrics['accuracy']:.6f}",
        "multimodal_macro_f1": f"{multimodal_metrics['macro_f1']:.6f}",
        "multimodal_weighted_f1": f"{multimodal_metrics['weighted_f1']:.6f}",
        "multimodal_hate_recall": f"{multimodal_metrics['hate_recall']:.6f}",
        "model_output_dir": model_output_dir,
    }


def save_summary(rows: List[Dict[str, object]], output_path: str) -> None:
    if not rows:
        return
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a simple multimodal stacker from RoBERTa hate probability plus four rating features."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-mapping", choices=["hateonly", "binary"], default="hateonly")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    for model_dir in args.models:
        summary_rows.append(
            run_for_model(
                dataset_path=dataset_path,
                model_dir=os.path.abspath(model_dir),
                output_dir=output_dir,
                label_mapping=args.label_mapping,
                max_length=args.max_length,
                test_ratio=args.test_ratio,
                seed=args.seed,
                epochs=args.epochs,
                lr=args.lr,
            )
        )

    summary_path = os.path.join(output_dir, "multimodal_comparison_summary.csv")
    save_summary(summary_rows, summary_path)
    print(f"Saved summary: {summary_path}")
    for row in summary_rows:
        print(
            f"{row['model_name']}: "
            f"text_macro_f1={row['text_macro_f1']} -> multimodal_macro_f1={row['multimodal_macro_f1']}, "
            f"text_hate_recall={row['text_hate_recall']} -> multimodal_hate_recall={row['multimodal_hate_recall']}"
        )


if __name__ == "__main__":
    main()
