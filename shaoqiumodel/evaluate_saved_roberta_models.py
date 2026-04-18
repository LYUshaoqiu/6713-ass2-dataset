import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_NAMES = ["Non-hate", "Hate"]


def normalize_header(name: str) -> str:
    return str(name).replace("\ufeff", "").strip().lower()


def load_model(model_dir: str):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device


def map_label(raw_label: int, label_mapping: str) -> int:
    if label_mapping == "identity":
        mapped = raw_label
    elif label_mapping == "02_only":
        if raw_label == 1:
            return -1
        mapped = 1 if raw_label == 2 else 0
    elif label_mapping == "hateonly":
        mapped = 1 if raw_label == 2 else 0
    elif label_mapping == "binary":
        mapped = 1 if raw_label in (1, 2) else 0
    else:
        raise ValueError(f"Unsupported label mapping: {label_mapping}")

    if mapped not in (0, 1):
        raise ValueError(
            f"Label mapping '{label_mapping}' produced unsupported label {mapped!r} from raw label {raw_label!r}."
        )
    return mapped


def read_dataset(csv_path: str, label_mapping: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in dataset: {csv_path}")

        header_map = {normalize_header(name): name for name in reader.fieldnames}
        text_key = header_map.get("text") or header_map.get("comment")
        label_key = header_map.get("label")

        if text_key is None or label_key is None:
            raise ValueError(
                f"Dataset must contain text/comment and label columns. Found: {reader.fieldnames}"
            )

        for idx, row in enumerate(reader, start=1):
            text = str(row.get(text_key, "")).strip()
            label_raw = str(row.get(label_key, "")).strip()
            if not text or label_raw == "":
                continue
            raw_label = int(label_raw)
            label = map_label(raw_label, label_mapping)
            if label == -1:
                continue
            rows.append({"row_id": idx, "text": text, "label": label, "raw_label": raw_label})
    return rows


def predict_rows(
    rows: List[Dict[str, object]],
    tokenizer,
    model,
    device,
    max_length: int,
) -> List[Dict[str, object]]:
    predictions: List[Dict[str, object]] = []
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
        pred_label = int(max(range(len(probs)), key=lambda i: probs[i]))
        predictions.append(
            {
                "row_id": row["row_id"],
                "text": row["text"],
                "true_label": row["label"],
                "raw_label": row["raw_label"],
                "pred_label": pred_label,
                "true_name": LABEL_NAMES[int(row["label"])],
                "pred_name": LABEL_NAMES[pred_label],
                "confidence": float(probs[pred_label]),
                "prob_non_hate": float(probs[0]),
                "prob_hate": float(probs[1]),
            }
        )
    return predictions


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def per_class_metrics(y_true: List[int], y_pred: List[int], target: int) -> Dict[str, float]:
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


def compute_metrics(predictions: List[Dict[str, object]]) -> Dict[str, object]:
    y_true = [int(row["true_label"]) for row in predictions]
    y_pred = [int(row["pred_label"]) for row in predictions]
    total = len(predictions)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)

    metrics_0 = per_class_metrics(y_true, y_pred, 0)
    metrics_1 = per_class_metrics(y_true, y_pred, 1)
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


def save_predictions(predictions: List[Dict[str, object]], output_path: str) -> None:
    fieldnames = [
        "row_id",
        "text",
        "true_label",
        "raw_label",
        "pred_label",
        "true_name",
        "pred_name",
        "confidence",
        "prob_non_hate",
        "prob_hate",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def save_summary(summary_rows: List[Dict[str, object]], output_path: str) -> None:
    fieldnames = [
        "model_name",
        "model_dir",
        "dataset_path",
        "label_mapping",
        "num_samples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "non_hate_precision",
        "non_hate_recall",
        "non_hate_f1",
        "non_hate_support",
        "hate_precision",
        "hate_recall",
        "hate_f1",
        "hate_support",
        "true_0_pred_0",
        "true_0_pred_1",
        "true_1_pred_0",
        "true_1_pred_1",
        "predictions_file",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def evaluate_model(
    model_name: str,
    model_dir: str,
    dataset_path: str,
    output_dir: str,
    max_length: int,
    label_mapping: str,
) -> Tuple[Dict[str, object], str]:
    tokenizer, model, device = load_model(model_dir)
    rows = read_dataset(dataset_path, label_mapping=label_mapping)
    predictions = predict_rows(rows, tokenizer, model, device, max_length=max_length)
    metrics = compute_metrics(predictions)

    predictions_path = os.path.join(output_dir, f"{model_name}_predictions.csv")
    save_predictions(predictions, predictions_path)

    metrics_json_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    summary_row = {
        "model_name": model_name,
        "model_dir": model_dir,
        "dataset_path": dataset_path,
        "label_mapping": label_mapping,
        "num_samples": metrics["num_samples"],
        "accuracy": f"{metrics['accuracy']:.6f}",
        "macro_f1": f"{metrics['macro_f1']:.6f}",
        "weighted_f1": f"{metrics['weighted_f1']:.6f}",
        "non_hate_precision": f"{metrics['non_hate_precision']:.6f}",
        "non_hate_recall": f"{metrics['non_hate_recall']:.6f}",
        "non_hate_f1": f"{metrics['non_hate_f1']:.6f}",
        "non_hate_support": metrics["non_hate_support"],
        "hate_precision": f"{metrics['hate_precision']:.6f}",
        "hate_recall": f"{metrics['hate_recall']:.6f}",
        "hate_f1": f"{metrics['hate_f1']:.6f}",
        "hate_support": metrics["hate_support"],
        "true_0_pred_0": metrics["confusion_matrix"]["true_0_pred_0"],
        "true_0_pred_1": metrics["confusion_matrix"]["true_0_pred_1"],
        "true_1_pred_0": metrics["confusion_matrix"]["true_1_pred_0"],
        "true_1_pred_1": metrics["confusion_matrix"]["true_1_pred_1"],
        "predictions_file": predictions_path,
    }
    return summary_row, metrics_json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved Twitter-RoBERTa binary classifiers on a labeled CSV dataset."
    )
    parser.add_argument("--dataset", required=True, help="CSV file containing text/comment and label columns.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more model directories to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where comparison metrics and per-model predictions will be saved.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Max token length for tokenization.",
    )
    parser.add_argument(
        "--label-mapping",
        choices=["identity", "02_only", "hateonly", "binary"],
        default="identity",
        help=(
            "How to map dataset labels before evaluation. "
            "'02_only' drops label 1, maps 0->0 and 2->1. "
            "'hateonly' maps 0+1->0 and 2->1. "
            "'binary' maps 0->0 and 1+2->1."
        ),
    )
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    metrics_files: List[str] = []

    for model_arg in args.models:
        model_dir = os.path.abspath(model_arg)
        model_name = os.path.basename(os.path.normpath(model_dir))
        print(f"Evaluating {model_name} on {dataset_path}", file=sys.stderr)
        summary_row, metrics_json_path = evaluate_model(
            model_name=model_name,
            model_dir=model_dir,
            dataset_path=dataset_path,
            output_dir=output_dir,
            max_length=args.max_length,
            label_mapping=args.label_mapping,
        )
        summary_rows.append(summary_row)
        metrics_files.append(metrics_json_path)

    summary_path = os.path.join(output_dir, "model_comparison_summary.csv")
    save_summary(summary_rows, summary_path)

    print()
    print(f"Saved summary: {summary_path}")
    for path in metrics_files:
        print(f"Saved metrics: {path}")
    for row in summary_rows:
        print(
            f"{row['model_name']}: "
            f"accuracy={row['accuracy']} "
            f"macro_f1={row['macro_f1']} "
            f"weighted_f1={row['weighted_f1']}"
        )


if __name__ == "__main__":
    main()
