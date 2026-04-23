import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer


SCORE_COLUMNS = ["difficulty", "workload", "teaching_quality", "recommendation"]
LABEL_NAMES = ["Non-hate", "Hate"]

#CSV header processing Remove unnecessary information
def normalize_header(name: str) -> str:
    return str(name).replace("\ufeff", "").strip().lower()




#Map original labels to final binary labels.
def map_label(raw_label: int, label_mapping: str) -> int:
    if label_mapping == "binary":
        return 1 if raw_label in (1, 2) else 0
    if label_mapping == "hateonly":
        return 1 if raw_label == 2 else 0
    raise ValueError(f"Unsupported label mapping: {label_mapping}")



#To prevent the denominator from being 0.
def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

#dictionary is stored.
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


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))

#Packaged Sample
@dataclass
class ReviewRow:
    row_id: int
    text: str
    raw_label: int
    label: int
    scores: List[float]#4 Numerical Features



#The `load_rows` function is responsible for checking the data format, filtering out empty samples, completing label mapping, and modifying the CSV file.
def load_rows(dataset_path: str, label_mapping: str) -> List[ReviewRow]:
    rows: List[ReviewRow] = []
    with open(dataset_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:

            raise ValueError(f"No header found in dataset: {dataset_path}")

        header_map = {normalize_header(name): name for name in reader.fieldnames}
        text_key = header_map.get("comment") or header_map.get("text")
        label_key = header_map.get("label")


        
        if text_key is None or label_key is None:
            raise ValueError(
                f"Dataset must contain comment/text and label columns. Found: {reader.fieldnames}"
            )

        score_keys = {}
        for feature_name in SCORE_COLUMNS:
            if feature_name not in header_map:
                raise ValueError(f"Missing required score column '{feature_name}' in {dataset_path}")
            score_keys[feature_name] = header_map[feature_name]

        for idx, row in enumerate(reader, start=1):
            text = str(row.get(text_key, "")).strip()
            raw_label_text = str(row.get(label_key, "")).strip()
            if not text or raw_label_text == "":
                continue

            raw_label = int(raw_label_text)
            score_values = [float(row[score_keys[name]]) for name in SCORE_COLUMNS]
            rows.append(
                ReviewRow(
                    row_id=idx,
                    text=text,
                    raw_label=raw_label,
                    label=map_label(raw_label, label_mapping),
                    scores=score_values,
                )
            )
    if not rows:
        raise ValueError(f"No valid rows found in dataset: {dataset_path}")
    return rows



#This is for dividing the datasets.
#Because the overall dataset is relatively small, there may be overfitting issues.
def stratified_split(
    rows: Sequence[ReviewRow],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[ReviewRow], List[ReviewRow], List[ReviewRow]]:
    rng = random.Random(seed)
    grouped: Dict[int, List[ReviewRow]] = {0: [], 1: []}
    for row in rows:
        grouped[int(row.label)].append(row)

    train_rows: List[ReviewRow] = []
    val_rows: List[ReviewRow] = []
    test_rows: List[ReviewRow] = []

    for label, items in grouped.items():
        if len(items) < 3:
            raise ValueError(
                f"Label {label} only has {len(items)} samples after mapping; cannot make train/val/test split."
            )
        items = list(items)
        rng.shuffle(items)

        #Calculate the number of elements in each class.

        test_count = max(1, int(round(len(items) * test_ratio)))
        val_count = max(1, int(round(len(items) * val_ratio)))

#Our model has a relatively small number of samples. To prevent issues related to the number of trainees, a dynamic control is used here.
        while len(items) - test_count - val_count < 1:
            if test_count >= val_count and test_count > 1:
                test_count -= 1
            elif val_count > 1:
                val_count -= 1
            else:
                break

        test_rows.extend(items[:test_count])
        val_rows.extend(items[test_count:test_count + val_count])
        train_rows.extend(items[test_count + val_count:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows

#Feature statistics and normalization
def compute_score_stats(rows: Sequence[ReviewRow]) -> Tuple[List[float], List[float]]:
    means = []
    stds = []
    for idx in range(len(SCORE_COLUMNS)):
        values = [float(row.scores[idx]) for row in rows]
        mean_value = sum(values) / len(values)
#Calculating Population Variance
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std_value = math.sqrt(variance) if variance > 0 else 1.0
        means.append(mean_value)
        stds.append(std_value)
    return means, stds


def normalize_scores(score_values: Sequence[float], means: Sequence[float], stds: Sequence[float]) -> List[float]:
    return [(value - mean) / std for value, mean, std in zip(score_values, means, stds)]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


#eembedding part
def encode_rows(
    rows: Sequence[ReviewRow],
    model_dir: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> Dict[int, List[float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    encoder = AutoModel.from_pretrained(model_dir).to(device)
    encoder.eval()

    embeddings: Dict[int, List[float]] = {}
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            encodings = tokenizer(
                [row.text for row in batch_rows],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            #join the demo
            outputs = encoder(
                input_ids=encodings["input_ids"].to(device),
                attention_mask=encodings["attention_mask"].to(device),
            )
            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()
            for row, embedding in zip(batch_rows, cls_embeddings):
                embeddings[row.row_id] = embedding.tolist()

    return embeddings

#feature tensors 
def build_feature_tensors(
    rows: Sequence[ReviewRow],
    embeddings: Dict[int, List[float]],
    score_means: Sequence[float],
    score_stds: Sequence[float],
    include_scores: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features: List[List[float]] = []
    labels: List[float] = []
    for row in rows:
        feature_values = list(embeddings[row.row_id])
        if include_scores:
            feature_values.extend(normalize_scores(row.scores, score_means, score_stds))
        features.append(feature_values)
        labels.append(float(row.label))
#change to the pytorch
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return x, y


#Used for binary classification
class FusionClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


#Evaluation indicators like TP FP FN support precision recall f1

def compute_binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, object]:
    total = len(y_true)
    correct = sum(1 for target, pred in zip(y_true, y_pred) if target == pred)

    def per_class(target_label: int) -> Dict[str, float]:
        tp = sum(1 for target, pred in zip(y_true, y_pred) if target == target_label and pred == target_label)
        fp = sum(1 for target, pred in zip(y_true, y_pred) if target != target_label and pred == target_label)
        fn = sum(1 for target, pred in zip(y_true, y_pred) if target == target_label and pred != target_label)
        support = sum(1 for target in y_true if target == target_label)
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
        "true_0_pred_0": sum(1 for target, pred in zip(y_true, y_pred) if target == 0 and pred == 0),
        "true_0_pred_1": sum(1 for target, pred in zip(y_true, y_pred) if target == 0 and pred == 1),
        "true_1_pred_0": sum(1 for target, pred in zip(y_true, y_pred) if target == 1 and pred == 0),
        "true_1_pred_1": sum(1 for target, pred in zip(y_true, y_pred) if target == 1 and pred == 1),
    }

#different standand
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

#Forward propagation test
def evaluate_model(
    model: FusionClassifier,
    loader: DataLoader,
    loss_fn: nn.Module,


    device: torch.device,
) -> Tuple[Dict[str, object], List[float], List[int], List[int]]:
    model.eval()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    probs_all: List[float] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))
            loss = loss_fn(logits, batch_y.to(device))
            losses.append(float(loss.item()))

            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().tolist()
            labels = batch_y.squeeze(1).int().tolist()
            preds = [1 if prob >= 0.5 else 0 for prob in probs]

            probs_all.extend(float(prob) for prob in probs)
            y_true.extend(int(label) for label in labels)
            y_pred.extend(int(pred) for pred in preds)

    metrics = compute_binary_metrics(y_true, y_pred)
    metrics["loss"] = sum(losses) / len(losses) if losses else 0.0
    return metrics, probs_all, y_true, y_pred

#train part
def train_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    hidden_dim: int,


    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    patience: int,
    device: torch.device,
) -> Tuple[FusionClassifier, List[Dict[str, object]], Dict[str, object]]:
    model = FusionClassifier(train_x.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    pos_count = int(train_y.sum().item())
    neg_count = int(train_y.shape[0] - pos_count)
    pos_weight = torch.tensor([safe_div(neg_count, max(pos_count, 1))], dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)

    history: List[Dict[str, object]] = []
    best_state = None
    best_metrics: Dict[str, object] = {}
    best_val_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: List[float] = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x.to(device))
            loss = loss_fn(logits, batch_y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))

            #            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # optimizer.step()
            # batch_losses.append(float(loss.item()))

        train_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
#test part
        val_metrics, _, _, _ = evaluate_model(model, val_loader, loss_fn, device)
        history.append(
            {


                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_hate_recall": val_metrics["hate_recall"],
            }
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["macro_f1"])
            best_metrics = dict(val_metrics)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
# Saving optimal parameters
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history, best_metrics


def build_prediction_rows(
    rows: Sequence[ReviewRow],
    probs: Sequence[float],
    preds: Sequence[int],
) -> List[Dict[str, object]]:
    prediction_rows: List[Dict[str, object]] = []
    for row, prob, pred in zip(rows, probs, preds):
        prediction_rows.append(
            {
                "row_id": row.row_id,
                "text": row.text,
                "raw_label": row.raw_label,
                "mapped_label": row.label,
                "difficulty": row.scores[0],
                "workload": row.scores[1],
                "teaching_quality": row.scores[2],
                "recommendation": row.scores[3],
                "prob_hate": float(prob),
                "pred_label": int(pred),
                "pred_name": LABEL_NAMES[int(pred)],
                "true_name": LABEL_NAMES[int(row.label)],


            }
        )
    return prediction_rows

#single sxp part
def run_single_experiment(
    experiment_name: str,
    train_rows: Sequence[ReviewRow],
    val_rows: Sequence[ReviewRow],
    test_rows: Sequence[ReviewRow],
    embeddings: Dict[int, List[float]],


    score_means: Sequence[float],
    score_stds: Sequence[float],
    include_scores: bool,
    hidden_dim: int,
    dropout: float,
    lr: float,
    epochs: int,


    batch_size: int,
    patience: int,
    device: torch.device,
    output_dir: str,
) -> Dict[str, object]:
    experiment_dir = os.path.join(output_dir, experiment_name)
    ensure_dir(experiment_dir)

    train_x, train_y = build_feature_tensors(
        train_rows, embeddings, score_means, score_stds, include_scores=include_scores
    )
    val_x, val_y = build_feature_tensors(
        val_rows, embeddings, score_means, score_stds, include_scores=include_scores
    )
    test_x, test_y = build_feature_tensors(
        test_rows, embeddings, score_means, score_stds, include_scores=include_scores
    )

    model, history, best_val_metrics = train_classifier(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        device=device,


    )

    loss_fn = nn.BCEWithLogitsLoss()
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)
    test_metrics, test_probs, _, test_preds = evaluate_model(model, test_loader, loss_fn, device)
    prediction_rows = build_prediction_rows(test_rows, test_probs, test_preds)


#save all the reasults
    torch.save(model.state_dict(), os.path.join(experiment_dir, "classifier_head.pt"))
    save_json({"history": history, "best_val_metrics": best_val_metrics}, os.path.join(experiment_dir, "training_history.json"))
    save_json(test_metrics, os.path.join(experiment_dir, "test_metrics.json"))
    save_csv(prediction_rows, os.path.join(experiment_dir, "test_predictions.csv"))

    return {
        "experiment": experiment_name,
        "include_scores": include_scores,
        "test_metrics": test_metrics,
        "best_val_metrics": best_val_metrics,
        "output_dir": experiment_dir,
    }


def summarize_split(rows: Sequence[ReviewRow]) -> Dict[str, int]:
    counts = Counter(int(row.label) for row in rows)
    return {
        "total": len(rows),
        "label_0": counts.get(0, 0),
        "label_1": counts.get(1, 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare frozen RoBERTa text-only vs text+scores joint modeling on course review data."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--model-dir",
        default=os.path.join(os.path.dirname(__file__), "roberta_da_course"),
        help="Path to the base RoBERTa encoder directory (default: roberta_da_course).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "results_text_scores_joint_da_course"),
        help="Directory for joint-model outputs (default: results_text_scores_joint_da_course).",
    )
    parser.add_argument("--label-mapping", choices=["binary", "hateonly"], default="binary")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)

    #drop num i design the 0.2/0.3
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_path = os.path.abspath(args.dataset)
    model_dir = os.path.abspath(args.model_dir)
    output_dir = os.path.abspath(args.output_dir)
    ensure_dir(output_dir)

    rows = load_rows(dataset_path, label_mapping=args.label_mapping)
    train_rows, val_rows, test_rows = stratified_split(
        rows, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )
    score_means, score_stds = compute_score_stats(train_rows)

#show the path
    device = get_device()
    print(f"Using device: {device}")
    print(f"Dataset: {dataset_path}")
    print(f"Base model: {model_dir}")

#show the dataset
    print(f"Label mapping: {args.label_mapping}")
    print(f"Train split: {summarize_split(train_rows)}")
    print(f"Val split:   {summarize_split(val_rows)}")
    print(f"Test split:  {summarize_split(test_rows)}")

    embeddings = encode_rows(
        rows=rows,
        model_dir=model_dir,
        max_length=args.max_length,
        batch_size=args.encode_batch_size,
        device=device,
    )

#test part the 1 test
    text_only_result = run_single_experiment(
        experiment_name="text_only",
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        embeddings=embeddings,
        score_means=score_means,
        score_stds=score_stds,

        include_scores=False,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.train_batch_size,

        patience=args.patience,
        device=device,
        output_dir=output_dir,
    )
#plus test
    text_plus_scores_result = run_single_experiment(
        experiment_name="text_plus_scores",
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        embeddings=embeddings,


        score_means=score_means,
        score_stds=score_stds,
        include_scores=True,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,

        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        patience=args.patience,
        device=device,
        output_dir=output_dir,
    )


#final result summ
    summary = {
        "dataset": dataset_path,
        "model_dir": model_dir,
        "label_mapping": args.label_mapping,
        "seed": args.seed,
        "max_length": args.max_length,
        "score_columns": SCORE_COLUMNS,
        "score_normalization": {
            "means": {name: float(value) for name, value in zip(SCORE_COLUMNS, score_means)},
            "stds": {name: float(value) for name, value in zip(SCORE_COLUMNS, score_stds)},
        },
        "splits": {
            "train": summarize_split(train_rows),
            "val": summarize_split(val_rows),
            "test": summarize_split(test_rows),
        },
        "results": {
            "text_only": text_only_result["test_metrics"],
            "text_plus_scores": text_plus_scores_result["test_metrics"],
        },
        "output_dirs": {
            "text_only": text_only_result["output_dir"],
            "text_plus_scores": text_plus_scores_result["output_dir"],
        },
    }

    save_json(summary, os.path.join(output_dir, "summary.json"))

    text_only_metrics = text_only_result["test_metrics"]
    joint_metrics = text_plus_scores_result["test_metrics"]
    print()
    print(
        f"text_only        accuracy={text_only_metrics['accuracy']:.4f} "
        f"macro_f1={text_only_metrics['macro_f1']:.4f} "
        f"hate_recall={text_only_metrics['hate_recall']:.4f}"
    )
    print(
        f"text_plus_scores accuracy={joint_metrics['accuracy']:.4f} "
        f"macro_f1={joint_metrics['macro_f1']:.4f} "
        f"hate_recall={joint_metrics['hate_recall']:.4f}"
    )
    print(f"Saved summary: {os.path.join(output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
