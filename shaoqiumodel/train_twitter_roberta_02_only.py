import argparse
import csv
import gc
import json
import math
import os
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.enabled = False


LABELS = ["Normal", "Hate"]


def normalize_header(name: str) -> str:
    return str(name).replace("\ufeff", "").strip().lower()


def resolve_model_ref(model_ref: str) -> str:
    return os.path.abspath(model_ref) if os.path.exists(model_ref) else model_ref


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_filtered_rows(csv_path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {csv_path}")

        header_map = {normalize_header(name): name for name in reader.fieldnames}
        text_key = header_map.get("text") or header_map.get("comment")
        label_key = header_map.get("label")
        if text_key is None or label_key is None:
            raise ValueError(f"Expected text/comment and label columns in {csv_path}")

        for row in reader:
            text = str(row[text_key]).strip()
            raw_label = int(str(row[label_key]).strip())
            if raw_label == 1:
                continue
            if raw_label not in (0, 2):
                continue
            rows.append(
                {
                    "text": text,
                    "label": 1 if raw_label == 2 else 0,
                    "raw_label": raw_label,
                }
            )
    return rows


def count_labels(rows: List[Dict[str, object]]) -> Dict[int, int]:
    counts = {0: 0, 1: 0}
    for row in rows:
        counts[int(row["label"])] += 1
    return counts


class TextDataset(Dataset):
    def __init__(self, rows: List[Dict[str, object]], tokenizer, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        enc = self.tokenizer(
            str(row["text"]),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


class ManualAdamW:
    def __init__(self, params, lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            grad = param.grad
            param.mul_(1.0 - self.lr * self.wd)
            self.m[i].mul_(self.b1).add_(grad, alpha=1.0 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(grad, grad, value=1.0 - self.b2)
            param.addcdiv_(self.m[i] / bc1, (self.v[i] / bc2).sqrt().add_(self.eps), value=-self.lr)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(labels: List[int], preds: List[int], avg_loss: float) -> Dict[str, object]:
    total = len(labels)
    correct = sum(1 for gold, pred in zip(labels, preds) if gold == pred)

    def per_class(target: int) -> Dict[str, float]:
        tp = sum(1 for gold, pred in zip(labels, preds) if gold == target and pred == target)
        fp = sum(1 for gold, pred in zip(labels, preds) if gold != target and pred == target)
        fn = sum(1 for gold, pred in zip(labels, preds) if gold == target and pred != target)
        support = sum(1 for gold in labels if gold == target)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    class_0 = per_class(0)
    class_1 = per_class(1)
    macro_f1 = (class_0["f1"] + class_1["f1"]) / 2
    weighted_f1 = safe_div(
        class_0["f1"] * class_0["support"] + class_1["f1"] * class_1["support"],
        total,
    )
    confusion = {
        "true_0_pred_0": sum(1 for gold, pred in zip(labels, preds) if gold == 0 and pred == 0),
        "true_0_pred_1": sum(1 for gold, pred in zip(labels, preds) if gold == 0 and pred == 1),
        "true_1_pred_0": sum(1 for gold, pred in zip(labels, preds) if gold == 1 and pred == 0),
        "true_1_pred_1": sum(1 for gold, pred in zip(labels, preds) if gold == 1 and pred == 1),
    }
    return {
        "loss": avg_loss,
        "accuracy": safe_div(correct, total),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "normal_precision": class_0["precision"],
        "normal_recall": class_0["recall"],
        "normal_f1": class_0["f1"],
        "normal_support": class_0["support"],
        "hate_precision": class_1["precision"],
        "hate_recall": class_1["recall"],
        "hate_f1": class_1["f1"],
        "hate_support": class_1["support"],
        "confusion_matrix": confusion,
    }


def evaluate(model, loader, device) -> Tuple[Dict[str, object], List[int], List[int]]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=labels,
            )
            losses.append(float(outputs.loss.item()))
            preds = outputs.logits.argmax(dim=-1).detach().cpu().tolist()
            all_preds.extend(int(x) for x in preds)
            all_labels.extend(int(x) for x in batch["label"].tolist())
    metrics = compute_metrics(all_labels, all_preds, safe_div(sum(losses), len(losses)))
    return metrics, all_labels, all_preds


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["label"].to(device),
        )
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(outputs.loss.item())
    return safe_div(total_loss, len(loader))


def save_json(data: Dict[str, object], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_predictions(rows: List[Dict[str, object]], preds: List[int], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text", "raw_label", "label", "pred_label", "true_name", "pred_name"],
        )
        writer.writeheader()
        for row, pred in zip(rows, preds):
            writer.writerow(
                {
                    "text": row["text"],
                    "raw_label": row["raw_label"],
                    "label": row["label"],
                    "pred_label": pred,
                    "true_name": LABELS[int(row["label"])],
                    "pred_name": LABELS[int(pred)],
                }
            )


def run_training(
    dataset_name: str,
    base_model_dir: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    output_root: str,
    batch_size: int,
    epochs: int,
    max_length: int,
    learning_rates: List[float],
    seed: int,
) -> Dict[str, object]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{dataset_name}] device={device}")
    if device.type == "cuda":
        print(f"[{dataset_name}] GPU={torch.cuda.get_device_name(0)}")

    train_rows = load_filtered_rows(train_csv)
    val_rows = load_filtered_rows(val_csv)
    test_rows = load_filtered_rows(test_csv)
    print(f"[{dataset_name}] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    print(f"[{dataset_name}] label counts train={count_labels(train_rows)} val={count_labels(val_rows)} test={count_labels(test_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    train_loader = DataLoader(TextDataset(train_rows, tokenizer, max_length), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TextDataset(val_rows, tokenizer, max_length), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(TextDataset(test_rows, tokenizer, max_length), batch_size=batch_size, shuffle=False, num_workers=0)

    best_state = None
    best_lr = None
    best_val_macro_f1 = -1.0
    best_history: List[Dict[str, object]] = []

    for lr in learning_rates:
        print(f"[{dataset_name}] trying lr={lr}")
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_dir,
            num_labels=2,
            ignore_mismatched_sizes=True,
        ).to(device)
        optimizer = ManualAdamW(model.parameters(), lr=lr, weight_decay=0.01)
        history: List[Dict[str, object]] = []

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_metrics, _, _ = evaluate(model, val_loader, device)
            epoch_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_hate_recall": val_metrics["hate_recall"],
            }
            history.append(epoch_row)
            print(
                f"[{dataset_name}] lr={lr} epoch={epoch}/{epochs} "
                f"train_loss={train_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"val_hate_recall={val_metrics['hate_recall']:.4f}"
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            torch.cuda.empty_cache()

        run_best = max(history, key=lambda item: item["val_macro_f1"])
        if run_best["val_macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = float(run_best["val_macro_f1"])
            best_lr = lr
            best_history = history
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        del model
        gc.collect()
        torch.cuda.empty_cache()

    if best_state is None or best_lr is None:
        raise RuntimeError(f"No model state was captured for {dataset_name}")

    final_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_dir,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    final_model.load_state_dict(best_state)
    final_model = final_model.to(device)

    test_metrics, test_labels, test_preds = evaluate(final_model, test_loader, device)
    print(
        f"[{dataset_name}] TEST accuracy={test_metrics['accuracy']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} "
        f"hate_recall={test_metrics['hate_recall']:.4f}"
    )

    output_dir = os.path.join(output_root, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    final_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    save_json(
        {
            "dataset_name": dataset_name,
            "base_model_dir": base_model_dir,
            "device": str(device),
            "best_learning_rate": best_lr,
            "best_val_macro_f1": best_val_macro_f1,
            "batch_size": batch_size,
            "epochs": epochs,
            "max_length": max_length,
            "seed": seed,
            "train_counts": count_labels(train_rows),
            "val_counts": count_labels(val_rows),
            "test_counts": count_labels(test_rows),
            "history": best_history,
            "test_metrics": test_metrics,
        },
        os.path.join(output_dir, "training_summary.json"),
    )
    save_predictions(test_rows, test_preds, os.path.join(output_dir, "test_predictions.csv"))

    del final_model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "dataset_name": dataset_name,
        "device": str(device),
        "best_learning_rate": best_lr,
        "best_val_macro_f1": round(best_val_macro_f1, 6),
        "test_accuracy": round(float(test_metrics["accuracy"]), 6),
        "test_macro_f1": round(float(test_metrics["macro_f1"]), 6),
        "test_weighted_f1": round(float(test_metrics["weighted_f1"]), 6),
        "test_hate_recall": round(float(test_metrics["hate_recall"]), 6),
        "model_dir": model_dir,
    }


def save_summary_csv(rows: List[Dict[str, object]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain Twitter-RoBERTa on labels 0 and 2 only, dropping label 1."
    )
    parser.add_argument("--hatexplain-train", required=True)
    parser.add_argument("--hatexplain-val", required=True)
    parser.add_argument("--hatexplain-test", required=True)
    parser.add_argument("--tweeteval-train", required=True)
    parser.add_argument("--tweeteval-val", required=True)
    parser.add_argument("--tweeteval-test", required=True)
    parser.add_argument("--hatexplain-base-model", required=True)
    parser.add_argument("--tweeteval-base-model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-5, 2e-5, 3e-5])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    summary_rows = []
    summary_rows.append(
        run_training(
            dataset_name="HateXplain_02_only",
            base_model_dir=resolve_model_ref(args.hatexplain_base_model),
            train_csv=os.path.abspath(args.hatexplain_train),
            val_csv=os.path.abspath(args.hatexplain_val),
            test_csv=os.path.abspath(args.hatexplain_test),
            output_root=output_root,
            batch_size=args.batch_size,
            epochs=args.epochs,
            max_length=args.max_length,
            learning_rates=args.learning_rates,
            seed=args.seed,
        )
    )
    summary_rows.append(
        run_training(
            dataset_name="TweetEval_02_only",
            base_model_dir=resolve_model_ref(args.tweeteval_base_model),
            train_csv=os.path.abspath(args.tweeteval_train),
            val_csv=os.path.abspath(args.tweeteval_val),
            test_csv=os.path.abspath(args.tweeteval_test),
            output_root=output_root,
            batch_size=args.batch_size,
            epochs=args.epochs,
            max_length=args.max_length,
            learning_rates=args.learning_rates,
            seed=args.seed,
        )
    )

    save_summary_csv(summary_rows, os.path.join(output_root, "training_results_summary.csv"))
    print(f"Saved summary: {os.path.join(output_root, 'training_results_summary.csv')}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
