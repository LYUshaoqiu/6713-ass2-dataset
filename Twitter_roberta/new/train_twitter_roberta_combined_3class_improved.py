from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# =========================================================
# 1. Basic settings
# =========================================================
MODEL_NAME = "cardiffnlp/twitter-roberta-base-hate"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Combined"
OUT_DIR = SCRIPT_DIR / "training_outputs" / "twitter_roberta_combined_3class_improved"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
SEED = 42
NUM_LABELS = 3
PATIENCE = 2
WARMUP_RATIO = 0.10
WEIGHT_STRENGTH = 0.25
MIN_F1_DELTA = 1e-4
LOSS_TIE_DELTA = 0.003

LABEL_NAMES = ["normal", "offensive", "hate"]
ID2LABEL = {i: label for i, label in enumerate(LABEL_NAMES)}
LABEL2ID = {label: i for i, label in ID2LABEL.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 2. Reproducibility
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# =========================================================
# 3. Dataset
# =========================================================
class TextDataset(Dataset):
    def __init__(self, csv_path: Path, tokenizer, max_length: int):
        df = pd.read_csv(csv_path)

        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{csv_path} must contain columns: text, label")

        df = df.dropna(subset=["text", "label"]).copy()
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(int)

        invalid = df[~df["label"].isin([0, 1, 2])]
        if len(invalid) > 0:
            raise ValueError(f"{csv_path} contains invalid labels outside 0/1/2")

        self.df = df.reset_index(drop=True)
        self.texts = self.df["text"].tolist()
        self.labels = self.df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"\nLoaded {csv_path.name}: {len(self.texts)} samples")
        print(self.df["label"].value_counts().sort_index())
        if "source" in self.df.columns:
            print("Source counts:")
            print(self.df["source"].value_counts())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# =========================================================
# 4. Class weights
# =========================================================
def compute_class_weights(csv_path: Path, num_labels: int):
    df = pd.read_csv(csv_path).dropna(subset=["text", "label"]).copy()
    labels = df["label"].astype(int).to_numpy()

    counts = np.bincount(labels, minlength=num_labels).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"Cannot compute class weights with empty class counts: {counts}")

    base_weights = len(labels) / (num_labels * counts)
    gentle_weights = 1.0 + WEIGHT_STRENGTH * (base_weights - 1.0)

    print("\nClass counts:", counts.tolist())
    print("Base class weights:", base_weights.tolist())
    print(f"Gentle class weights, strength={WEIGHT_STRENGTH}:", gentle_weights.tolist())

    return torch.tensor(gentle_weights, dtype=torch.float)


# =========================================================
# 5. Evaluation helpers
# =========================================================
def evaluate(model, dataloader, criterion):
    model.eval()

    all_labels = []
    all_preds = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            model_inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            outputs = model(**model_inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=-1)

            total_loss += loss.item()
            num_batches += 1
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / max(num_batches, 1)
    acc = accuracy_score(all_labels, all_preds)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "labels": all_labels,
        "preds": all_preds,
    }


def should_save_best(result, best_macro_f1: float, best_val_loss: float) -> bool:
    f1_gain = result["macro_f1"] - best_macro_f1
    if f1_gain > MIN_F1_DELTA:
        return True

    f1_is_tied = abs(f1_gain) <= LOSS_TIE_DELTA
    loss_is_better = result["loss"] < best_val_loss
    return f1_is_tied and loss_is_better


def write_split_report(split_name: str, result: dict, out_dir: Path) -> None:
    report = classification_report(
        result["labels"],
        result["preds"],
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(result["labels"], result["preds"], labels=[0, 1, 2])

    report_path = out_dir / f"{split_name}_classification_report.txt"
    cm_path = out_dir / f"{split_name}_confusion_matrix.csv"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    cm_df.to_csv(cm_path, encoding="utf-8-sig")

    print(f"\n{split_name.upper()} Classification Report:")
    print(report)
    print(f"{split_name.upper()} Confusion Matrix:")
    print(cm_df)
    print(f"Saved report to: {report_path}")
    print(f"Saved confusion matrix to: {cm_path}")


def write_source_metrics(test_df: pd.DataFrame, out_dir: Path) -> None:
    if "source" not in test_df.columns:
        return

    rows = []
    for source, group in test_df.groupby("source"):
        labels = group["label"].astype(int).tolist()
        preds = group["pred"].astype(int).tolist()
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="macro",
            zero_division=0,
        )
        rows.append(
            {
                "source": source,
                "n": len(group),
                "accuracy": accuracy_score(labels, preds),
                "macro_f1": macro_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
            }
        )

    source_metrics = pd.DataFrame(rows)
    source_path = out_dir / "test_metrics_by_source.csv"
    source_metrics.to_csv(source_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved source-level metrics to: {source_path}")
    print(source_metrics)


# =========================================================
# 6. Training
# =========================================================
def train():
    print("Using device:", device)
    print("Output dir:", OUT_DIR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    class_weights = compute_class_weights(TRAIN_PATH, NUM_LABELS).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_dataset = TextDataset(TRAIN_PATH, tokenizer, MAX_LENGTH)
    val_dataset = TextDataset(VAL_PATH, tokenizer, MAX_LENGTH)
    test_dataset = TextDataset(TEST_PATH, tokenizer, MAX_LENGTH)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_training_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    print(f"Total training steps: {total_training_steps}")
    print(f"Warmup steps: {warmup_steps}")

    history = []
    best_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_path = OUT_DIR / "best_model.pt"
    no_improve_count = 0

    config_path = OUT_DIR / "experiment_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "max_length": MAX_LENGTH,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "seed": SEED,
                "patience": PATIENCE,
                "warmup_ratio": WARMUP_RATIO,
                "weight_strength": WEIGHT_STRENGTH,
                "min_f1_delta": MIN_F1_DELTA,
                "loss_tie_delta": LOSS_TIE_DELTA,
            },
            f,
            indent=2,
        )

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'=' * 70}")

        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            labels = batch["labels"].to(device)
            model_inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            optimizer.zero_grad()
            outputs = model(**model_inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / max(num_batches, 1)
        val_result = evaluate(model, val_loader, criterion)

        row = {
            "Epoch": epoch,
            "Training Loss": avg_train_loss,
            "Validation Loss": val_result["loss"],
            "Accuracy": val_result["accuracy"],
            "Macro F1": val_result["macro_f1"],
            "Macro Precision": val_result["macro_precision"],
            "Macro Recall": val_result["macro_recall"],
            "Learning Rate": scheduler.get_last_lr()[0],
        }
        history.append(row)

        print(f"Training Loss   : {avg_train_loss:.6f}")
        print(f"Validation Loss : {val_result['loss']:.6f}")
        print(f"Accuracy        : {val_result['accuracy']:.6f}")
        print(f"Macro F1        : {val_result['macro_f1']:.6f}")
        print(f"Macro Precision : {val_result['macro_precision']:.6f}")
        print(f"Macro Recall    : {val_result['macro_recall']:.6f}")

        if should_save_best(val_result, best_macro_f1, best_val_loss):
            best_macro_f1 = val_result["macro_f1"]
            best_val_loss = val_result["loss"]
            best_epoch = epoch
            no_improve_count = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to: {best_model_path}")
        else:
            no_improve_count += 1
            print(f"No improvement count: {no_improve_count}/{PATIENCE}")

        if no_improve_count >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    history_df = pd.DataFrame(history)
    history_csv = OUT_DIR / "epoch_metrics.csv"
    history_df.to_csv(history_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved epoch history to: {history_csv}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model from epoch {best_epoch}")

    hf_dir = OUT_DIR / "best_hf_model"
    model.save_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)
    print(f"Saved HuggingFace model to: {hf_dir}")

    val_result = evaluate(model, val_loader, criterion)
    test_result = evaluate(model, test_loader, criterion)

    print(f"\n{'=' * 70}")
    print("TEST RESULTS (BEST MODEL)")
    print(f"{'=' * 70}")
    print(f"Best Epoch      : {best_epoch}")
    print(f"Test Loss       : {test_result['loss']:.6f}")
    print(f"Accuracy        : {test_result['accuracy']:.6f}")
    print(f"Macro F1        : {test_result['macro_f1']:.6f}")
    print(f"Macro Precision : {test_result['macro_precision']:.6f}")
    print(f"Macro Recall    : {test_result['macro_recall']:.6f}")

    write_split_report("val", val_result, OUT_DIR)
    write_split_report("test", test_result, OUT_DIR)

    test_df = pd.read_csv(TEST_PATH).dropna(subset=["text", "label"]).copy()
    test_df["pred"] = test_result["preds"]
    test_pred_csv = OUT_DIR / "test_predictions_best_model.csv"
    test_df.to_csv(test_pred_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved test predictions to: {test_pred_csv}")

    write_source_metrics(test_df, OUT_DIR)

    summary_path = OUT_DIR / "final_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
        f.write(f"Best Validation Macro F1: {best_macro_f1:.6f}\n")
        f.write(f"Test Loss: {test_result['loss']:.6f}\n")
        f.write(f"Accuracy: {test_result['accuracy']:.6f}\n")
        f.write(f"Macro F1: {test_result['macro_f1']:.6f}\n")
        f.write(f"Macro Precision: {test_result['macro_precision']:.6f}\n")
        f.write(f"Macro Recall: {test_result['macro_recall']:.6f}\n")
        f.write(f"Warmup Ratio: {WARMUP_RATIO:.2f}\n")
        f.write(f"Class Weight Strength: {WEIGHT_STRENGTH:.2f}\n")
    print(f"Saved final summary to: {summary_path}")


if __name__ == "__main__":
    train()
