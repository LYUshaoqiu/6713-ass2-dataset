import os
import gc
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =========================
# 1. Paths
# =========================
HATEXPLAIN_PATH = "HateXPlain_data/"
TWEETEVAL_PATH = "Tweeteval三类分/"
SAVE_DIR = "bertweet_finetune_outputs"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# 2. Load data
# =========================
hx_train = pd.read_csv(HATEXPLAIN_PATH + "train.csv")
hx_val   = pd.read_csv(HATEXPLAIN_PATH + "val.csv")
hx_test  = pd.read_csv(HATEXPLAIN_PATH + "test.csv")

te_train = pd.read_csv(TWEETEVAL_PATH + "train.csv").rename(columns={"comment": "text"})
te_val   = pd.read_csv(TWEETEVAL_PATH + "val.csv").rename(columns={"comment": "text"})
te_test  = pd.read_csv(TWEETEVAL_PATH + "test.csv").rename(columns={"comment": "text"})

print(f"HateXPlain train:{len(hx_train)} val:{len(hx_val)} test:{len(hx_test)}")
print(f"TweetEval  train:{len(te_train)} val:{len(te_val)} test:{len(te_test)}")

LABEL_NAMES = ["Normal", "Offensive", "Hate"]

# =========================
# 3. Model / Tokenizer
# =========================
MODEL_NAME = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
print(f"Loaded tokenizer: {MODEL_NAME}")

# =========================
# 4. Dataset
# =========================
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# =========================
# 5. Train / Eval functions
# =========================
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0

    bar = tqdm(loader, desc="train", leave=False)
    for batch in bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)

def evaluate_model(model, loader):
    model.eval()

    all_preds = []
    all_labels = []
    all_losses = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        "loss": round(float(np.mean(all_losses)), 6),
        "accuracy": round(accuracy_score(all_labels, all_preds), 6),
        "macro_f1": round(f1_score(all_labels, all_preds, average="macro"), 6),
        "macro_precision": round(precision_score(all_labels, all_preds, average="macro", zero_division=0), 6),
        "macro_recall": round(recall_score(all_labels, all_preds, average="macro", zero_division=0), 6),
    }

    return all_preds, all_labels, metrics

# =========================
# 6. Fine-tune function
# =========================
LR_CANDIDATES = [1e-5, 2e-5, 3e-5]
BATCH_SIZE = 8
EPOCHS = 3

def run_finetune(train_df, val_df, test_df, dataset_name):
    print(f"\n========== {dataset_name} ==========")

    train_dataset = HateSpeechDataset(train_df["text"], train_df["label"], tokenizer)
    val_dataset   = HateSpeechDataset(val_df["text"], val_df["label"], tokenizer)
    test_dataset  = HateSpeechDataset(test_df["text"], test_df["label"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_lr = None
    best_val_f1 = -1
    best_state_dict = None
    best_history = None

    for lr in LR_CANDIDATES:
        print(f"\nTrying learning rate = {lr}")

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=3
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        history = []

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")

            train_loss = train_epoch(model, train_loader, optimizer)
            _, _, val_metrics = evaluate_model(model, val_loader)

            row = {
                "epoch": epoch + 1,
                "learning_rate": lr,
                "train_loss": round(train_loss, 6),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"]
            }
            history.append(row)

            print(
                f"train_loss={train_loss:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

        current_best_f1 = max(x["val_macro_f1"] for x in history)

        if current_best_f1 > best_val_f1:
            best_val_f1 = current_best_f1
            best_lr = lr
            best_history = history
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # save training history
    history_df = pd.DataFrame(best_history)
    history_path = os.path.join(SAVE_DIR, f"{dataset_name.lower()}_training_history.csv")
    history_df.to_csv(history_path, index=False)

    print(f"\nBest learning rate for {dataset_name}: {best_lr}")
    print(f"Best validation macro F1: {best_val_f1:.4f}")
    print(f"Saved training history: {history_path}")

    # rebuild best model
    best_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )
    best_model.load_state_dict(best_state_dict)
    best_model = best_model.to(device)

    # save best model
    model_path = os.path.join(SAVE_DIR, f"{dataset_name.lower()}_best_model.pt")
    torch.save(best_model.state_dict(), model_path)
    print(f"Saved model: {model_path}")

    # test evaluation
    test_preds, test_labels, test_metrics = evaluate_model(best_model, test_loader)

    print(f"\n=== {dataset_name} Test Results ===")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print()
    print(classification_report(test_labels, test_preds, target_names=LABEL_NAMES))

    # save classification report
    report_dict = classification_report(
        test_labels, test_preds,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(SAVE_DIR, f"{dataset_name.lower()}_classification_report.csv")
    report_df.to_csv(report_path)
    print(f"Saved classification report: {report_path}")

    # confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES
    )
    plt.title(f"BERTweet Fine-tuned\n{dataset_name} Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    cm_path = os.path.join(SAVE_DIR, f"{dataset_name.lower()}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {cm_path}")

    result_row = {
        "Model": "BERTweet fine-tuned",
        "Dataset": dataset_name,
        "Best LR": best_lr,
        "Accuracy": test_metrics["accuracy"],
        "Macro F1": test_metrics["macro_f1"],
        "Macro Precision": test_metrics["macro_precision"],
        "Macro Recall": test_metrics["macro_recall"]
    }

    return result_row

# =========================
# 7. Run both datasets
# =========================
all_results = []

all_results.append(run_finetune(hx_train, hx_val, hx_test, "HateXPlain"))
all_results.append(run_finetune(te_train, te_val, te_test, "TweetEval"))

results_df = pd.DataFrame(all_results)
results_path = os.path.join(SAVE_DIR, "bertweet_finetune_results.csv")
results_df.to_csv(results_path, index=False)

print("\n========== Final Summary ==========")
print(results_df.to_string(index=False))
print(f"\nSaved final results: {results_path}")