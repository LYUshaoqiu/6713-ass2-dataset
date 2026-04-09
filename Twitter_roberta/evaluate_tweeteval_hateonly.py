from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

# 1. 基本设置
MODEL_NAME = "cardiffnlp/twitter-roberta-base-hate"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Tweeteval"

VAL_PATH = DATA_DIR / "val_hateonly.csv"
TEST_PATH = DATA_DIR / "test_hateonly.csv"

BATCH_SIZE = 256
MAX_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. 读取模型
print("Loading model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

print("Model loaded.")
print("id2label:", model.config.id2label)
print("label2id:", model.config.label2id)

# 3. 读取数据
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\nval shape:", val_df.shape)
print("test shape:", test_df.shape)

assert "text" in val_df.columns and "label" in val_df.columns
assert "text" in test_df.columns and "label" in test_df.columns

val_df = val_df.dropna(subset=["text", "label"]).copy()
test_df = test_df.dropna(subset=["text", "label"]).copy()

val_df["text"] = val_df["text"].astype(str)
test_df["text"] = test_df["text"].astype(str)

# 4. 批量预测
def predict_dataframe(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    texts = df["text"].tolist()
    preds = []
    prob_non_hate = []
    prob_hate = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Predicting {split_name}"):
        batch_texts = texts[i:i+BATCH_SIZE]

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )

        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            logits = model(**encodings).logits
            probs = F.softmax(logits, dim=-1)

        batch_preds = torch.argmax(probs, dim=-1).cpu().tolist()
        batch_probs = probs.cpu().tolist()

        preds.extend(batch_preds)
        prob_non_hate.extend([p[0] for p in batch_probs])
        prob_hate.extend([p[1] for p in batch_probs])

    out_df = df.copy()
    out_df["pred"] = preds
    out_df["prob_non_hate"] = prob_non_hate
    out_df["prob_hate"] = prob_hate
    return out_df

# 5. 评估函数
def evaluate_predictions(df: pd.DataFrame, split_name: str):
    y_true = df["label"].tolist()
    y_pred = df["pred"].tolist()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*70}")
    print(f"{split_name.upper()} RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix:")
    print(cm)

# 6. 运行
val_pred_df = predict_dataframe(val_df, "val")
test_pred_df = predict_dataframe(test_df, "test")

evaluate_predictions(val_pred_df, "val")
evaluate_predictions(test_pred_df, "test")

# 7. 保存结果
val_pred_path = DATA_DIR / "val_predictions_twitter_roberta_hateonly.csv"
test_pred_path = DATA_DIR / "test_predictions_twitter_roberta_hateonly.csv"

val_pred_df.to_csv(val_pred_path, index=False)
test_pred_df.to_csv(test_pred_path, index=False)

print("\nSaved prediction files:")
print(val_pred_path)
print(test_pred_path)