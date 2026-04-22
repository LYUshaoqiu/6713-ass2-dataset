from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.tools import tool

# =========================================================
# Config
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_NAME = "cardiffnlp/twitter-roberta-base-hate"
MODEL_WEIGHTS = SCRIPT_DIR / "training_outputs" / "twitter_roberta_combined_3class_improved" / "best_model.pt"

LABEL_NAMES = ["normal", "offensive", "hate"]
MAX_LENGTH = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Load model once
# =========================================================
print(f"Loading model from: {MODEL_WEIGHTS}")
print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    ignore_mismatched_sizes=True,
)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# =========================================================
# Core prediction function
# =========================================================
def predict_text(text: str) -> dict:
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    inputs = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        pred_idx = int(torch.argmax(logits, dim=-1).item())

    result = {
        "text": text,
        "label": LABEL_NAMES[pred_idx],
        "confidence": round(probs[pred_idx], 4),
        "probabilities": {
            "normal": round(probs[0], 4),
            "offensive": round(probs[1], 4),
            "hate": round(probs[2], 4),
        },
    }
    return result


# =========================================================
# LangChain Tool
# =========================================================
@tool
def classify_hate_speech(text: str) -> str:
    """
    Classify input text into one of three labels:
    normal, offensive, or hate.
    Returns predicted label, confidence, and class probabilities.
    """
    result = predict_text(text)
    return json.dumps(result, ensure_ascii=False, indent=2)


# =========================================================
# Simple demo when run directly
# =========================================================
if __name__ == "__main__":
    demo_texts = [
        "I hate all minorities",
        "You are stupid and useless",
        "I disagree with your opinion",
    ]

    print("\n=== LangChain Tool Demo ===")
    for text in demo_texts:
        print(f"\nInput: {text}")
        output = classify_hate_speech.invoke(text)
        print(output)