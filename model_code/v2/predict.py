"""
predict.py — Hate Speech Classifier CLI
Usage:
  python predict.py --text "your text here"
  python predict.py --file input.txt
  python predict.py --text "text" --model saved_model_roberta_hx_course

Available models:
  saved_model_roberta_hx_course  (default) RoBERTa-HateXplain + Course Domain Adaptation
  saved_model_bert_hx_course              BERT-HateXplain + Course Domain Adaptation
  saved_model_roberta_hx                  RoBERTa-HateXplain (social media domain)
  saved_model_bert_hx                     BERT-HateXplain (social media domain)
  saved_model_roberta_te                  RoBERTa-TweetEval (social media domain)
  saved_model_bert_te                     BERT-TweetEval (social media domain)
"""
import argparse
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Default model: best model for course domain ────────────────
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), 'saved_model_roberta_hx_course')

LABELS = ['Normal', 'Offensive', 'Hate']

def load_model(model_dir):
    if not os.path.isdir(model_dir):
        print(f"Error: model directory not found: {model_dir}", file=sys.stderr)
        print("Please run step2_roberta_domain_adapt.py first to train and save the model.",
              file=sys.stderr)
        sys.exit(1)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return tokenizer, model, device

def predict(texts, tokenizer, model, device, max_len=128):
    results = []
    for text in texts:
        text = str(text).strip()
        if not text:
            continue
        enc = tokenizer(text, padding='max_length', truncation=True,
                        max_length=max_len, return_tensors='pt')
        input_ids      = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs     = F.softmax(logits, dim=-1).squeeze()
        pred_idx  = probs.argmax().item()
        confidence = probs[pred_idx].item()
        results.append({
            'text':       text,
            'label':      LABELS[pred_idx],
            'confidence': confidence,
            'probs':      {LABELS[i]: round(probs[i].item(), 4) for i in range(len(LABELS))}
        })
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Hate Speech Classifier (RoBERTa-HateXplain + Course Domain Adaptation)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', type=str,
                       help='Single text string to classify')
    group.add_argument('--file', type=str,
                       help='Path to a text file (one sentence per line)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='Path to saved model directory (default: saved_model_roberta_hx_course)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show per-class probabilities')
    args = parser.parse_args()

    # Resolve model path relative to script dir if not absolute
    if not os.path.isabs(args.model):
        args.model = os.path.join(os.path.dirname(__file__), args.model)

    print(f"Loading model from: {args.model}", file=sys.stderr)
    tokenizer, model, device = load_model(args.model)
    print(f"Device: {device}", file=sys.stderr)

    if args.text:
        texts = [args.text]
    else:
        if not os.path.isfile(args.file):
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(texts)} lines from {args.file}", file=sys.stderr)

    results = predict(texts, tokenizer, model, device)

    print()
    for r in results:
        print(f"Text:       {r['text']}")
        print(f"Prediction: {r['label']}  (confidence: {r['confidence']:.4f})")
        if args.verbose:
            for lbl, prob in r['probs'].items():
                print(f"  {lbl:10s}: {prob:.4f}")
        print()

if __name__ == '__main__':
    main()
