"""
step5_predict.py — Hate Speech Classifier CLI

Usage:
  python step5_predict.py --text "your text here"
  python step5_predict.py --file input.txt
  python step5_predict.py --text "text" --model models/bert_da_course
  python step5_predict.py --text "text" --model models/roberta_HateXPlain --verbose

Available models (run the corresponding step first to generate them):
  models/roberta_da_course      (default) RoBERTa-HateXPlain + Domain Adaptation  [step3]
  models/bert_da_course                  BERT-HateXPlain + Domain Adaptation       [step2]
  models/roberta_HateXPlain              RoBERTa fine-tuned on HateXPlain          [step1]
  models/roberta_TweetEval               RoBERTa fine-tuned on TweetEval           [step1]
  models/roberta_combined                RoBERTa fine-tuned on combined dataset    [step1]
  models/bert_HateXPlain                 BERT fine-tuned on HateXPlain             [step1]
  models/bert_TweetEval                  BERT fine-tuned on TweetEval              [step1]
  models/bert_combined                   BERT fine-tuned on combined dataset       [step1]
  models/hatebert_HateXPlain             HateBERT fine-tuned on HateXPlain         [step1]
  models/hatebert_TweetEval              HateBERT fine-tuned on TweetEval          [step1]
  models/bertweet_HateXPlain             BERTweet fine-tuned on HateXPlain         [step1]
  models/bertweet_TweetEval              BERTweet fine-tuned on TweetEval          [step1]
"""
import argparse
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Default model: best model for course domain ────────────────
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), 'models', 'roberta_da_course')

LABELS = ['Normal', 'Offensive', 'Hate']


def load_model(model_dir):
    if not os.path.isdir(model_dir):
        print(f"Error: model directory not found: {model_dir}", file=sys.stderr)
        print("Please run the corresponding training step first:", file=sys.stderr)
        print("  step1_finetune_all.py  — fine-tuned base models", file=sys.stderr)
        print("  step2_bert_domain_adapt.py    — BERT domain adaptation", file=sys.stderr)
        print("  step3_roberta_domain_adapt.py — RoBERTa domain adaptation", file=sys.stderr)
        sys.exit(1)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)
    return tokenizer, model, device


def predict(texts, tokenizer, model, device, max_len=128):
    results = []
    for text in texts:
        text = str(text).strip()
        if not text:
            continue
        enc = tokenizer(text, padding='max_length', truncation=True,
                        max_length=max_len, return_tensors='pt')
        kwargs = {
            'input_ids':      enc['input_ids'].to(device),
            'attention_mask': enc['attention_mask'].to(device),
        }
        if 'token_type_ids' in enc:
            kwargs['token_type_ids'] = enc['token_type_ids'].to(device)
        with torch.no_grad():
            logits = model(**kwargs).logits
        probs      = F.softmax(logits, dim=-1).squeeze()
        pred_idx   = probs.argmax().item()
        confidence = probs[pred_idx].item()
        results.append({
            'text':       text,
            'label':      LABELS[pred_idx],
            'confidence': confidence,
            'probs':      {LABELS[i]: round(probs[i].item(), 4) for i in range(len(LABELS))},
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Hate Speech Classifier CLI — COMP6713 Group Project')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', type=str,
                       help='Single text string to classify')
    group.add_argument('--file', type=str,
                       help='Path to a text file (one sentence per line)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='Path to saved model directory '
                             '(default: models/roberta_da_course)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show per-class probabilities for each prediction')
    args = parser.parse_args()

    # Resolve model path relative to script dir if not absolute
    if not os.path.isabs(args.model):
        args.model = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model)

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
