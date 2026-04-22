"""
Zero-Shot Evaluation — 4 Pre-trained Models
Models (no fine-tuning, used as encoders):
  1. BERT-base-uncased         (bert-base-uncased)
  2. HateBERT                  (GroNLP/hateBERT)
  3. Twitter-RoBERTa-base-hate (cardiffnlp/twitter-roberta-base-hate)
  4. BERTweet                  (vinai/bertweet-base)

Method: [CLS] cosine similarity against label prototype embeddings
        — no gradient updates, no task-specific fine-tuning.

Outputs:
  zeroshot_all_results.csv          — combined metrics table
  zeroshot_<model>_<dataset>_cm.png — confusion matrix per model × dataset
"""
import os, gc
os.environ['CUDA_LAUNCH_BLOCKING']    = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import warnings
warnings.filterwarnings("ignore")
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.enabled = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if device.type == 'cuda':
    print(f'GPU : {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

# ── Datasets ──────────────────────────────────────────────────
HX_DIR = '../dataset/6713-ass2-dataset/HateXPlain_data/'
TE_DIR = '../dataset/6713-ass2-dataset/Tweeteval三类分/'

hx_test = pd.read_csv(HX_DIR + 'test.csv')
te_test = pd.read_csv(TE_DIR + 'test.csv').rename(columns={'comment': 'text'})
for df in [hx_test, te_test]:
    df.dropna(subset=['text', 'label'], inplace=True)
    df.reset_index(drop=True, inplace=True)

DATASETS = {
    'HateXPlain': hx_test,
    'TweetEval':  te_test,
}
LABELS = ['Normal', 'Offensive', 'Hate']
print(f'HateXPlain test: {len(hx_test)}  |  TweetEval test: {len(te_test)}')

# ── Label prototype sentences ──────────────────────────────────
# Same set used for every model to ensure fair comparison
LABEL_DESCRIPTIONS = [
    "This is a normal, neutral, or positive comment.",          # 0 Normal
    "This is an offensive or rude comment using bad language.", # 1 Offensive
    "This is hate speech targeting a person or specific group." # 2 Hate
]

# ── Model registry ─────────────────────────────────────────────
# (display_name, hf_model_id, tokenizer_extra_kwargs, model_extra_kwargs)
MODELS = [
    (
        'BERT-base-uncased',
        'bert-base-uncased',
        {},                                          # tokenizer kwargs
        {'attn_implementation': 'eager'},            # model kwargs
    ),
    (
        'HateBERT',
        'GroNLP/hateBERT',
        {},
        {'attn_implementation': 'eager'},
    ),
    (
        'Twitter-RoBERTa-base-hate',
        'cardiffnlp/twitter-roberta-base-hate',
        {},
        {},                                          # RoBERTa — no eager needed
    ),
    (
        'BERTweet',
        'vinai/bertweet-base',
        {'use_fast': False},                         # BERTweet requires slow tokenizer
        {},
    ),
]

# ── CLS embedding helper ───────────────────────────────────────
def get_cls_embeddings(texts, tokenizer, model, batch_size=32):
    """Return normalised [CLS] embeddings for a list of strings."""
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size),
                  desc='  embed', leave=False, ncols=80):
        batch = [str(t) for t in texts[i:i + batch_size]]
        enc   = tokenizer(batch, padding=True, truncation=True,
                          max_length=128, return_tensors='pt').to(device)
        # RoBERTa / BERTweet do not use token_type_ids
        enc = {k: v for k, v in enc.items() if k != 'token_type_ids'
               or 'token_type_ids' in tokenizer.model_input_names}
        with torch.no_grad():
            out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu()
        all_embs.append(cls)
    embs = torch.cat(all_embs, dim=0)
    return F.normalize(embs, dim=1)

# ── Zero-shot predict ──────────────────────────────────────────
def zeroshot_predict(texts, tokenizer, model, label_embs):
    text_embs = get_cls_embeddings(texts, tokenizer, model)
    sims      = text_embs @ label_embs.T       # (N, 3)
    return sims.argmax(dim=1).numpy()

# ── Safe filename ──────────────────────────────────────────────
def safe(s):
    return s.replace(' ', '_').replace('/', '-').replace('(', '').replace(')', '')

# ── Main loop ──────────────────────────────────────────────────
all_rows = []

for display_name, model_id, tok_kwargs, mdl_kwargs in MODELS:
    print(f'\n{"="*60}')
    print(f'Model : {display_name}')
    print(f'HF ID : {model_id}')
    print('='*60)

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    model     = AutoModel.from_pretrained(model_id, **mdl_kwargs).to(device)
    model.eval()
    print(f'  Loaded on {device}')

    # Pre-compute label embeddings
    label_embs = get_cls_embeddings(LABEL_DESCRIPTIONS, tokenizer, model, batch_size=16)
    print(f'  Label embeddings: {label_embs.shape}')

    for ds_name, df in DATASETS.items():
        print(f'\n  ── {ds_name} ({len(df)} samples) ──')
        texts  = df['text'].fillna('').tolist()
        labels = df['label'].values

        preds = zeroshot_predict(texts, tokenizer, model, label_embs)

        acc    = round(accuracy_score(labels, preds), 6)
        mf1    = round(f1_score(labels, preds, average='macro',    zero_division=0), 6)
        mprec  = round(precision_score(labels, preds, average='macro', zero_division=0), 6)
        mrec   = round(recall_score(labels, preds, average='macro',    zero_division=0), 6)
        wf1    = round(f1_score(labels, preds, average='weighted', zero_division=0), 6)
        f1_per = f1_score(labels, preds, average=None, zero_division=0)

        print(classification_report(labels, preds, target_names=LABELS))

        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=LABELS, yticklabels=LABELS)
        plt.title(f'{display_name} — Zero-Shot\n{ds_name} Test Set')
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.tight_layout()
        fname = f'zeroshot_{safe(display_name)}_{ds_name}_cm.png'
        plt.savefig(fname, dpi=150); plt.close()
        print(f'  Saved: {fname}')

        all_rows.append({
            'Model':           display_name,
            'HF Model ID':     model_id,
            'Dataset':         ds_name,
            'Accuracy':        acc,
            'Macro F1':        mf1,
            'Macro Precision': mprec,
            'Macro Recall':    mrec,
            'Weighted F1':     wf1,
            'F1 Normal':       round(f1_per[0], 6),
            'F1 Offensive':    round(f1_per[1], 6),
            'F1 Hate':         round(f1_per[2], 6),
        })

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# ── Combined results table ─────────────────────────────────────
print('\n' + '='*90)
print('=== Zero-Shot Results — All Models ===')
print('='*90)
results_df = pd.DataFrame(all_rows)
print(results_df.to_string(index=False))
results_df.to_csv('zeroshot_all_results.csv', index=False)
print('\nSaved: zeroshot_all_results.csv')

# ── Per-dataset summary ────────────────────────────────────────
for ds in ['HateXPlain', 'TweetEval']:
    sub = results_df[results_df['Dataset'] == ds][
        ['Model', 'Accuracy', 'Macro F1', 'F1 Normal', 'F1 Offensive', 'F1 Hate']
    ]
    print(f'\n── {ds} ──')
    print(sub.to_string(index=False))
