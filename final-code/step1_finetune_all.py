"""
Unified Fine-Tuning — 4 Models × 2 Datasets  (no Domain Adaptation)
======================================================================
Models:
  1. BERT-base-uncased         (bert-base-uncased)
  2. HateBERT                  (GroNLP/hateBERT)
  3. Twitter-RoBERTa-base-hate (cardiffnlp/twitter-roberta-base-hate)
  4. BERTweet                  (vinai/bertweet-base)

Datasets: HateXPlain, TweetEval  (3-class: 0=Normal / 1=Offensive / 2=Hate)

Training setup:
  - LR search: {1e-5, 2e-5, 3e-5}
  - Optimizer: ManualAdamW (weight_decay=0.01) — bypasses PyTorch 2.6 fused CUDA bug
  - Batch size: 8  |  Epochs per LR: 3  |  Max sequence length: 128
  - Best LR selected by highest val macro-F1 across all epochs

Outputs (saved under results/ and models/):
  results/finetune_all_results.csv              — combined test-set metrics (8 rows)
  results/<tag>_<dataset>_history.csv           — full training history (all LRs)
  results/<tag>_<dataset>_cm.png                — confusion matrix heatmaps
  models/<tag>_<dataset>/                       — HuggingFace checkpoint for inference
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
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
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

# ── Output directories ─────────────────────────────────────────
os.makedirs('results', exist_ok=True)
os.makedirs('models',  exist_ok=True)

# ── Dataset paths ──────────────────────────────────────────────
HX_DIR = '../dataset/6713-ass2-dataset/HateXPlain_data/'
TE_DIR = '../dataset/6713-ass2-dataset/Tweeteval三类分/'

hx_train = pd.read_csv(HX_DIR + 'train.csv')
hx_val   = pd.read_csv(HX_DIR + 'val.csv')
hx_test  = pd.read_csv(HX_DIR + 'test.csv')

te_train = pd.read_csv(TE_DIR + 'train.csv').rename(columns={'comment': 'text'})
te_val   = pd.read_csv(TE_DIR + 'val.csv').rename(columns={'comment': 'text'})
te_test  = pd.read_csv(TE_DIR + 'test.csv').rename(columns={'comment': 'text'})

for name, df in [('hx_train', hx_train), ('hx_val', hx_val), ('hx_test', hx_test),
                 ('te_train', te_train), ('te_val', te_val), ('te_test', te_test)]:
    df.dropna(subset=['text', 'label'], inplace=True)
    df.reset_index(drop=True, inplace=True)

print(f'HateXPlain  train:{len(hx_train)} val:{len(hx_val)} test:{len(hx_test)}')
print(f'TweetEval   train:{len(te_train)} val:{len(te_val)} test:{len(te_test)}')

DATASETS = {
    'HateXPlain': (hx_train, hx_val, hx_test),
    'TweetEval':  (te_train, te_val, te_test),
}
CLASS_LABELS = ['Normal', 'Offensive', 'Hate']

# ── Model registry ─────────────────────────────────────────────
# (display_name, hf_model_id, file_tag, tokenizer_kwargs, model_kwargs)
# Notes:
#   - BERT/HateBERT need attn_implementation='eager' to avoid fused attention bug
#   - BERTweet needs use_fast=False for its tweet-specific BPE tokenizer
#   - RoBERTa and BERTweet do not use token_type_ids
MODELS = [
    ('BERT-base-uncased',
     'bert-base-uncased',
     'bert',
     {},
     {'attn_implementation': 'eager'}),

    ('HateBERT',
     'GroNLP/hateBERT',
     'hatebert',
     {},
     {'attn_implementation': 'eager'}),

    ('Twitter-RoBERTa-base-hate',
     'cardiffnlp/twitter-roberta-base-hate',
     'roberta',
     {},
     {}),

    ('BERTweet',
     'vinai/bertweet-base',
     'bertweet',
     {'use_fast': False},
     {}),
]

# ── ManualAdamW ────────────────────────────────────────────────
# Bypasses the fused AdamW CUDA kernel bug in PyTorch 2.6 on RTX cards.
class ManualAdamW:
    def __init__(self, params, lr=2e-5, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr  = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd  = weight_decay
        self.m   = [torch.zeros_like(p.data) for p in self.params]
        self.v   = [torch.zeros_like(p.data) for p in self.params]
        self.t   = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            p.mul_(1.0 - self.lr * self.wd)
            self.m[i].mul_(self.b1).add_(g, alpha=1.0 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
            p.addcdiv_(self.m[i] / bc1,
                       (self.v[i] / bc2).sqrt().add_(self.eps),
                       value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# ── Dataset ────────────────────────────────────────────────────
# Automatically includes token_type_ids only for BERT-family tokenizers.
class HateDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.use_tti   = 'token_type_ids' in tokenizer.model_input_names

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
        )
        item = {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.use_tti:
            item['token_type_ids'] = enc['token_type_ids'].squeeze()
        return item


# ── Helpers ────────────────────────────────────────────────────
LR_CANDIDATES = [1e-5, 2e-5, 3e-5]
BATCH_SIZE    = 8
EPOCHS        = 3


def build_kwargs(batch):
    """Build model forward kwargs, conditionally adding token_type_ids."""
    kw = {
        'input_ids':      batch['input_ids'].to(device),
        'attention_mask': batch['attention_mask'].to(device),
    }
    if 'token_type_ids' in batch:
        kw['token_type_ids'] = batch['token_type_ids'].to(device)
    return kw


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    bar = tqdm(loader, desc='  train', leave=False, ncols=90)
    for batch in bar:
        optimizer.zero_grad()
        kw         = build_kwargs(batch)
        kw['labels'] = batch['label'].to(device)
        out        = model(**kw)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += out.loss.item()
        bar.set_postfix(loss=f'{out.loss.item():.3f}')
    return total_loss / len(loader)


def evaluate_loader(model, loader):
    """Evaluate on a DataLoader; return (preds, labels, metrics_dict)."""
    model.eval()
    preds_all, labels_all, losses = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='  eval ', leave=False, ncols=90):
            labs = batch['label'].to(device)
            kw   = build_kwargs(batch)
            kw['labels'] = labs
            out  = model(**kw)
            losses.append(out.loss.item())
            preds_all.extend(out.logits.argmax(dim=-1).cpu().numpy())
            labels_all.extend(batch['label'].numpy())
    preds  = np.array(preds_all)
    labels = np.array(labels_all)
    return preds, labels, {
        'val_loss':        round(float(np.mean(losses)), 6),
        'accuracy':        round(accuracy_score(labels, preds), 6),
        'macro_f1':        round(f1_score(labels, preds, average='macro'), 6),
        'macro_precision': round(precision_score(labels, preds,
                                                 average='macro', zero_division=0), 6),
        'macro_recall':    round(recall_score(labels, preds,
                                              average='macro', zero_division=0), 6),
    }


def safe(s):
    return s.replace(' ', '_').replace('/', '-').replace('(', '').replace(')', '')


# ── Fine-tuning ────────────────────────────────────────────────
def run_finetune(train_df, val_df, model_id, mdl_kwargs, tokenizer,
                 display_name, ds_name, model_tag):
    """
    Run LR search, return best model state + full training history.
    History contains one row per epoch for every LR tried.
    """
    all_history   = []   # accumulates all LR × epoch rows
    best_lr       = None
    best_val_f1   = -1.0
    best_state    = None

    for lr in LR_CANDIDATES:
        print(f'    lr={lr}', flush=True)
        ft_model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=3, ignore_mismatched_sizes=True, **mdl_kwargs
        ).to(device)
        optimizer    = ManualAdamW(ft_model.parameters(), lr=lr, weight_decay=0.01)
        train_loader = DataLoader(
            HateDataset(train_df['text'], train_df['label'], tokenizer),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            HateDataset(val_df['text'], val_df['label'], tokenizer),
            batch_size=BATCH_SIZE, num_workers=0,
        )

        lr_best_f1    = -1.0
        lr_best_state = None

        for ep in range(EPOCHS):
            train_loss = train_epoch(ft_model, train_loader, optimizer)
            _, _, m    = evaluate_loader(ft_model, val_loader)
            row = {
                'Model':           display_name,
                'Dataset':         ds_name,
                'LR':              lr,
                'Epoch':           ep + 1,
                'Training Loss':   round(train_loss, 6),
                'Validation Loss': m['val_loss'],
                'Accuracy':        m['accuracy'],
                'Macro F1':        m['macro_f1'],
                'Macro Precision': m['macro_precision'],
                'Macro Recall':    m['macro_recall'],
            }
            all_history.append(row)
            print(f'      epoch {ep+1}/{EPOCHS}  '
                  f'train_loss={train_loss:.4f}  val_f1={m["macro_f1"]:.4f}', flush=True)
            if m['macro_f1'] > lr_best_f1:
                lr_best_f1    = m['macro_f1']
                lr_best_state = {k: v.cpu().clone()
                                 for k, v in ft_model.state_dict().items()}
            if device.type == 'cuda':
                torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if lr_best_f1 > best_val_f1:
            best_val_f1 = lr_best_f1
            best_lr     = lr
            best_state  = lr_best_state

        del ft_model; gc.collect(); torch.cuda.empty_cache()

    print(f'  -> Best LR={best_lr}  best_val_macro_f1={best_val_f1:.4f}')
    return best_state, best_lr, best_val_f1, all_history


# ── Test evaluation + checkpoint saving ────────────────────────
def eval_and_save(best_state, test_df, model_id, mdl_kwargs, tokenizer,
                  display_name, ds_name, model_tag):
    model_ft = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=3, ignore_mismatched_sizes=True, **mdl_kwargs
    )
    model_ft.load_state_dict(best_state)
    model_ft = model_ft.to(device)

    test_loader = DataLoader(
        HateDataset(test_df['text'], test_df['label'], tokenizer),
        batch_size=BATCH_SIZE, num_workers=0,
    )
    preds, labels, m = evaluate_loader(model_ft, test_loader)

    print(f'\n── {display_name} | {ds_name} — Test Results ──')
    print(classification_report(labels, preds, target_names=CLASS_LABELS))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.title(f'{display_name} — Fine-tuned\n{ds_name} Test Set')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    cm_path = f'results/{model_tag}_{safe(ds_name)}_cm.png'
    plt.savefig(cm_path, dpi=150); plt.close()
    print(f'  CM saved : {cm_path}')

    # Save HuggingFace checkpoint
    save_dir = f'models/{model_tag}_{safe(ds_name)}'
    os.makedirs(save_dir, exist_ok=True)
    model_ft.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f'  Model saved : {save_dir}')

    del model_ft; gc.collect(); torch.cuda.empty_cache()

    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    return {
        'Model':           display_name,
        'Dataset':         ds_name,
        'Accuracy':        m['accuracy'],
        'Macro F1':        m['macro_f1'],
        'Macro Precision': m['macro_precision'],
        'Macro Recall':    m['macro_recall'],
        'Weighted F1':     round(f1_score(labels, preds, average='weighted'), 6),
        'F1 Normal':       round(per_class_f1[0], 6),
        'F1 Offensive':    round(per_class_f1[1], 6),
        'F1 Hate':         round(per_class_f1[2], 6),
    }


# ── Main loop ──────────────────────────────────────────────────
all_test_rows = []

for display_name, model_id, model_tag, tok_kwargs, mdl_kwargs in MODELS:
    print(f'\n{"="*70}')
    print(f'Model : {display_name}')
    print(f'HF ID : {model_id}')
    print('='*70)

    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    has_tti   = 'token_type_ids' in tokenizer.model_input_names
    print(f'  Tokenizer loaded.  token_type_ids: {"yes" if has_tti else "no"}')

    for ds_name, (train_df, val_df, test_df) in DATASETS.items():
        print(f'\n  ── {ds_name}  '
              f'train:{len(train_df)} val:{len(val_df)} test:{len(test_df)} ──')

        best_state, best_lr, best_val_f1, history = run_finetune(
            train_df, val_df, model_id, mdl_kwargs, tokenizer,
            display_name, ds_name, model_tag,
        )

        # Save full training history (all LRs × epochs)
        hist_path = f'results/{model_tag}_{safe(ds_name)}_history.csv'
        pd.DataFrame(history).to_csv(hist_path, index=False)
        print(f'  History saved : {hist_path}')

        print(f'\n  ── Test Evaluation ({ds_name}) ──')
        row = eval_and_save(
            best_state, test_df, model_id, mdl_kwargs, tokenizer,
            display_name, ds_name, model_tag,
        )
        all_test_rows.append(row)

    del tokenizer; gc.collect(); torch.cuda.empty_cache()

# ── Aggregate results ──────────────────────────────────────────
results_df = pd.DataFrame(all_test_rows)

print('\n' + '='*90)
print('=== Fine-Tuning Results — All 4 Models × 2 Datasets — Test Set ===')
print('='*90)
print(results_df.to_string(index=False))
results_df.to_csv('results/finetune_all_results.csv', index=False)
print('\nSaved: results/finetune_all_results.csv')

# ── Per-dataset summary ─────────────────────────────────────────
for ds in ['HateXPlain', 'TweetEval']:
    sub = results_df[results_df['Dataset'] == ds][
        ['Model', 'Accuracy', 'Macro F1', 'F1 Normal', 'F1 Offensive', 'F1 Hate']
    ]
    print(f'\n── {ds} ──')
    print(sub.to_string(index=False))


# ══════════════════════════════════════════════════════════════════
#  PART 2 — Combined Dataset Training
#  Train BERT-base-uncased and Twitter-RoBERTa-base-hate on
#  HateXPlain + TweetEval merged, then evaluate on each test set
#  separately so results are directly comparable to Part 1.
# ══════════════════════════════════════════════════════════════════

print(f'\n\n{"#"*70}')
print('#  COMBINED DATASET TRAINING  (BERT-base + Twitter-RoBERTa)')
print(f'{"#"*70}')

# Build combined splits (shuffle train to mix both sources)
combined_train = pd.concat([hx_train, te_train], ignore_index=True) \
                   .sample(frac=1, random_state=42).reset_index(drop=True)
combined_val   = pd.concat([hx_val,   te_val],   ignore_index=True) \
                   .reset_index(drop=True)

print(f'Combined  train:{len(combined_train)}  val:{len(combined_val)}')

COMBINED_MODELS = [
    ('BERT-base-uncased',
     'bert-base-uncased',
     'bert_combined',
     {},
     {'attn_implementation': 'eager'}),

    ('Twitter-RoBERTa-base-hate',
     'cardiffnlp/twitter-roberta-base-hate',
     'roberta_combined',
     {},
     {}),
]

combined_test_rows = []

for display_name, model_id, model_tag, tok_kwargs, mdl_kwargs in COMBINED_MODELS:
    print(f'\n{"="*70}')
    print(f'Model : {display_name}  [Combined training]')
    print(f'HF ID : {model_id}')
    print('='*70)

    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)

    # ── Train on combined ────────────────────────────────────────
    best_state, best_lr, best_val_f1, history = run_finetune(
        combined_train, combined_val, model_id, mdl_kwargs, tokenizer,
        display_name, 'HateXPlain+TweetEval', model_tag,
    )

    # Save training history
    hist_path = f'results/{model_tag}_history.csv'
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f'  History saved : {hist_path}')

    # ── Save model checkpoint once ───────────────────────────────
    model_ft = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=3, ignore_mismatched_sizes=True, **mdl_kwargs
    )
    model_ft.load_state_dict(best_state)
    model_ft = model_ft.to(device)

    save_dir = f'models/{model_tag}'
    os.makedirs(save_dir, exist_ok=True)
    model_ft.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f'  Model saved : {save_dir}')

    # ── Evaluate on each test set separately ─────────────────────
    for test_name, test_df in [('HateXPlain', hx_test), ('TweetEval', te_test)]:
        print(f'\n  ── Test on {test_name} ({len(test_df)} samples) ──')
        test_loader = DataLoader(
            HateDataset(test_df['text'], test_df['label'], tokenizer),
            batch_size=BATCH_SIZE, num_workers=0,
        )
        preds, labels, m = evaluate_loader(model_ft, test_loader)
        print(classification_report(labels, preds, target_names=CLASS_LABELS))

        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
        plt.title(f'{display_name} — Combined Training\nTest: {test_name}')
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.tight_layout()
        cm_path = f'results/{model_tag}_test_{safe(test_name)}_cm.png'
        plt.savefig(cm_path, dpi=150); plt.close()
        print(f'  CM saved : {cm_path}')

        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        combined_test_rows.append({
            'Model':           display_name,
            'Train_Dataset':   'HateXPlain+TweetEval',
            'Test_Dataset':    test_name,
            'Accuracy':        m['accuracy'],
            'Macro F1':        m['macro_f1'],
            'Macro Precision': m['macro_precision'],
            'Macro Recall':    m['macro_recall'],
            'Weighted F1':     round(f1_score(labels, preds, average='weighted'), 6),
            'F1 Normal':       round(per_class_f1[0], 6),
            'F1 Offensive':    round(per_class_f1[1], 6),
            'F1 Hate':         round(per_class_f1[2], 6),
        })

    del model_ft; gc.collect(); torch.cuda.empty_cache()
    del tokenizer; gc.collect(); torch.cuda.empty_cache()

# ── Combined results summary ────────────────────────────────────
combined_df = pd.DataFrame(combined_test_rows)
print('\n' + '='*90)
print('=== Combined-Training Results — BERT-base & Twitter-RoBERTa — Test Set ===')
print('='*90)
print(combined_df.to_string(index=False))
combined_df.to_csv('results/finetune_combined_results.csv', index=False)
print('\nSaved: results/finetune_combined_results.csv')
