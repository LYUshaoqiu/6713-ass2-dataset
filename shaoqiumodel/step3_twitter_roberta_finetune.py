import os, gc
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

# ── Paths ─────────────────────────────────────────────────
BASE = '../dataset/6713-ass2-dataset/Twitter_roberta/'
HX   = BASE + 'HateXplain/'
TE   = BASE + 'Tweeteval/'

# Use hateonly split: 0=non-hate(Normal+Offensive), 1=hate
hx_train = pd.read_csv(HX + 'train_hateonly.csv')
hx_val   = pd.read_csv(HX + 'val_hateonly.csv')
hx_test  = pd.read_csv(HX + 'test_hateonly.csv')
te_train = pd.read_csv(TE + 'train_hateonly.csv')
te_val   = pd.read_csv(TE + 'val_hateonly.csv')
te_test  = pd.read_csv(TE + 'test_hateonly.csv')

print(f'HateXplain  train:{len(hx_train)} val:{len(hx_val)} test:{len(hx_test)}')
print(f'TweetEval   train:{len(te_train)} val:{len(te_val)} test:{len(te_test)}')

MODEL_NAME = 'cardiffnlp/twitter-roberta-base-hate'
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f'Loaded tokenizer: {MODEL_NAME}')

# ── ManualAdamW (bypasses PyTorch 2.6 optimizer CUDA kernel bug) ──
class ManualAdamW:
    def __init__(self, params, lr=2e-5, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr; self.b1, self.b2 = betas; self.eps = eps; self.wd = weight_decay
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1**self.t; bc2 = 1.0 - self.b2**self.t
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad
            p.mul_(1.0 - self.lr * self.wd)
            self.m[i].mul_(self.b1).add_(g, alpha=1.0 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
            p.addcdiv_(self.m[i]/bc1, (self.v[i]/bc2).sqrt().add_(self.eps), value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad.zero_()

# ── Dataset (RoBERTa has no token_type_ids) ───────────────
class HateDataset(Dataset):
    def __init__(self, texts, labels, max_len=128):
        self.texts = list(texts); self.labels = list(labels); self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(str(self.texts[idx]), padding='max_length', truncation=True,
                        max_length=self.max_len, return_tensors='pt')
        return {'input_ids':      enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}

# ── Train / Evaluate ──────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    bar = tqdm(loader, desc='  train', leave=False, ncols=90)
    for batch in bar:
        optimizer.zero_grad()
        out = model(input_ids      = batch['input_ids'].to(device),
                    attention_mask = batch['attention_mask'].to(device),
                    labels         = batch['label'].to(device))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += out.loss.item()
        bar.set_postfix(loss=f'{out.loss.item():.3f}')
    return total_loss / len(loader)

def evaluate_model(model, loader):
    model.eval()
    preds_all, labels_all, losses = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='  eval ', leave=False, ncols=90):
            labs = batch['label'].to(device)
            out  = model(input_ids      = batch['input_ids'].to(device),
                         attention_mask = batch['attention_mask'].to(device),
                         labels         = labs)
            losses.append(out.loss.item())
            preds_all.extend(out.logits.argmax(dim=-1).cpu().numpy())
            labels_all.extend(batch['label'].numpy())
    preds = np.array(preds_all); labels = np.array(labels_all)
    return preds, labels, {
        'val_loss':        round(float(np.mean(losses)), 6),
        'accuracy':        round(accuracy_score(labels, preds), 6),
        'macro_f1':        round(f1_score(labels, preds, average='macro'), 6),
        'macro_precision': round(precision_score(labels, preds, average='macro', zero_division=0), 6),
        'macro_recall':    round(recall_score(labels, preds, average='macro', zero_division=0), 6),
    }

# ── Fine-tuning ───────────────────────────────────────────
LR_CANDIDATES = [1e-5, 2e-5, 3e-5]
BATCH_SIZE    = 8
EPOCHS        = 3

def run_finetune(train_df, val_df, dataset_name):
    print(f'\n[{dataset_name}] train:{len(train_df)}  val:{len(val_df)}')
    best_lr, best_val_f1, best_state, best_history = None, -1, None, None

    for lr in LR_CANDIDATES:
        print(f'  lr={lr}', flush=True)
        ft_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
        ).to(device)
        optimizer    = ManualAdamW(ft_model.parameters(), lr=lr, weight_decay=0.01)
        train_loader = DataLoader(HateDataset(train_df['text'], train_df['label']),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader   = DataLoader(HateDataset(val_df['text'],   val_df['label']),
                                  batch_size=BATCH_SIZE, num_workers=0)
        history = []
        for ep in range(EPOCHS):
            train_loss = train_epoch(ft_model, train_loader, optimizer)
            _, _, m    = evaluate_model(ft_model, val_loader)
            history.append({'Epoch':           ep+1,
                            'Training Loss':   round(train_loss, 6),
                            'Validation Loss': m['val_loss'],
                            'Accuracy':        m['accuracy'],
                            'Macro F1':        m['macro_f1'],
                            'Macro Precision': m['macro_precision'],
                            'Macro Recall':    m['macro_recall']})
            print(f'    epoch {ep+1}/{EPOCHS}  train_loss={train_loss:.4f}  val_f1={m["macro_f1"]:.4f}', flush=True)
            if device.type == 'cuda': torch.cuda.synchronize()
            torch.cuda.empty_cache()

        run_best_f1 = max(h['Macro F1'] for h in history)
        if run_best_f1 > best_val_f1:
            best_val_f1  = run_best_f1
            best_lr      = lr
            best_state   = {k: v.cpu().clone() for k, v in ft_model.state_dict().items()}
            best_history = history

        del ft_model; gc.collect(); torch.cuda.empty_cache()

    print(f'\n  -> Best lr={best_lr}  best_val_macro_f1={best_val_f1:.4f}')
    print(f'\n── Training Progress (lr={best_lr}, based on {dataset_name}) ──')
    print(pd.DataFrame(best_history).to_string(index=False))
    return best_state, best_lr

hx_best_state, hx_best_lr = run_finetune(hx_train, hx_val, 'HateXplain')
te_best_state, te_best_lr = run_finetune(te_train, te_val, 'TweetEval')

# ── Evaluate on test set + save best model ────────────────
LABELS = ['Non-hate', 'Hate']

def eval_and_save(best_state, test_df, dataset_name, save_dir):
    model_ft = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    )
    model_ft.load_state_dict(best_state)
    model_ft = model_ft.to(device)

    test_loader = DataLoader(HateDataset(test_df['text'], test_df['label']),
                             batch_size=BATCH_SIZE, num_workers=0)
    preds, labels, m = evaluate_model(model_ft, test_loader)

    print(f'\n── {dataset_name} Test Results ──')
    print(classification_report(labels, preds, target_names=LABELS))

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f'Twitter-RoBERTa Fine-tuned\n{dataset_name}')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    fname = f'roberta_finetune_{dataset_name.lower()}_cm.png'
    plt.savefig(fname, dpi=150); plt.close()
    print(f'Saved: {fname}')

    # Save model for CLI use
    os.makedirs(save_dir, exist_ok=True)
    model_ft.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f'Model saved to: {save_dir}')

    del model_ft; gc.collect(); torch.cuda.empty_cache()

    return {'Dataset':         dataset_name,
            'Mapping':         'hateonly (Non-hate vs Hate)',
            'Accuracy':        m['accuracy'],
            'Macro F1':        m['macro_f1'],
            'Macro Precision': m['macro_precision'],
            'Macro Recall':    m['macro_recall'],
            'Weighted F1':     round(f1_score(labels, preds, average='weighted'), 6)}

rows = [
    eval_and_save(hx_best_state, hx_test, 'HateXplain', 'saved_model_roberta_hx'),
    eval_and_save(te_best_state, te_test,  'TweetEval',  'saved_model_roberta_te'),
]

results_df = pd.DataFrame(rows)
print('\n=== Fine-tuned twitter-roberta-base-hate — Test Set Results ===')
print(results_df.to_string(index=False))
results_df.to_csv('roberta_finetune_results.csv', index=False)
print('\nSaved: roberta_finetune_results.csv')
