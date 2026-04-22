"""
Cascaded Binary Classification with BERT-base-uncased + Course Data Augmentation.

Pipeline per dataset:
  Stage 1: Normal (0) vs Not-Normal (1)   — trained on main + course data
  Stage 2: Offensive (0) vs Hate (1)      — trained on main + course data
  Inference: Stage1 routes each sample; Not-Normal samples go through Stage2.
  Final labels: 0=Normal, 1=Offensive, 2=Hate

Outputs (saved to results/ and models/):
  results/bert_cascaded_<dataset>_cm.png          — main test set CM
  results/bert_cascaded_course(<dataset>)_cm.png  — course test set CM
  results/bert_cascaded_results.csv               — all 4 result rows
  models/bert_<prefix>_stage1/                    — Stage 1 checkpoints
  models/bert_<prefix>_stage2/                    — Stage 2 checkpoints
"""
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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from transformers import BertTokenizer, BertForSequenceClassification
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

os.makedirs('results', exist_ok=True)
os.makedirs('models',  exist_ok=True)

# ── Paths ──────────────────────────────────────────────────
HX_DIR     = '../dataset/6713-ass2-dataset/HateXPlain_data/'
TE_DIR     = '../dataset/6713-ass2-dataset/Tweeteval三类分/'
COURSE_DIR = '../dataset/6713-ass2-dataset/ourdataset/'

# ── Load main datasets ──────────────────────────────────────
hx_train = pd.read_csv(HX_DIR + 'train.csv')
hx_val   = pd.read_csv(HX_DIR + 'val.csv')
hx_test  = pd.read_csv(HX_DIR + 'test.csv')
te_train = pd.read_csv(TE_DIR + 'train.csv').rename(columns={'comment': 'text'})
te_val   = pd.read_csv(TE_DIR + 'val.csv').rename(columns={'comment': 'text'})
te_test  = pd.read_csv(TE_DIR + 'test.csv').rename(columns={'comment': 'text'})

for df in [hx_train, hx_val, hx_test, te_train, te_val, te_test]:
    df.dropna(subset=['text', 'label'], inplace=True)
    df.reset_index(drop=True, inplace=True)

# ── Load & split course dataset (70 / 15 / 15) ─────────────
course_df = pd.read_csv(COURSE_DIR + 'course_reviews_cleaned.csv') \
              .rename(columns={'comment': 'text'})
course_df.dropna(subset=['text', 'label'], inplace=True)
course_train_val, course_test = train_test_split(
    course_df, test_size=0.15, random_state=42, stratify=course_df['label'])
course_train, course_val = train_test_split(
    course_train_val, test_size=0.176, random_state=42,
    stratify=course_train_val['label'])
# 0.176 ≈ 15/85 → final split ≈ 70/15/15

print(f'HateXPlain  train:{len(hx_train)} val:{len(hx_val)} test:{len(hx_test)}')
print(f'TweetEval   train:{len(te_train)} val:{len(te_val)} test:{len(te_test)}')
print(f'Course      train:{len(course_train)} val:{len(course_val)} test:{len(course_test)}')

MODEL_NAME    = 'bert-base-uncased'
tokenizer     = BertTokenizer.from_pretrained(MODEL_NAME)
LABELS        = ['Normal', 'Offensive', 'Hate']
LR_CANDIDATES = [1e-5, 2e-5, 3e-5]
BATCH_SIZE    = 8
EPOCHS        = 3

# ── ManualAdamW ─────────────────────────────────────────────
class ManualAdamW:
    def __init__(self, params, lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr; self.b1, self.b2 = betas; self.eps = eps; self.wd = weight_decay
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t; bc2 = 1.0 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad
            p.mul_(1.0 - self.lr * self.wd)
            self.m[i].mul_(self.b1).add_(g, alpha=1.0 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
            p.addcdiv_(self.m[i] / bc1, (self.v[i] / bc2).sqrt().add_(self.eps), value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad.zero_()

# ── Dataset ─────────────────────────────────────────────────
class HateDataset(Dataset):
    def __init__(self, texts, labels, max_len=128):
        self.texts = list(texts); self.labels = list(labels); self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(str(self.texts[idx]), padding='max_length', truncation=True,
                        max_length=self.max_len, return_tensors='pt')
        return {'input_ids':      enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(),
                'token_type_ids': enc['token_type_ids'].squeeze(),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}

# ── Data preparation helpers ─────────────────────────────────
def mix_with_course(main_df, course_portion):
    """Mix main dataset with course data and shuffle."""
    mixed = pd.concat([main_df, course_portion], ignore_index=True)
    return mixed.sample(frac=1, random_state=42).reset_index(drop=True)

def make_stage1_data(df):
    """Stage 1: Normal(0) vs Not-Normal(1). Offensive+Hate → 1."""
    d = df[['text', 'label']].copy()
    d['label'] = d['label'].apply(lambda x: 0 if x == 0 else 1)
    return d

def make_stage2_data(df):
    """Stage 2: Offensive(0) vs Hate(1). Keep only label 1 and 2."""
    d = df[df['label'].isin([1, 2])][['text', 'label']].copy()
    d['label'] = d['label'].apply(lambda x: 0 if x == 1 else 1)
    return d.reset_index(drop=True)

def get_class_weights(labels, n_classes):
    w = compute_class_weight('balanced', classes=np.arange(n_classes), y=np.array(labels))
    return torch.tensor(w, dtype=torch.float)

# ── Train / Evaluate ────────────────────────────────────────
def train_epoch(model, loader, optimizer, cw=None):
    model.train()
    total_loss = 0
    loss_fct = torch.nn.CrossEntropyLoss(weight=cw.to(device) if cw is not None else None)
    bar = tqdm(loader, desc='  train', leave=False, ncols=90)
    for batch in bar:
        optimizer.zero_grad()
        out  = model(input_ids      = batch['input_ids'].to(device),
                     attention_mask = batch['attention_mask'].to(device),
                     token_type_ids = batch['token_type_ids'].to(device))
        loss = loss_fct(out.logits, batch['label'].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        bar.set_postfix(loss=f'{loss.item():.3f}')
    return total_loss / len(loader)

def evaluate_model(model, loader):
    """Unweighted loss for honest validation metrics."""
    model.eval()
    preds_all, labels_all, losses = [], [], []
    loss_fct = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in tqdm(loader, desc='  eval ', leave=False, ncols=90):
            labs = batch['label'].to(device)
            out  = model(input_ids      = batch['input_ids'].to(device),
                         attention_mask = batch['attention_mask'].to(device),
                         token_type_ids = batch['token_type_ids'].to(device))
            losses.append(loss_fct(out.logits, labs).item())
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

# ── Generic fine-tune runner ─────────────────────────────────
def run_finetune(train_df, val_df, tag, num_labels, cw=None):
    print(f'\n  [{tag}] train:{len(train_df)}  val:{len(val_df)}', flush=True)
    if cw is not None:
        print(f'  class weights: {[round(x, 3) for x in cw.tolist()]}')

    best_lr, best_val_f1, best_state, best_history = None, -1, None, None

    for lr in LR_CANDIDATES:
        print(f'  lr={lr}', flush=True)
        ft_model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=num_labels, attn_implementation='eager'
        ).to(device)
        optimizer    = ManualAdamW(ft_model.parameters(), lr=lr, weight_decay=0.01)
        train_loader = DataLoader(HateDataset(train_df['text'], train_df['label']),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader   = DataLoader(HateDataset(val_df['text'],   val_df['label']),
                                  batch_size=BATCH_SIZE, num_workers=0)
        history = []
        for ep in range(EPOCHS):
            train_loss = train_epoch(ft_model, train_loader, optimizer, cw)
            _, _, m    = evaluate_model(ft_model, val_loader)
            history.append({'LR':              lr,
                            'Epoch':           ep + 1,
                            'Training Loss':   round(train_loss, 6),
                            'Validation Loss': m['val_loss'],
                            'Accuracy':        m['accuracy'],
                            'Macro F1':        m['macro_f1'],
                            'Macro Precision': m['macro_precision'],
                            'Macro Recall':    m['macro_recall']})
            print(f'    epoch {ep+1}/{EPOCHS}  '
                  f'train_loss={train_loss:.4f}  val_f1={m["macro_f1"]:.4f}', flush=True)
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
    print(pd.DataFrame(best_history).to_string(index=False))
    return best_state

# ── Cascaded Inference ───────────────────────────────────────
def predict_cascaded(texts, s1_state, s2_state):
    """
    Stage 1: Normal(0) vs Not-Normal(1)
    Stage 2: Offensive(0) vs Hate(1)  [only for Not-Normal samples]
    Final:   0=Normal, 1=Offensive, 2=Hate
    """
    texts = list(texts)
    dummy = [0] * len(texts)

    # ── Stage 1 ──
    s1_model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, attn_implementation='eager')
    s1_model.load_state_dict(s1_state)
    s1_model = s1_model.to(device); s1_model.eval()

    s1_loader = DataLoader(HateDataset(texts, dummy), batch_size=BATCH_SIZE, num_workers=0)
    s1_preds  = []
    with torch.no_grad():
        for batch in tqdm(s1_loader, desc='  Stage1 infer', leave=False, ncols=90):
            out = s1_model(input_ids      = batch['input_ids'].to(device),
                           attention_mask = batch['attention_mask'].to(device),
                           token_type_ids = batch['token_type_ids'].to(device))
            s1_preds.extend(out.logits.argmax(dim=-1).cpu().numpy())
    del s1_model; gc.collect(); torch.cuda.empty_cache()

    # ── Stage 2 (only Not-Normal samples) ──
    not_normal_idx = [i for i, p in enumerate(s1_preds) if p == 1]
    s2_results     = {}

    if not_normal_idx:
        s2_texts = [texts[i] for i in not_normal_idx]
        s2_dummy = [0] * len(s2_texts)

        s2_model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2, attn_implementation='eager')
        s2_model.load_state_dict(s2_state)
        s2_model = s2_model.to(device); s2_model.eval()

        s2_loader = DataLoader(HateDataset(s2_texts, s2_dummy),
                               batch_size=BATCH_SIZE, num_workers=0)
        s2_preds  = []
        with torch.no_grad():
            for batch in tqdm(s2_loader, desc='  Stage2 infer', leave=False, ncols=90):
                out = s2_model(input_ids      = batch['input_ids'].to(device),
                               attention_mask = batch['attention_mask'].to(device),
                               token_type_ids = batch['token_type_ids'].to(device))
                s2_preds.extend(out.logits.argmax(dim=-1).cpu().numpy())
        del s2_model; gc.collect(); torch.cuda.empty_cache()
        s2_results = {idx: p for idx, p in zip(not_normal_idx, s2_preds)}

    # ── Merge ──
    final_preds = []
    for i, p1 in enumerate(s1_preds):
        if p1 == 0:
            final_preds.append(0)
        else:
            final_preds.append(1 if s2_results.get(i, 0) == 0 else 2)
    return np.array(final_preds)

# ── Evaluate & save ──────────────────────────────────────────
def eval_and_save(s1_state, s2_state, test_df, dataset_name, save_s1, save_s2):
    print(f'\n── {dataset_name} Cascaded Test Results ──')

    preds  = predict_cascaded(test_df['text'], s1_state, s2_state)
    labels = test_df['label'].values

    print(classification_report(labels, preds, target_names=LABELS))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f'BERT Cascaded Fine-tuned\n{dataset_name}')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    safe_name = dataset_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    fname = f'results/bert_cascaded_{safe_name}_cm.png'
    plt.savefig(fname, dpi=150); plt.close()
    print(f'Saved: {fname}')

    # Save Stage 1 checkpoint
    m_s1 = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, attn_implementation='eager')
    m_s1.load_state_dict(s1_state)
    os.makedirs(save_s1, exist_ok=True)
    m_s1.save_pretrained(save_s1); tokenizer.save_pretrained(save_s1)
    print(f'Stage1 model saved to: {save_s1}')
    del m_s1

    # Save Stage 2 checkpoint
    m_s2 = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, attn_implementation='eager')
    m_s2.load_state_dict(s2_state)
    os.makedirs(save_s2, exist_ok=True)
    m_s2.save_pretrained(save_s2); tokenizer.save_pretrained(save_s2)
    print(f'Stage2 model saved to: {save_s2}')
    del m_s2
    gc.collect(); torch.cuda.empty_cache()

    per_f1 = f1_score(labels, preds, average=None, zero_division=0)
    return {'Dataset':         dataset_name,
            'Mapping':         'Cascaded+CourseData',
            'Accuracy':        round(accuracy_score(labels, preds), 6),
            'Macro F1':        round(f1_score(labels, preds, average='macro'), 6),
            'Macro Precision': round(precision_score(labels, preds, average='macro', zero_division=0), 6),
            'Macro Recall':    round(recall_score(labels, preds, average='macro', zero_division=0), 6),
            'Weighted F1':     round(f1_score(labels, preds, average='weighted'), 6),
            'F1 Normal':       round(per_f1[0], 6),
            'F1 Offensive':    round(per_f1[1], 6),
            'F1 Hate':         round(per_f1[2], 6)}

# ── Full pipeline per dataset ─────────────────────────────────
def run_pipeline(train_df, val_df, test_df, dataset_name, prefix):
    print(f'\n{"="*65}')
    print(f'  Dataset: {dataset_name}')
    print(f'{"="*65}')

    # Mix course data into train and val
    mixed_train = mix_with_course(train_df, course_train)
    mixed_val   = mix_with_course(val_df,   course_val)
    print(f'  Mixed train: {len(mixed_train)}  val: {len(mixed_val)}')

    # ── Stage 1: Normal vs Not-Normal ──
    print(f'\n── Stage 1: Normal vs Not-Normal ──')
    s1_train = make_stage1_data(mixed_train)
    s1_val   = make_stage1_data(mixed_val)
    s1_cw    = get_class_weights(s1_train['label'], n_classes=2)
    s1_state = run_finetune(s1_train, s1_val,
                            tag=f'{dataset_name}/Stage1', num_labels=2, cw=s1_cw)

    # ── Stage 2: Offensive vs Hate ──
    print(f'\n── Stage 2: Offensive vs Hate ──')
    s2_train = make_stage2_data(mixed_train)
    s2_val   = make_stage2_data(mixed_val)
    print(f'  Stage2 train:{len(s2_train)}  val:{len(s2_val)}')
    s2_cw    = get_class_weights(s2_train['label'], n_classes=2)
    s2_state = run_finetune(s2_train, s2_val,
                            tag=f'{dataset_name}/Stage2', num_labels=2, cw=s2_cw)

    # ── Evaluate on main test set ──
    result_main = eval_and_save(
        s1_state, s2_state, test_df, dataset_name,
        save_s1=f'models/bert_{prefix}_stage1',
        save_s2=f'models/bert_{prefix}_stage2')

    # ── Evaluate on course test set ──
    print(f'\n── Course Test Set ({dataset_name} model) ──')
    result_course = eval_and_save(
        s1_state, s2_state, course_test, f'Course({dataset_name})',
        save_s1=f'models/bert_{prefix}_stage1',   # already saved, overwrite ok
        save_s2=f'models/bert_{prefix}_stage2')

    return result_main, result_course

# ── Run ───────────────────────────────────────────────────────
hx_main,  hx_course  = run_pipeline(hx_train, hx_val, hx_test, 'HateXPlain', 'hx')
te_main,  te_course  = run_pipeline(te_train, te_val, te_test,  'TweetEval',  'te')

# ── Final Results Table ───────────────────────────────────────
rows       = [hx_main, te_main, hx_course, te_course]
results_df = pd.DataFrame(rows)

print('\n' + '='*100)
print('=== BERT Cascaded (Stage1: Normal vs Not-Normal | Stage2: Offensive vs Hate) + Course Data ===')
print('='*100)
print(results_df.to_string(index=False))
results_df.to_csv('results/bert_cascaded_results.csv', index=False)
print('\nSaved: results/bert_cascaded_results.csv')
