"""
Domain Adaptation: fine-tune saved_model_bert_hx on course dataset.
Strategy: start from the already-trained HateXplain model, continue
training on course data with a small LR to avoid catastrophic forgetting.
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
BASE_MODEL   = 'saved_model_bert_hx'          # 已训练好的 HateXplain 模型
COURSE_TRAIN = '../dataset/6713-ass2-dataset/ourdataset/course_reviews_cleaned.csv'
COURSE_TEST  = '../dataset/6713-ass2-dataset/ourdataset/course_test_300.csv'
SAVE_DIR     = 'saved_model_bert_hx_course'   # 领域适应后保存

LABELS = ['Normal', 'Offensive', 'Hate']

# ── Load tokenizer from base model ────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
print(f'Loaded tokenizer from: {BASE_MODEL}')

# ── Load & split course training data (70 / 30) ───────────
course_df = pd.read_csv(COURSE_TRAIN).rename(columns={'comment': 'text'})
course_df.dropna(subset=['text', 'label'], inplace=True)
course_df.reset_index(drop=True, inplace=True)

course_train, course_val = train_test_split(
    course_df, test_size=0.2, random_state=42, stratify=course_df['label'])
course_train = course_train.reset_index(drop=True)
course_val   = course_val.reset_index(drop=True)

# ── Load course test set ───────────────────────────────────
course_test = pd.read_csv(COURSE_TEST)
course_test.dropna(subset=['text', 'label'], inplace=True)
course_test.reset_index(drop=True, inplace=True)

print(f'\nCourse  train:{len(course_train)}  val:{len(course_val)}  test:{len(course_test)}')
print('Train label distribution:')
for lbl, cnt in course_train['label'].value_counts().sort_index().items():
    print(f'  {LABELS[lbl]}: {cnt} ({cnt/len(course_train)*100:.1f}%)')

# ── Class weights ─────────────────────────────────────────
cw_vals  = compute_class_weight('balanced', classes=np.array([0,1,2]),
                                 y=course_train['label'].values)
cw_tensor = torch.tensor(cw_vals, dtype=torch.float)
print(f'\nClass weights: Normal={cw_vals[0]:.3f}  '
      f'Offensive={cw_vals[1]:.3f}  Hate={cw_vals[2]:.3f}')

# ── ManualAdamW ───────────────────────────────────────────
class ManualAdamW:
    def __init__(self, params, lr=1e-5, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01):
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

# ── Dataset ───────────────────────────────────────────────
class HateDataset(Dataset):
    def __init__(self, texts, labels, max_len=128):
        self.texts = list(texts); self.labels = list(labels); self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(str(self.texts[idx]), padding='max_length', truncation=True,
                        max_length=self.max_len, return_tensors='pt')
        item = {'input_ids':      enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}
        if 'token_type_ids' in enc:
            item['token_type_ids'] = enc['token_type_ids'].squeeze()
        return item

# ── Train / Evaluate ──────────────────────────────────────
def train_epoch(model, loader, optimizer, cw):
    model.train()
    total_loss = 0
    loss_fct = torch.nn.CrossEntropyLoss(weight=cw.to(device))
    bar = tqdm(loader, desc='  train', leave=False, ncols=90)
    for batch in bar:
        optimizer.zero_grad()
        kwargs = {'input_ids':      batch['input_ids'].to(device),
                  'attention_mask': batch['attention_mask'].to(device)}
        if 'token_type_ids' in batch:
            kwargs['token_type_ids'] = batch['token_type_ids'].to(device)
        out  = model(**kwargs)
        loss = loss_fct(out.logits, batch['label'].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        bar.set_postfix(loss=f'{loss.item():.3f}')
    return total_loss / len(loader)

def evaluate_model(model, loader):
    model.eval()
    preds_all, labels_all, losses = [], [], []
    loss_fct = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in tqdm(loader, desc='  eval ', leave=False, ncols=90):
            labs = batch['label'].to(device)
            kwargs = {'input_ids':      batch['input_ids'].to(device),
                      'attention_mask': batch['attention_mask'].to(device)}
            if 'token_type_ids' in batch:
                kwargs['token_type_ids'] = batch['token_type_ids'].to(device)
            out = model(**kwargs)
            losses.append(loss_fct(out.logits, labs).item())
            preds_all.extend(out.logits.argmax(dim=-1).cpu().numpy())
            labels_all.extend(batch['label'].numpy())
    preds  = np.array(preds_all); labels = np.array(labels_all)
    return preds, labels, {
        'val_loss':        round(float(np.mean(losses)), 6),
        'accuracy':        round(accuracy_score(labels, preds), 6),
        'macro_f1':        round(f1_score(labels, preds, average='macro'), 6),
        'macro_precision': round(precision_score(labels, preds, average='macro', zero_division=0), 6),
        'macro_recall':    round(recall_score(labels, preds, average='macro', zero_division=0), 6),
    }

# ── Step 1: Baseline — before fine-tuning ─────────────────
print('\n' + '='*60)
print('STEP 1: Baseline (saved_model_bert_hx on course test)')
print('='*60)

base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL).to(device)
test_loader = DataLoader(HateDataset(course_test['text'], course_test['label']),
                         batch_size=16, num_workers=0)
base_preds, base_labels, _ = evaluate_model(base_model, test_loader)
print(classification_report(base_labels, base_preds, target_names=LABELS))
base_f1_per = f1_score(base_labels, base_preds, average=None, zero_division=0)
baseline_row = {
    'Stage':         'Before adaptation',
    'Accuracy':      round(accuracy_score(base_labels, base_preds), 4),
    'Macro F1':      round(f1_score(base_labels, base_preds, average='macro'), 4),
    'F1 Normal':     round(base_f1_per[0], 4),
    'F1 Offensive':  round(base_f1_per[1], 4),
    'F1 Hate':       round(base_f1_per[2], 4),
}
del base_model; gc.collect(); torch.cuda.empty_cache()

# ── Step 2: Domain adaptation fine-tuning ─────────────────
print('\n' + '='*60)
print('STEP 2: Domain adaptation fine-tuning on course data')
print('='*60)

# Small LR candidates to avoid forgetting original knowledge
LR_CANDIDATES = [5e-6, 1e-5, 2e-5]
BATCH_SIZE    = 8
EPOCHS        = 5   # more epochs since dataset is small

train_loader = DataLoader(HateDataset(course_train['text'], course_train['label']),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(HateDataset(course_val['text'],   course_val['label']),
                          batch_size=BATCH_SIZE, num_workers=0)

best_lr, best_val_f1, best_state, best_history = None, -1, None, None

for lr in LR_CANDIDATES:
    print(f'\n  lr={lr}', flush=True)
    # Always reload from BASE_MODEL to ensure fair LR comparison
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL).to(device)
    optimizer = ManualAdamW(model.parameters(), lr=lr, weight_decay=0.01)

    history = []
    for ep in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, cw_tensor)
        _, _, m    = evaluate_model(model, val_loader)
        history.append({'Epoch':           ep+1,
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

    run_best = max(h['Macro F1'] for h in history)
    if run_best > best_val_f1:
        best_val_f1  = run_best
        best_lr      = lr
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_history = history

    del model; gc.collect(); torch.cuda.empty_cache()

print(f'\n  -> Best lr={best_lr}  best_val_macro_f1={best_val_f1:.4f}')
print(f'\n── Training Progress (lr={best_lr}) ──')
print(pd.DataFrame(best_history).to_string(index=False))

# ── Step 3: Evaluate adapted model on course test ─────────
print('\n' + '='*60)
print('STEP 3: Adapted model on course test set')
print('='*60)

adapted_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
adapted_model.load_state_dict(best_state)
adapted_model = adapted_model.to(device)

adapt_preds, adapt_labels, _ = evaluate_model(adapted_model, test_loader)
print(classification_report(adapt_labels, adapt_preds, target_names=LABELS))

cm = confusion_matrix(adapt_labels, adapt_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABELS, yticklabels=LABELS)
plt.title('BERT-HateXplain + Domain Adaptation\nCourse Test Set')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig('course_domain_adapt_cm.png', dpi=150); plt.close()
print('Saved: course_domain_adapt_cm.png')

# Save adapted model
os.makedirs(SAVE_DIR, exist_ok=True)
adapted_model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f'Adapted model saved to: {SAVE_DIR}')

adapt_f1_per = f1_score(adapt_labels, adapt_preds, average=None, zero_division=0)
adapted_row = {
    'Stage':        'After adaptation',
    'Accuracy':     round(accuracy_score(adapt_labels, adapt_preds), 4),
    'Macro F1':     round(f1_score(adapt_labels, adapt_preds, average='macro'), 4),
    'F1 Normal':    round(adapt_f1_per[0], 4),
    'F1 Offensive': round(adapt_f1_per[1], 4),
    'F1 Hate':      round(adapt_f1_per[2], 4),
}
del adapted_model; gc.collect(); torch.cuda.empty_cache()

# ── Step 4: Before vs After comparison ────────────────────
print('\n' + '='*60)
print('=== Before vs After Domain Adaptation (Course Test Set) ===')
print('='*60)
comparison = pd.DataFrame([baseline_row, adapted_row])
print(comparison.to_string(index=False))
comparison.to_csv('course_domain_adapt_results.csv', index=False)
print('\nSaved: course_domain_adapt_results.csv')
