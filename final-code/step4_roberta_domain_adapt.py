
#This program does 5 things:

#1. Load a RoBERTa model that was already trained on HateXPlain
#2. Test this original model on three datasets
#3. Continue training the model on our course dataset
#4. Save the new adapted model
#5. Test the adapted model again and compare the results

#Our goal is to see whether extra training on the course dataset can improve the model's performance.

#Step 1: load all datasets
#We use:
#1. the course dataset for extra training
#2. three test sets for evaluation

#Step 2: test the original model
#this helps us see the model performance before adaptation

#Step 3: continue training on the course dataset
#we try many learning rates and keep the best model

#Step 4: save the adapted model

#Step 5: test the adapted model again
#then we compare before and after results
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

os.makedirs('results', exist_ok=True)
os.makedirs('models',  exist_ok=True)

# ── Paths ──────────────────────────────────────────────────
BASE_MODEL   = 'models/roberta_HateXPlain'   # output of step1_finetune_all.py (NOT modified)
COURSE_TRAIN = '../dataset/6713-ass2-dataset/ourdataset/course_reviews_cleaned.csv'
COURSE_TEST  = '../dataset/6713-ass2-dataset/ourdataset/course_test_300.csv'
HX_DIR       = '../dataset/6713-ass2-dataset/HateXPlain_data/'
TE_DIR       = '../dataset/6713-ass2-dataset/Tweeteval三类分/'
SAVE_DIR     = 'models/roberta_da_course'



LABELS = ['Normal', 'Offensive', 'Hate']

# ── Load tokenizer from base model ─────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
print(f'Loaded tokenizer from: {BASE_MODEL}')

# ── Load course training data (80 / 20 split) ──────────────
course_df = pd.read_csv(COURSE_TRAIN).rename(columns={'comment': 'text'})
course_df.dropna(subset=['text', 'label'], inplace=True)
course_df.reset_index(drop=True, inplace=True)

course_train, course_val = train_test_split(
    course_df, test_size=0.2, random_state=42, stratify=course_df['label'])
course_train = course_train.reset_index(drop=True)
course_val = course_val.reset_index(drop=True)

# ── Load all three test sets ────────────────────────────────
course_test = pd.read_csv(COURSE_TEST)
course_test.dropna(subset=['text', 'label'], inplace=True)

course_test.reset_index(drop=True, inplace=True)

hatexplain_test_df = pd.read_csv(HX_DIR + 'test.csv')
hatexplain_test_df.dropna(subset=['text', 'label'], inplace=True)
hatexplain_test_df.reset_index(drop=True, inplace=True)

tweeteval_test_df = pd.read_csv(TE_DIR + 'test.csv').rename(columns={'comment': 'text'})
tweeteval_test_df.dropna(subset=['text', 'label'], inplace=True)
tweeteval_test_df.reset_index(drop=True, inplace=True)

print(f'\nCourse  train:{len(course_train)} val:{len(course_val)} test:{len(course_test)}')
print(f'HateXPlain test: {len(hatexplain_test_df)}   TweetEval test: {len(tweeteval_test_df)}')
print('Course train label dist:')
for lbl, cnt in course_train['label'].value_counts().sort_index().items():
    print(f'  {LABELS[lbl]}: {cnt} ({cnt/len(course_train)*100:.1f}%)')

# ── Class weights ───────────────────────────────────────────
class_weight_values = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=course_train['label'].values)

cw_tensor = torch.tensor(class_weight_values, dtype=torch.float)
print(f'\nClass weights: Normal={class_weight_values[0]:.3f}  '
      f'Offensive={class_weight_values[1]:.3f}  Hate={class_weight_values[2]:.3f}')

# ── ManualAdamW ─────────────────────────────────────────────
class ManualAdamW:
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr; 
        self.b1, 
        self.b2 = betas; 
        self.eps = eps; 
        self.wd = weight_decay

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

# ── Dataset for RoBERTa (RoBERTa has no token_type_ids) ─────────────────
class HateDataset(Dataset):
    """
    a simple dataset class
    it stores text samples and their labels, and tokenizes each text when it is requested
    """

    def __init__(self, texts, labels, max_len=128):
        self.texts = list(texts); self.labels = list(labels); self.max_len = max_len

    def __len__(self): return len(self.texts)


    def __getitem__(self, idx):
        enc = tokenizer(str(self.texts[idx]), padding='max_length', truncation=True,
                        max_length=self.max_len, return_tensors='pt')
        return {'input_ids':      enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}

# ── Train / Evaluate ────────────────────────────────────────
def train_epoch(model, loader, optimizer, cw):

    
    #train the model for one full pass by the training data
    #return the average training loss
    
    model.train()
    total_loss = 0
    loss_fct = torch.nn.CrossEntropyLoss(weight=cw.to(device))
    bar = tqdm(loader, desc='  train', leave=False, ncols=90)
    for batch in bar:
        optimizer.zero_grad()
        out  = model(input_ids      = batch['input_ids'].to(device),
                     attention_mask = batch['attention_mask'].to(device))
        loss = loss_fct(out.logits, batch['label'].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        bar.set_postfix(loss=f'{loss.item():.3f}')
    return total_loss / len(loader)

def evaluate_model(model, loader):
    
    #run model on a dataset directly
    #return predictions, true labels, and evaluation scores
    
    model.eval()
    preds_all, labels_all, losses = [], [], []
    loss_fct = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in tqdm(loader, desc='  eval ', leave=False, ncols=90):
            labs = batch['label'].to(device)
            out  = model(input_ids      = batch['input_ids'].to(device),
                         attention_mask = batch['attention_mask'].to(device))
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

def eval_on_testset(model, test_df, test_name, cm_path):

    #evaluate the model in one test dataset
    #save a confusion matrix figure and return summary metrics

   

    loader = DataLoader(HateDataset(test_df['text'], test_df['label']),
                        batch_size=16, num_workers=0)
    preds, labels, _ = evaluate_model(model, loader)
    print(f'\n── Test: {test_name} ({len(test_df)} samples) ──')
    print(classification_report(labels, preds, target_names=LABELS))
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f'RoBERTa-HateXPlain + DA\n{test_name}')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150); plt.close()
    print(f'  CM saved: {cm_path}')
    per_f1 = f1_score(labels, preds, average=None, zero_division=0)
    return {
        'Test Set':        test_name,
        'Accuracy':        round(accuracy_score(labels, preds), 4),
        'Macro F1':        round(f1_score(labels, preds, average='macro'), 4),
        'Macro Precision': round(precision_score(labels, preds, average='macro', zero_division=0), 4),
        'Macro Recall':    round(recall_score(labels, preds, average='macro', zero_division=0), 4),
        'Weighted F1':     round(f1_score(labels, preds, average='weighted'), 4),
        'F1 Normal':       round(per_f1[0], 4),
        'F1 Offensive':    round(per_f1[1], 4),
        'F1 Hate':         round(per_f1[2], 4),
    }

# ── STEP 1: Baseline — base model on all three test sets ────
print('\n' + '='*60)
print('STEP 1: Baseline — roberta_HateXPlain (before DA) on all test sets')
print('='*60)

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, attn_implementation='eager').to(device)
baseline_model_results = [
    eval_on_testset(base_model, hatexplain_test_df,    'HateXPlain', 'results/roberta_base_hx_cm.png'),
    eval_on_testset(base_model, tweeteval_test_df,     'TweetEval',  'results/roberta_base_te_cm.png'),
    eval_on_testset(base_model, course_test, 'Course',     'results/roberta_base_course_cm.png'),
]
del base_model; gc.collect(); torch.cuda.empty_cache()

# ── STEP 2: Domain adaptation training ──────────────────────
print('\n' + '='*60)
print('STEP 2: Domain adaptation fine-tuning on course data')
print('='*60)

LR_CANDIDATES = [5e-6, 1e-5, 2e-5]
BATCH_SIZE    = 8
EPOCHS        = 5

train_loader = DataLoader(HateDataset(course_train['text'], course_train['label']),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(HateDataset(course_val['text'],   course_val['label']),
                          batch_size=BATCH_SIZE, num_workers=0)

best_lr, best_val_f1, best_state, all_history = None, -1, None, []

for lr in LR_CANDIDATES:
    print(f'\n  lr={lr}', flush=True)
    # Always reload from BASE_MODEL for fair LR comparison
    model     = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, attn_implementation='eager').to(device)
    optimizer = ManualAdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lr_best_f1, lr_best_state = -1, None

    for ep in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, cw_tensor)
        _, _, m    = evaluate_model(model, val_loader)
        all_history.append({'LR': lr, 'Epoch': ep + 1,
                            'Training Loss':   round(train_loss, 6),
                            'Validation Loss': m['val_loss'],
                            'Accuracy':        m['accuracy'],
                            'Macro F1':        m['macro_f1'],
                            'Macro Precision': m['macro_precision'],
                            'Macro Recall':    m['macro_recall']})
        print(f'    epoch {ep+1}/{EPOCHS}  '
              f'train_loss={train_loss:.4f}  val_f1={m["macro_f1"]:.4f}', flush=True)
        if m['macro_f1'] > lr_best_f1:
            lr_best_f1    = m['macro_f1']
            lr_best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if device.type == 'cuda': torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if lr_best_f1 > best_val_f1:
        best_val_f1 = lr_best_f1
        best_lr     = lr
        best_state  = lr_best_state

    del model; gc.collect(); torch.cuda.empty_cache()

print(f'\n  -> Best LR={best_lr}  best_val_macro_f1={best_val_f1:.4f}')
pd.DataFrame(all_history).to_csv('results/roberta_da_history.csv', index=False)
print('Saved: results/roberta_da_history.csv')

# ── STEP 3: Save DA model ────────────────────────────────────
print('\n' + '='*60)
print('STEP 3: Save domain-adapted model')
print('='*60)

adapted_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, attn_implementation='eager')
adapted_model.load_state_dict(best_state)
adapted_model = adapted_model.to(device)

os.makedirs(SAVE_DIR, exist_ok=True)
adapted_model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f'DA model saved to: {SAVE_DIR}')

# ── STEP 4: Evaluate DA model on all three test sets ─────────
print('\n' + '='*60)
print('STEP 4: DA model evaluation on all three test sets')
print('='*60)

adapted_model_results = [
    eval_on_testset(adapted_model, hatexplain_test_df,    'HateXPlain', 'results/roberta_da_hatexplain_cm.png'),
    eval_on_testset(adapted_model, tweeteval_test_df,     'TweetEval',  'results/roberta_da_tweeteval_cm.png'),
    eval_on_testset(adapted_model, course_test, 'Course',     'results/roberta_da_course_cm.png'),
]
del adapted_model; gc.collect(); torch.cuda.empty_cache()

# ── STEP 5: Summary comparison table ─────────────────────────
print('\n' + '='*70)
print('=== RoBERTa Domain Adaptation — Before vs After (all test sets) ===')
print('='*70)

for row in baseline_model_results: row['Stage'] = 'Before DA (roberta_HateXPlain)'
for row in adapted_model_results:       row['Stage'] = f'After DA (best_lr={best_lr})'

all_rows = []
for b, a in zip(baseline_model_results, adapted_model_results):
    all_rows.append(b); all_rows.append(a)

summary_df = pd.DataFrame(all_rows)[
    ['Stage', 'Test Set', 'Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall',
     'Weighted F1', 'F1 Normal', 'F1 Offensive', 'F1 Hate']
]
print(summary_df.to_string(index=False))
summary_df.to_csv('results/roberta_da_results.csv', index=False)
print('\nSaved: results/roberta_da_results.csv')
