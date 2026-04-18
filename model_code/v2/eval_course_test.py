import os, warnings
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

LABELS     = ['Normal', 'Offensive', 'Hate']
BATCH_SIZE = 16
COURSE_TEST = '../dataset/6713-ass2-dataset/ourdataset/course_test_300.csv'

# ── 模型目录（可在此添加更多）──────────────────────────
MODEL_DIRS = {
    'BERT-HateXplain (3-class)':               'saved_model_bert_hx',
    'BERT-TweetEval  (3-class)':               'saved_model_bert_te',
    'RoBERTa-HateXplain (3-class)':            'saved_model_roberta_hx',
    'RoBERTa-TweetEval  (3-class)':            'saved_model_roberta_te',
    'BERT-HateXplain+DomainAdapt (3-class)':   'saved_model_bert_hx_course',
    'RoBERTa-HateXplain+DomainAdapt (3-class)':'saved_model_roberta_hx_course',
}

# ── Dataset ───────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts); self.labels = list(labels)
        self.tokenizer = tokenizer; self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), padding='max_length',
                             truncation=True, max_length=self.max_len,
                             return_tensors='pt')
        item = {'input_ids':      enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}
        if 'token_type_ids' in enc:
            item['token_type_ids'] = enc['token_type_ids'].squeeze()
        return item

# ── Evaluate ──────────────────────────────────────────
def evaluate(model_name, model_dir, test_df):
    if not os.path.isdir(model_dir):
        print(f'  [SKIP] {model_name}: directory not found ({model_dir})')
        return None

    print(f'\n{"─"*60}')
    print(f'  Model : {model_name}')
    print(f'  Dir   : {model_dir}')

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model     = model.to(device); model.eval()

    dataset = TextDataset(test_df['text'], test_df['label'], tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='  Evaluating', leave=False, ncols=80):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            kwargs = {'input_ids': ids, 'attention_mask': mask}
            if 'token_type_ids' in batch:
                kwargs['token_type_ids'] = batch['token_type_ids'].to(device)
            out = model(**kwargs)
            preds_all.extend(out.logits.argmax(dim=-1).cpu().numpy())
            labels_all.extend(batch['label'].numpy())

    del model; torch.cuda.empty_cache()

    preds  = np.array(preds_all)
    labels = np.array(labels_all)

    print(classification_report(labels, preds, target_names=LABELS))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS)
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')
    plt.title(f'{model_name}\nCourse Test Set (286 samples)')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    fname = f'course_test_{safe_name}_cm.png'
    plt.savefig(fname, dpi=150); plt.close()
    print(f'  Saved: {fname}')

    per_f1 = f1_score(labels, preds, average=None, zero_division=0)
    return {
        'Model':           model_name,
        'Accuracy':        round(accuracy_score(labels, preds), 4),
        'Macro F1':        round(f1_score(labels, preds, average='macro'), 4),
        'Macro Precision': round(precision_score(labels, preds, average='macro', zero_division=0), 4),
        'Macro Recall':    round(recall_score(labels, preds, average='macro', zero_division=0), 4),
        'Weighted F1':     round(f1_score(labels, preds, average='weighted'), 4),
        'F1 Normal':       round(per_f1[0], 4),
        'F1 Offensive':    round(per_f1[1], 4),
        'F1 Hate':         round(per_f1[2], 4),
    }

# ── Main ──────────────────────────────────────────────
test_df = pd.read_csv(COURSE_TEST)
test_df.dropna(subset=['text', 'label'], inplace=True)
print(f'Course test set: {len(test_df)} samples')
print('Label distribution:')
for lbl, cnt in test_df['label'].value_counts().sort_index().items():
    print(f'  {LABELS[lbl]}: {cnt} ({cnt/len(test_df)*100:.1f}%)')

rows = []
for name, path in MODEL_DIRS.items():
    result = evaluate(name, path, test_df)
    if result:
        rows.append(result)

print('\n' + '='*90)
print('=== All Models — Course Test Set Results ===')
print('='*90)
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv('course_test_results.csv', index=False)
print('\nSaved: course_test_results.csv')
