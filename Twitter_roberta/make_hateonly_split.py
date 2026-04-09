from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR

train_path = DATA_DIR / "train.csv"
val_path = DATA_DIR / "val.csv"
test_path = DATA_DIR / "test.csv"

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

# 原始标签：
# 0 = normal
# 1 = offensive
# 2 = hate speech
#
# 新映射：
# 0,1 -> 0 (non-hate)
# 2   -> 1 (hate)

def map_hateonly(label: int) -> int:
    return 1 if label == 2 else 0

for df in [train_df, val_df, test_df]:
    df["label_hateonly"] = df["label"].apply(map_hateonly)

train_new = train_df[["text", "label_hateonly"]].rename(columns={"label_hateonly": "label"})
val_new = val_df[["text", "label_hateonly"]].rename(columns={"label_hateonly": "label"})
test_new = test_df[["text", "label_hateonly"]].rename(columns={"label_hateonly": "label"})

print("train label counts:")
print(train_new["label"].value_counts().sort_index())
print("\nval label counts:")
print(val_new["label"].value_counts().sort_index())
print("\ntest label counts:")
print(test_new["label"].value_counts().sort_index())

train_new.to_csv(DATA_DIR / "train_hateonly.csv", index=False)
val_new.to_csv(DATA_DIR / "val_hateonly.csv", index=False)
test_new.to_csv(DATA_DIR / "test_hateonly.csv", index=False)

print("\nSaved:")
print(DATA_DIR / "train_hateonly.csv")
print(DATA_DIR / "val_hateonly.csv")
print(DATA_DIR / "test_hateonly.csv")