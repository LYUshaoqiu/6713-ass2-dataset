from pathlib import Path
import pandas as pd

# =========================================================
# TweetEval label meaning (from your readme)
# 0 = normal
# 1 = offensive
# 2 = hate
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Tweeteval"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"


def load_and_standardize(csv_path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    print(f"\n===== {split_name} original =====")
    print("shape:", df.shape)
    print("columns:", list(df.columns))

    if "comment" in df.columns:
        df = df.rename(columns={"comment": "text"})

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"{split_name} must contain text/comment and label columns, got {list(df.columns)}"
        )

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    print(f"===== {split_name} standardized =====")
    print("shape:", df.shape)
    print("label counts:")
    print(df["label"].value_counts().sort_index())

    return df[["text", "label"]]


def map_binary(label: int) -> int:
    # 0 -> 0
    # 1,2 -> 1
    return 0 if label == 0 else 1


def map_hateonly(label: int) -> int:
    # 0,1 -> 0
    # 2   -> 1
    return 1 if label == 2 else 0


def make_mapped_split(df: pd.DataFrame, mapping_type: str) -> pd.DataFrame:
    out = df.copy()

    if mapping_type == "binary":
        out["label"] = out["label"].apply(map_binary)
    elif mapping_type == "hateonly":
        out["label"] = out["label"].apply(map_hateonly)
    else:
        raise ValueError(f"Unknown mapping_type: {mapping_type}")

    return out[["text", "label"]]


def save_split(df: pd.DataFrame, filename: str):
    save_path = DATA_DIR / filename
    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")


def main():
    train_df = load_and_standardize(TRAIN_PATH, "train")
    val_df = load_and_standardize(VAL_PATH, "val")
    test_df = load_and_standardize(TEST_PATH, "test")

    # 保存一份标准化原始三分类（text,label）
    save_split(train_df, "train_3class.csv")
    save_split(val_df, "val_3class.csv")
    save_split(test_df, "test_3class.csv")

    # binary: normal vs (offensive + hate)
    train_binary = make_mapped_split(train_df, "binary")
    val_binary = make_mapped_split(val_df, "binary")
    test_binary = make_mapped_split(test_df, "binary")

    print("\n===== binary label counts =====")
    print("train:")
    print(train_binary["label"].value_counts().sort_index())
    print("val:")
    print(val_binary["label"].value_counts().sort_index())
    print("test:")
    print(test_binary["label"].value_counts().sort_index())

    save_split(train_binary, "train_binary.csv")
    save_split(val_binary, "val_binary.csv")
    save_split(test_binary, "test_binary.csv")

    # hateonly: (normal + offensive) vs hate
    train_hateonly = make_mapped_split(train_df, "hateonly")
    val_hateonly = make_mapped_split(val_df, "hateonly")
    test_hateonly = make_mapped_split(test_df, "hateonly")

    print("\n===== hateonly label counts =====")
    print("train:")
    print(train_hateonly["label"].value_counts().sort_index())
    print("val:")
    print(val_hateonly["label"].value_counts().sort_index())
    print("test:")
    print(test_hateonly["label"].value_counts().sort_index())

    save_split(train_hateonly, "train_hateonly.csv")
    save_split(val_hateonly, "val_hateonly.csv")
    save_split(test_hateonly, "test_hateonly.csv")

    print("\nAll TweetEval splits are ready.")


if __name__ == "__main__":
    main()