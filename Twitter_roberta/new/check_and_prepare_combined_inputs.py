from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

FILES = {
    "HateXplain_train": SCRIPT_DIR / "HateXplain" / "train.csv",
    "HateXplain_val": SCRIPT_DIR / "HateXplain" / "val.csv",
    "HateXplain_test": SCRIPT_DIR / "HateXplain" / "test.csv",
    "TweetEval_train": SCRIPT_DIR / "Tweeteval" / "train_3class.csv",
    "TweetEval_val": SCRIPT_DIR / "Tweeteval" / "val_3class.csv",
    "TweetEval_test": SCRIPT_DIR / "Tweeteval" / "test_3class.csv",
}

OUT_DIR = SCRIPT_DIR / "prepared_for_combined"
OUT_DIR.mkdir(exist_ok=True)

LABEL_MAP = {
    0: "normal",
    1: "offensive",
    2: "hate",
}

SOURCE_MAP = {
    "HateXplain": "HateXplain",
    "TweetEval": "TweetEval",
}


def infer_source(name: str) -> str:
    if name.startswith("HateXplain"):
        return SOURCE_MAP["HateXplain"]
    elif name.startswith("TweetEval"):
        return SOURCE_MAP["TweetEval"]
    else:
        raise ValueError(f"Unknown source for {name}")


def check_and_prepare(name: str, path: Path):
    print(f"\n===== Checking {name} =====")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{name} must contain columns: text, label")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    invalid = df[~df["label"].isin([0, 1, 2])]
    if len(invalid) > 0:
        raise ValueError(f"{name} contains invalid labels outside 0/1/2")

    source = infer_source(name)
    df["source"] = source

    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("label counts:")
    print(df["label"].value_counts().sort_index())

    for k, v in LABEL_MAP.items():
        print(f"  {k} = {v}")

    out_path = OUT_DIR / f"{name}.csv"
    df[["text", "label", "source"]].to_csv(out_path, index=False, encoding="utf-8-sig")
    print("saved to:", out_path)


def main():
    for name, path in FILES.items():
        check_and_prepare(name, path)

    print("\nAll 6 files checked and standardized successfully.")
    print(f"Prepared files are saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()