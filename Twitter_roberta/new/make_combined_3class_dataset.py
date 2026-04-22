from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

IN_DIR = SCRIPT_DIR / "prepared_for_combined"
OUT_DIR = SCRIPT_DIR / "Combined"
OUT_DIR.mkdir(exist_ok=True)

SPLITS = {
    "train": ["HateXplain_train.csv", "TweetEval_train.csv"],
    "val": ["HateXplain_val.csv", "TweetEval_val.csv"],
    "test": ["HateXplain_test.csv", "TweetEval_test.csv"],
}


def combine_split(split_name: str, file_list: list[str]):
    dfs = []

    print(f"\n===== Combining {split_name} =====")
    for file_name in file_list:
        path = IN_DIR / file_name
        df = pd.read_csv(path)

        required_cols = {"text", "label", "source"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{file_name} missing columns: {missing}")

        dfs.append(df)
        print(f"Loaded {file_name}: {df.shape}")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("Combined shape:", combined_df.shape)
    print("Label counts:")
    print(combined_df["label"].value_counts().sort_index())
    print("Source counts:")
    print(combined_df["source"].value_counts())

    out_path = OUT_DIR / f"{split_name}.csv"
    combined_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Saved to:", out_path)


def main():
    for split_name, file_list in SPLITS.items():
        combine_split(split_name, file_list)

    print("\nCombined 3-class dataset is ready.")
    print(f"Saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()