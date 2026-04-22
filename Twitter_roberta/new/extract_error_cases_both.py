from pathlib import Path
import pandas as pd

LABEL_MAP = {
    0: "normal",
    1: "offensive",
    2: "hate",
}

FILES = {
    "TweetEval": Path("training_outputs/twitter_roberta_tweeteval_3class_tuned/test_predictions_best_model.csv"),
    "HateXplain": Path("training_outputs/twitter_roberta_hatexplain_3class_tuned/test_predictions_best_model.csv"),
}

OUT_DIR = Path("training_outputs/error_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_label_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gold_label_name"] = df["label"].map(LABEL_MAP)
    df["pred_label_name"] = df["pred"].map(LABEL_MAP)
    df["error_pair"] = df["gold_label_name"] + " -> " + df["pred_label_name"]
    return df


def extract_errors(name: str, path: Path):
    df = pd.read_csv(path)

    required_cols = {"text", "label", "pred"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")

    error_df = df[df["label"] != df["pred"]].copy()
    error_df = add_label_names(error_df)

    # 保存所有错分
    all_out = OUT_DIR / f"{name.lower()}_all_error_cases.csv"
    error_df.to_csv(all_out, index=False, encoding="utf-8-sig")

    print(f"\n===== {name} =====")
    print(f"Total misclassified samples: {len(error_df)}")
    print("Error pair counts:")
    print(error_df["error_pair"].value_counts())

    # 按主要错分类型分别保存
    major_pairs = [
        "normal -> offensive",
        "offensive -> hate",
        "hate -> offensive",
        "normal -> hate",
        "offensive -> normal",
        "hate -> normal",
    ]

    for pair in major_pairs:
        pair_df = error_df[error_df["error_pair"] == pair].copy()
        if len(pair_df) > 0:
            out_path = OUT_DIR / f"{name.lower()}_{pair.replace(' ', '').replace('->', '_to_')}.csv"
            pair_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    return error_df


def main():
    all_results = []

    for name, path in FILES.items():
        error_df = extract_errors(name, path)
        error_df["dataset"] = name
        all_results.append(error_df)

    merged = pd.concat(all_results, ignore_index=True)
    merged_out = OUT_DIR / "combined_error_cases.csv"
    merged.to_csv(merged_out, index=False, encoding="utf-8-sig")

    print("\n===== Combined Summary =====")
    print(merged.groupby(["dataset", "error_pair"]).size().sort_values(ascending=False))
    print(f"\nSaved combined error cases to: {merged_out}")


if __name__ == "__main__":
    main()