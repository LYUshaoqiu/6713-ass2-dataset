from pathlib import Path
import pandas as pd
import math

LABEL_MAP = {
    0: "normal",
    1: "offensive",
    2: "hate",
}

FILES = {
    "TweetEval": Path("training_outputs/twitter_roberta_tweeteval_3class_tuned/test_predictions_best_model.csv"),
    "HateXplain": Path("training_outputs/twitter_roberta_hatexplain_3class_tuned/test_predictions_best_model.csv"),
}

OUT_DIR = Path("training_outputs/error_analysis_samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 50
RANDOM_STATE = 42


def prepare_errors(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = {"text", "label", "pred"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df = df[df["label"] != df["pred"]].copy()
    df["gold_label_name"] = df["label"].map(LABEL_MAP)
    df["pred_label_name"] = df["pred"].map(LABEL_MAP)
    df["error_pair"] = df["gold_label_name"] + " -> " + df["pred_label_name"]
    return df


def stratified_sample(df: pd.DataFrame, total_n: int) -> pd.DataFrame:
    if len(df) <= total_n:
        return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    counts = df["error_pair"].value_counts()
    proportions = counts / counts.sum()


    alloc = {k: math.floor(v * total_n) for k, v in proportions.items()}


    for k in alloc:
        if alloc[k] == 0 and counts[k] > 0:
            alloc[k] = 1

    current_total = sum(alloc.values())

    if current_total > total_n:

        for k in counts.index:
            while alloc[k] > 1 and current_total > total_n:
                alloc[k] -= 1
                current_total -= 1
    elif current_total < total_n:

        for k in counts.index:
            while alloc[k] < counts[k] and current_total < total_n:
                alloc[k] += 1
                current_total += 1

    parts = []
    for pair, n in alloc.items():
        pair_df = df[df["error_pair"] == pair]
        n = min(n, len(pair_df))
        sampled = pair_df.sample(n=n, random_state=RANDOM_STATE)
        parts.append(sampled)

    sampled_df = pd.concat(parts, ignore_index=True)

    if len(sampled_df) < total_n:
        remaining = df.drop(sampled_df.index, errors="ignore")
        need = min(total_n - len(sampled_df), len(remaining))
        if need > 0:
            extra = remaining.sample(n=need, random_state=RANDOM_STATE)
            sampled_df = pd.concat([sampled_df, extra], ignore_index=True)

    return sampled_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def main():
    summary_rows = []

    for dataset_name, path in FILES.items():
        error_df = prepare_errors(path)

        print(f"\n===== {dataset_name} =====")
        print(f"Total misclassified samples: {len(error_df)}")
        print("Error pair counts:")
        print(error_df["error_pair"].value_counts())

        sampled_df = stratified_sample(error_df, SAMPLE_SIZE)
        sampled_df["dataset"] = dataset_name
        sampled_df["manual_error_type"] = ""
        sampled_df["notes"] = ""

        out_path = OUT_DIR / f"{dataset_name.lower()}_50_error_samples.csv"
        sampled_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved sampled file to: {out_path}")

        summary_rows.append({
            "dataset": dataset_name,
            "total_errors": len(error_df),
            "sampled_errors": len(sampled_df),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "sampling_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved summary to: {OUT_DIR / 'sampling_summary.csv'}")


if __name__ == "__main__":
    main()