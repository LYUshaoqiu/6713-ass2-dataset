from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "visualizations"
OUT_DIR.mkdir(exist_ok=True)


df = pd.DataFrame([
    {
        "Experiment": "HateXplain binary",
        "Dataset": "HateXplain",
        "Mapping": "binary",
        "Accuracy": 0.4587,
        "Macro F1": 0.4230,
        "Weighted F1": 0.3955
    },
    {
        "Experiment": "HateXplain hateonly",
        "Dataset": "HateXplain",
        "Mapping": "hateonly",
        "Accuracy": 0.6352,
        "Macro F1": 0.4891,
        "Weighted F1": 0.5931
    },
    {
        "Experiment": "TweetEval binary",
        "Dataset": "TweetEval",
        "Mapping": "binary",
        "Accuracy": 0.6076,
        "Macro F1": 0.6074,
        "Weighted F1": 0.6055
    },
    {
        "Experiment": "TweetEval hateonly",
        "Dataset": "TweetEval",
        "Mapping": "hateonly",
        "Accuracy": 0.6603,
        "Macro F1": 0.6598,
        "Weighted F1": 0.6645
    }
])


cms = {
    "HateXplain binary": [[712, 102], [987, 211]],
    "HateXplain hateonly": [[1177, 212], [522, 101]],
    "TweetEval binary": [[1121, 1217], [286, 1206]],
    "TweetEval hateonly": [[1342, 1236], [65, 1187]],
}


summary_csv = OUT_DIR / "experiment_summary_metrics.csv"
df.to_csv(summary_csv, index=False)


for metric in ["Accuracy", "Macro F1", "Weighted F1"]:
    plt.figure(figsize=(9, 5))
    plt.bar(df["Experiment"], df[metric])
    plt.ylim(0, 1)
    plt.ylabel(metric)
    plt.title(f"{metric} Comparison Across Experiments")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{metric.lower().replace(' ', '_')}_comparison.png", dpi=200)
    plt.close()


for name, cm in cms.items():
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm)

    ax.set_title(f"Confusion Matrix: {name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")

    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    fig.savefig(OUT_DIR / f"{safe_name}_confusion_matrix.png", dpi=200)
    plt.close(fig)

print("Done.")
print("Saved to:", OUT_DIR)