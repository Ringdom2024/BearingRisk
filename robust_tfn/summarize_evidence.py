from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from robust_tfn.summarize_multidataset import CONDITIONS, EXPERIMENTS


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "research_runs"
ARCHITECTURES = ("CNN", "ResNet1D")


def read_tfn_metrics(run_name):
    frame = pd.read_csv(RUNS / run_name / "metrics.csv")
    return frame[frame["score"] == "evidence"].set_index("condition")


def architecture_rows():
    rows = []
    for dataset, seeds in EXPERIMENTS.items():
        for seed, (baseline_name, improved_name) in seeds.items():
            variants = {
                "TFN": read_tfn_metrics(baseline_name),
                "Physics-TFN": read_tfn_metrics(improved_name),
            }
            plain_name = f"{dataset.lower()}_seed_{seed}_plain_tfconv"
            plain_path = RUNS / plain_name / "metrics.csv"
            if plain_path.exists():
                variants["Plain-TFN"] = read_tfn_metrics(plain_name)
            for architecture in ARCHITECTURES:
                path = (
                    RUNS
                    / f"baseline_{dataset.lower()}_{architecture}_{seed}"
                    / "metrics.csv"
                )
                variants[architecture] = pd.read_csv(path).set_index("condition")
            for model, frame in variants.items():
                for condition in CONDITIONS:
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "condition": condition,
                            "model": model,
                            "accuracy": frame.loc[condition, "accuracy"],
                            "macro_f1": frame.loc[condition, "macro_f1"],
                        }
                    )
    return pd.DataFrame(rows)


def summarize(rows):
    return (
        rows.groupby(["dataset", "condition", "model"], sort=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .reset_index()
    )


def paired_tests(rows):
    tests = []
    comparisons = [("Physics-TFN", "TFN")]
    if "Plain-TFN" in set(rows["model"]):
        comparisons.append(("Physics-TFN", "Plain-TFN"))
    for condition in CONDITIONS:
        selected = rows[rows["condition"] == condition]
        for metric in ("accuracy", "macro_f1"):
            pivot = selected.pivot(
                index=["dataset", "seed"], columns="model", values=metric
            )
            for improved, reference in comparisons:
                delta = pivot[improved] - pivot[reference]
                statistic, p_value = wilcoxon(
                    delta, alternative="greater", zero_method="zsplit"
                )
                tests.append(
                    {
                        "condition": condition,
                        "metric": metric,
                        "improved": improved,
                        "reference": reference,
                        "pairs": len(delta),
                        "mean_delta": delta.mean(),
                        "median_delta": delta.median(),
                        "positive_pairs": int((delta > 0).sum()),
                        "wilcoxon_statistic": statistic,
                        "one_sided_p_value": p_value,
                    }
                )
    return pd.DataFrame(tests)


def main():
    detail = architecture_rows()
    summary = summarize(detail)
    tests = paired_tests(detail)
    detail.to_csv(RUNS / "architecture_baseline_by_seed.csv", index=False)
    summary.to_csv(RUNS / "architecture_baseline_summary.csv", index=False)
    tests.to_csv(RUNS / "paired_statistical_tests.csv", index=False)
    print(summary.to_string(index=False))
    print()
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
