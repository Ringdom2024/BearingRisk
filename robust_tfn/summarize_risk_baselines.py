from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from robust_tfn.summarize_multidataset import CONDITIONS, EXPERIMENTS


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "research_runs"
REFERENCES = ("confidence", "negative_entropy", "probability_margin")


def main():
    rows = []
    for dataset, seeds in EXPERIMENTS.items():
        for seed, (baseline_name, _) in seeds.items():
            frame = pd.read_csv(RUNS / baseline_name / "learned_risk.csv")
            for condition in CONDITIONS:
                selected = frame[frame["condition"] == condition].set_index("score")
                learned = selected.loc["learned_evidence"]
                for reference in REFERENCES:
                    baseline = selected.loc[reference]
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "condition": condition,
                            "reference": reference,
                            "learned_aurc": learned["aurc"],
                            "reference_aurc": baseline["aurc"],
                            "aurc_reduction": baseline["aurc"] - learned["aurc"],
                            "learned_top10_accuracy": learned["accuracy_at_10pct"],
                            "reference_top10_accuracy": baseline[
                                "accuracy_at_10pct"
                            ],
                            "top10_accuracy_gain": learned["accuracy_at_10pct"]
                            - baseline["accuracy_at_10pct"],
                        }
                    )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["dataset", "condition", "reference"], sort=False)
        .agg(
            learned_aurc_mean=("learned_aurc", "mean"),
            reference_aurc_mean=("reference_aurc", "mean"),
            aurc_reduction_mean=("aurc_reduction", "mean"),
            top10_accuracy_gain_mean=("top10_accuracy_gain", "mean"),
            better_seed_count=("aurc_reduction", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )
    tests = []
    for condition in CONDITIONS:
        for reference in REFERENCES:
            delta = detail[
                (detail["condition"] == condition)
                & (detail["reference"] == reference)
            ]["aurc_reduction"]
            statistic, p_value = wilcoxon(
                delta, alternative="greater", zero_method="zsplit"
            )
            tests.append(
                {
                    "condition": condition,
                    "reference": reference,
                    "pairs": len(delta),
                    "mean_aurc_reduction": delta.mean(),
                    "positive_pairs": int((delta > 0).sum()),
                    "wilcoxon_statistic": statistic,
                    "one_sided_p_value": p_value,
                }
            )
    tests = pd.DataFrame(tests)
    detail.to_csv(RUNS / "risk_baselines_by_seed.csv", index=False)
    summary.to_csv(RUNS / "risk_baselines_summary.csv", index=False)
    tests.to_csv(RUNS / "risk_baselines_paired_tests.csv", index=False)
    print(summary[summary["condition"] == "snr_-6"].to_string(index=False))
    print()
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
