from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "research_runs"
CONDITIONS = ["clean", "snr_6", "snr_0", "snr_-6"]
EXPERIMENTS = {
    "CWRU": {
        999: ("formal_noise_aug", "formal_tfconv_calibration"),
        1001: ("seed_1001_noise_aug", "seed_1001_tfconv_calibration"),
        1003: ("seed_1003_noise_aug", "seed_1003_tfconv_calibration"),
    },
    "JNU": {
        999: ("jnu_seed_999_noise_aug", "jnu_seed_999_tfconv_calibration"),
        1001: ("jnu_seed_1001_noise_aug", "jnu_seed_1001_tfconv_calibration"),
        1003: ("jnu_seed_1003_noise_aug", "jnu_seed_1003_tfconv_calibration"),
    },
    "PADERBORN": {
        999: (
            "paderborn_seed_999_noise_aug",
            "paderborn_seed_999_tfconv_calibration",
        ),
        1001: (
            "paderborn_seed_1001_noise_aug",
            "paderborn_seed_1001_tfconv_calibration",
        ),
        1003: (
            "paderborn_seed_1003_noise_aug",
            "paderborn_seed_1003_tfconv_calibration",
        ),
    },
}


def classification_rows():
    rows = []
    for dataset, seeds in EXPERIMENTS.items():
        for seed, (baseline_name, improved_name) in seeds.items():
            baseline = pd.read_csv(RUNS / baseline_name / "metrics.csv")
            improved = pd.read_csv(RUNS / improved_name / "metrics.csv")
            baseline = baseline[baseline["score"] == "evidence"].set_index(
                "condition"
            )
            improved = improved[improved["score"] == "evidence"].set_index(
                "condition"
            )
            for condition in CONDITIONS:
                for metric in ("accuracy", "macro_f1"):
                    base_value = baseline.loc[condition, metric]
                    improved_value = improved.loc[condition, metric]
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "condition": condition,
                            "metric": metric,
                            "baseline": base_value,
                            "improved": improved_value,
                            "delta": improved_value - base_value,
                        }
                    )
    return pd.DataFrame(rows)


def risk_rows():
    rows = []
    for dataset, seeds in EXPERIMENTS.items():
        for seed, (_, improved_name) in seeds.items():
            frame = pd.read_csv(RUNS / improved_name / "learned_risk.csv")
            for condition in CONDITIONS:
                selected = frame[frame["condition"] == condition].set_index("score")
                learned = selected.loc["learned_evidence"]
                confidence = selected.loc["confidence"]
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "condition": condition,
                        "learned_aurc": learned["aurc"],
                        "confidence_aurc": confidence["aurc"],
                        "aurc_reduction": confidence["aurc"] - learned["aurc"],
                        "learned_top10_accuracy": learned["accuracy_at_10pct"],
                        "confidence_top10_accuracy": confidence[
                            "accuracy_at_10pct"
                        ],
                        "top10_accuracy_gain": learned["accuracy_at_10pct"]
                        - confidence["accuracy_at_10pct"],
                    }
                )
    return pd.DataFrame(rows)


def summarize_classification(rows):
    return (
        rows.groupby(["dataset", "condition", "metric"], sort=False)
        .agg(
            baseline_mean=("baseline", "mean"),
            baseline_std=("baseline", "std"),
            improved_mean=("improved", "mean"),
            improved_std=("improved", "std"),
            delta_mean=("delta", "mean"),
            delta_std=("delta", "std"),
            improved_seed_count=("delta", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )


def summarize_risk(rows):
    return (
        rows.groupby(["dataset", "condition"], sort=False)
        .agg(
            learned_aurc_mean=("learned_aurc", "mean"),
            learned_aurc_std=("learned_aurc", "std"),
            confidence_aurc_mean=("confidence_aurc", "mean"),
            confidence_aurc_std=("confidence_aurc", "std"),
            aurc_reduction_mean=("aurc_reduction", "mean"),
            learned_top10_accuracy_mean=("learned_top10_accuracy", "mean"),
            confidence_top10_accuracy_mean=("confidence_top10_accuracy", "mean"),
            top10_accuracy_gain_mean=("top10_accuracy_gain", "mean"),
            aurc_better_seed_count=(
                "aurc_reduction",
                lambda values: int((values > 0).sum()),
            ),
        )
        .reset_index()
    )


def pct(value):
    return f"{100 * value:.2f}%"


def mean_std(mean, std):
    return f"{100 * mean:.2f} +/- {100 * std:.2f}"


def write_report(class_summary, risk_summary):
    accuracy = class_summary[class_summary["metric"] == "accuracy"]
    lines = [
        "# Multi-dataset Research Progress",
        "",
        "## Protocol",
        "",
        "- CWRU: loads 0/1 train, load 2 calibration, load 3 test.",
        "- JNU: 600 rpm train, 800 rpm calibration, 1000 rpm test.",
        "- Paderborn: two operating conditions train, one calibration, one test; "
        "healthy K001, outer-race KA01, inner-race KI01.",
        "- Three independent seeds: 999, 1001, 1003.",
        "- Baseline: TFN with random 0-12 dB training noise.",
        "- Improved: frozen-backbone physical counterfactual TFconv calibration.",
        "",
        "## Accuracy",
        "",
        "| Dataset | Test | Baseline | Improved | Delta | Better seeds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in accuracy.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['condition']} "
            f"| {mean_std(row['baseline_mean'], row['baseline_std'])} "
            f"| {mean_std(row['improved_mean'], row['improved_std'])} "
            f"| {pct(row['delta_mean'])} "
            f"| {int(row['improved_seed_count'])}/3 |"
        )

    lines.extend(
        [
            "",
            "## Learned Risk At -6 dB",
            "",
            "| Dataset | Learned AURC | Confidence AURC | Reduction | "
            "Top-10 gain | Better AURC seeds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    severe = risk_summary[risk_summary["condition"] == "snr_-6"]
    for _, row in severe.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['learned_aurc_mean']:.4f} "
            f"| {row['confidence_aurc_mean']:.4f} "
            f"| {row['aurc_reduction_mean']:.4f} "
            f"| {pct(row['top10_accuracy_gain_mean'])} "
            f"| {int(row['aurc_better_seed_count'])}/3 |"
        )

    lines.extend(
        [
            "",
            "## Evidence Status",
            "",
            "- The method now runs under three genuinely cross-condition datasets.",
            "- Classification gains remain small and seed-dependent; this is not yet "
            "a strong standalone claim.",
            "- Severe-noise risk ranking is strong on CWRU and Paderborn, but only "
            "marginal on JNU. The claim should therefore focus on datasets where "
            "severe corruption creates meaningful confidence failures.",
            "- Three datasets now satisfy the minimum breadth for an SCI Q4 study, "
            "but stronger baselines, statistical tests, and ablations are still "
            "required before submission.",
        ]
    )
    (ROOT / "MULTIDATASET_PROGRESS_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    class_detail = classification_rows()
    risk_detail = risk_rows()
    class_summary = summarize_classification(class_detail)
    risk_summary = summarize_risk(risk_detail)
    class_detail.to_csv(RUNS / "multidataset_classification_by_seed.csv", index=False)
    class_summary.to_csv(
        RUNS / "multidataset_classification_summary.csv", index=False
    )
    risk_detail.to_csv(RUNS / "multidataset_risk_by_seed.csv", index=False)
    risk_summary.to_csv(RUNS / "multidataset_risk_summary.csv", index=False)
    write_report(class_summary, risk_summary)
    print(class_summary.to_string(index=False))
    print()
    print(risk_summary.to_string(index=False))


if __name__ == "__main__":
    main()
