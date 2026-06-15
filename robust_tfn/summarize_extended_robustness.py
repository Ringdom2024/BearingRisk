import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from robust_tfn.summarize_multidataset import EXPERIMENTS


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "research_runs"
METHOD = "adaptive_stable"
REFERENCES = (
    "confidence",
    "negative_entropy",
    "probability_margin",
    "learned_evidence",
)
LENGTH_METHODS = tuple(
    f"adaptive_sequence_{length}" for length in (1, 8, 32, 128)
)


def main():
    frames = []
    gates = []
    metadata_rows = []
    for dataset, seeds in EXPERIMENTS.items():
        for seed, (run_name, _) in seeds.items():
            run = RUNS / run_name
            frame = pd.read_csv(run / "extended_risk.csv")
            frame["dataset"] = dataset
            frame["seed"] = seed
            frames.append(frame)
            gate = pd.read_csv(run / "extended_risk_gate.csv")
            gate["dataset"] = dataset
            gate["seed"] = seed
            gates.append(gate)
            metadata = json.loads(
                (run / "extended_risk_gate.json").read_text(encoding="utf-8")
            )
            metadata_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "calibration_gate_roc_auc": metadata[
                        "calibration_gate_roc_auc"
                    ],
                    "sequence_probability_floor": metadata[
                        "sequence_probability_floor"
                    ],
                    "anomaly_threshold_multiplier": metadata[
                        "anomaly_threshold_multiplier"
                    ],
                    "stable_sequence_length": metadata[
                        "stable_sequence_length"
                    ],
                    "stable_activation_rate": metadata[
                        "stable_activation_rate"
                    ],
                }
            )
    frame = pd.concat(frames, ignore_index=True)
    gates = pd.concat(gates, ignore_index=True)

    rows = []
    for (dataset, seed, condition), selected in frame.groupby(
        ["dataset", "seed", "condition"], sort=False
    ):
        selected = selected.set_index("score")
        adaptive = selected.loc[METHOD]
        for reference in REFERENCES:
            baseline = selected.loc[reference]
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "condition": condition,
                    "reference": reference,
                    "adaptive_aurc": adaptive["aurc"],
                    "reference_aurc": baseline["aurc"],
                    "aurc_reduction": baseline["aurc"] - adaptive["aurc"],
                    "top10_accuracy_gain": adaptive["accuracy_at_10pct"]
                    - baseline["accuracy_at_10pct"],
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["dataset", "condition", "reference"], sort=False)
        .agg(
            adaptive_aurc_mean=("adaptive_aurc", "mean"),
            adaptive_aurc_std=("adaptive_aurc", "std"),
            reference_aurc_mean=("reference_aurc", "mean"),
            aurc_reduction_mean=("aurc_reduction", "mean"),
            top10_accuracy_gain_mean=("top10_accuracy_gain", "mean"),
            better_seed_count=("aurc_reduction", lambda x: int((x > 0).sum())),
            equal_seed_count=("aurc_reduction", lambda x: int((x == 0).sum())),
        )
        .reset_index()
    )
    tests = []
    for (condition, reference), selected in detail.groupby(
        ["condition", "reference"], sort=False
    ):
        delta = selected["aurc_reduction"]
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
                "equal_pairs": int((delta == 0).sum()),
                "wilcoxon_statistic": statistic,
                "one_sided_p_value": p_value,
            }
        )
    tests = pd.DataFrame(tests)
    length_summary = (
        frame[frame["score"].isin(LENGTH_METHODS)]
        .groupby(["dataset", "condition", "score"], sort=False)
        .agg(aurc_mean=("aurc", "mean"), aurc_std=("aurc", "std"))
        .reset_index()
    )
    gate_summary = (
        gates.groupby(["dataset", "condition"], sort=False)
        .agg(
            sample_severe_rate=("severe_gate_rate", "mean"),
            sequence_severe_rate=("sequence_severe_rate", "mean"),
            stable_severe_rate=("stable_severe_rate", "mean"),
            stable_candidate_rate=("stable_candidate_rate", "mean"),
            mean_gate_probability=("mean_gate_probability", "mean"),
            mean_anomaly_score=("mean_anomaly_score", "mean"),
        )
        .reset_index()
    )

    frame.to_csv(RUNS / "extended_all_scores.csv", index=False)
    detail.to_csv(RUNS / "extended_robustness_by_seed.csv", index=False)
    summary.to_csv(RUNS / "extended_robustness_summary.csv", index=False)
    tests.to_csv(RUNS / "extended_robustness_paired_tests.csv", index=False)
    length_summary.to_csv(
        RUNS / "extended_sequence_length_ablation.csv", index=False
    )
    gates.to_csv(RUNS / "extended_gate_decisions.csv", index=False)
    gate_summary.to_csv(RUNS / "extended_gate_summary.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(
        RUNS / "extended_gate_metadata.csv", index=False
    )
    print(summary.to_string(index=False))
    print()
    print(tests.to_string(index=False))
    print()
    print(gate_summary.to_string(index=False))


if __name__ == "__main__":
    main()
