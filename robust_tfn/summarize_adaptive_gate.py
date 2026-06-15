import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from robust_tfn.summarize_multidataset import CONDITIONS, EXPERIMENTS


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "research_runs"
METHOD = "adaptive_sequence"
REFERENCES = (
    "confidence",
    "negative_entropy",
    "probability_margin",
    "learned_evidence",
)


def main():
    metric_rows = []
    gate_rows = []
    metadata_rows = []
    for dataset, seeds in EXPERIMENTS.items():
        for seed, (run_name, _) in seeds.items():
            run = RUNS / run_name
            frame = pd.read_csv(run / "adaptive_risk.csv")
            gates = pd.read_csv(run / "adaptive_risk_gate.csv")
            metadata = json.loads(
                (run / "adaptive_risk_gate.json").read_text(encoding="utf-8")
            )
            for condition in CONDITIONS:
                selected = frame[frame["condition"] == condition].set_index(
                    "score"
                )
                adaptive = selected.loc[METHOD]
                for reference in REFERENCES:
                    baseline = selected.loc[reference]
                    metric_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "condition": condition,
                            "reference": reference,
                            "adaptive_aurc": adaptive["aurc"],
                            "reference_aurc": baseline["aurc"],
                            "aurc_reduction": baseline["aurc"]
                            - adaptive["aurc"],
                            "adaptive_top10_accuracy": adaptive[
                                "accuracy_at_10pct"
                            ],
                            "reference_top10_accuracy": baseline[
                                "accuracy_at_10pct"
                            ],
                            "top10_accuracy_gain": adaptive[
                                "accuracy_at_10pct"
                            ]
                            - baseline["accuracy_at_10pct"],
                        }
                    )
            gates["dataset"] = dataset
            gates["seed"] = seed
            gate_rows.append(gates)
            metadata_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "calibration_gate_roc_auc": metadata[
                        "calibration_gate_roc_auc"
                    ],
                    "sequence_gate_threshold": metadata[
                        "sequence_gate_threshold"
                    ],
                    "sequence_length": metadata["sequence_length"],
                }
            )

    detail = pd.DataFrame(metric_rows)
    gates = pd.concat(gate_rows, ignore_index=True)
    metadata = pd.DataFrame(metadata_rows)
    summary = (
        detail.groupby(["dataset", "condition", "reference"], sort=False)
        .agg(
            adaptive_aurc_mean=("adaptive_aurc", "mean"),
            adaptive_aurc_std=("adaptive_aurc", "std"),
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
                    "equal_pairs": int((delta == 0).sum()),
                    "wilcoxon_statistic": statistic,
                    "one_sided_p_value": p_value,
                }
            )
    tests = pd.DataFrame(tests)
    gate_summary = (
        gates.groupby(["dataset", "condition"], sort=False)
        .agg(
            mean_gate_probability=("mean_gate_probability", "mean"),
            sample_severe_rate=("severe_gate_rate", "mean"),
            sequence_severe_rate=("sequence_severe_rate", "mean"),
        )
        .reset_index()
    )

    detail.to_csv(RUNS / "adaptive_gate_by_seed.csv", index=False)
    summary.to_csv(RUNS / "adaptive_gate_summary.csv", index=False)
    tests.to_csv(RUNS / "adaptive_gate_paired_tests.csv", index=False)
    gates.to_csv(RUNS / "adaptive_gate_decisions.csv", index=False)
    gate_summary.to_csv(RUNS / "adaptive_gate_decision_summary.csv", index=False)
    metadata.to_csv(RUNS / "adaptive_gate_metadata.csv", index=False)
    print(summary.to_string(index=False))
    print()
    print(tests.to_string(index=False))
    print()
    print(gate_summary.to_string(index=False))


if __name__ == "__main__":
    main()
