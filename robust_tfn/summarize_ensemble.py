import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "research_runs"
DATASETS = ("CWRU", "JNU", "PADERBORN")


def main():
    rows = []
    metadata_rows = []
    for dataset in DATASETS:
        name = dataset.lower()
        frame = pd.read_csv(RUNS / f"ensemble_{name}.csv")
        frame["dataset"] = dataset
        rows.append(frame)
        metadata = json.loads(
            (RUNS / f"ensemble_{name}.json").read_text(encoding="utf-8")
        )
        metadata_rows.append(metadata)
    frame = pd.concat(rows, ignore_index=True)
    frame.to_csv(RUNS / "ensemble_uncertainty_summary.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(
        RUNS / "ensemble_uncertainty_metadata.csv", index=False
    )
    print(
        frame[
            [
                "dataset",
                "condition",
                "score",
                "accuracy",
                "macro_f1",
                "aurc",
                "accuracy_at_10pct",
            ]
        ].to_string(index=False)
    )
    print()
    print(pd.DataFrame(metadata_rows).to_string(index=False))


if __name__ == "__main__":
    main()
