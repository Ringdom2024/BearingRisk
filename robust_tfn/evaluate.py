import argparse
import csv
from pathlib import Path

import torch

from robust_tfn.data import get_dataset_spec
from robust_tfn.experiment import collect_predictions, make_loaders
from robust_tfn.model import CounterfactualTFN
from robust_tfn.risk import (
    calibrate_threshold,
    ranking_metrics,
    selective_metrics,
)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_dataset_spec(args.dataset)
    _, calibration_loader, test_loaders = make_loaders(
        spec.name,
        args.data_root,
        args.batch_size,
        None,
        999,
        args.samples_per_class,
    )
    model = CounterfactualTFN(
        mid_channel=args.mid_channel,
        num_classes=spec.num_classes,
        sample_rate=spec.sample_rate,
        fault_ratios=spec.fault_ratios,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    calibration = collect_predictions(model, calibration_loader, device)
    thresholds = {}
    for score_name in ("evidence", "confidence"):
        thresholds[score_name] = calibrate_threshold(
            calibration[score_name], calibration["correct"], args.target_risk
        )[0]

    rows = []
    for condition, loader in test_loaders.items():
        result = collect_predictions(model, loader, device)
        for score_name in ("evidence", "confidence"):
            row = {"condition": condition, "score": score_name}
            row.update(
                selective_metrics(
                    result[score_name],
                    result["correct"],
                    thresholds[score_name],
                )
            )
            row.update(ranking_metrics(result[score_name], result["correct"]))
            rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dataset", default="CWRU", choices=["CWRU", "JNU", "PADERBORN"]
    )
    parser.add_argument("--data-root", default="./Datasets_dir/CWRU")
    parser.add_argument("--mid-channel", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-risk", type=float, default=0.05)
    parser.add_argument("--samples-per-class", type=int)
    main(parser.parse_args())
