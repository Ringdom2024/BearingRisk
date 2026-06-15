import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from robust_tfn.data import CrossConditionDataset, get_dataset_spec
from robust_tfn.learned_risk import loader
from robust_tfn.model import CounterfactualTFN
from robust_tfn.risk import calibrate_threshold, ranking_metrics, selective_metrics


@torch.no_grad()
def collect(models, data_loader, device):
    labels_all = []
    predictions_all = []
    score_parts = {
        "ensemble_confidence": [],
        "ensemble_negative_entropy": [],
        "ensemble_probability_margin": [],
        "negative_mutual_information": [],
    }
    for inputs, labels, _ in data_loader:
        inputs = inputs.to(device)
        member_probabilities = torch.stack(
            [F.softmax(model(inputs), dim=1) for model in models]
        )
        probability = member_probabilities.mean(dim=0)
        prediction = probability.argmax(dim=1)
        top_probabilities = probability.topk(k=2, dim=1).values
        predictive_entropy = -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(dim=1)
        member_entropy = -(
            member_probabilities
            * member_probabilities.clamp_min(1e-8).log()
        ).sum(dim=2).mean(dim=0)
        mutual_information = predictive_entropy - member_entropy

        labels_all.append(labels.numpy())
        predictions_all.append(prediction.cpu().numpy())
        score_parts["ensemble_confidence"].append(
            probability.max(dim=1).values.cpu().numpy()
        )
        score_parts["ensemble_negative_entropy"].append(
            (-predictive_entropy).cpu().numpy()
        )
        score_parts["ensemble_probability_margin"].append(
            (top_probabilities[:, 0] - top_probabilities[:, 1]).cpu().numpy()
        )
        score_parts["negative_mutual_information"].append(
            (-mutual_information).cpu().numpy()
        )
    labels = np.concatenate(labels_all)
    predictions = np.concatenate(predictions_all)
    return {
        "labels": labels,
        "predictions": predictions,
        "correct": (labels == predictions).astype(float),
        **{
            name: np.concatenate(parts)
            for name, parts in score_parts.items()
        },
    }


def dataset_for_condition(args, spec, condition):
    options = {
        "clean": {"noise_snr": None},
        "snr_6": {"noise_snr": 6},
        "snr_0": {"noise_snr": 0},
        "snr_-6": {"noise_snr": -6},
        "mixed_snr": {
            "corruption": "mixed_gaussian",
            "mixed_segment_length": args.mixed_segment_length,
        },
        "colored_-6": {"corruption": "colored", "corruption_snr": -6},
        "impulse_-6": {"corruption": "impulse", "corruption_snr": -6},
        "mechanical_-6": {
            "corruption": "mechanical",
            "corruption_snr": -6,
        },
    }[condition]
    return CrossConditionDataset(
        args.data_root,
        [spec.test_condition],
        dataset_name=spec.name,
        samples_per_class=args.samples_per_class,
        **options,
    )


def main(args):
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_dataset_spec(args.dataset)
    models = []
    for checkpoint in args.checkpoints:
        model = CounterfactualTFN(
            mid_channel=args.mid_channel,
            num_classes=spec.num_classes,
            sample_rate=spec.sample_rate,
            fault_ratios=spec.fault_ratios,
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.eval()
        models.append(model)

    calibration_parts = []
    for snr in (None, 6, 0, -6):
        dataset = CrossConditionDataset(
            args.data_root,
            [spec.calibration_condition],
            dataset_name=spec.name,
            noise_snr=snr,
            samples_per_class=args.samples_per_class,
        )
        calibration_parts.append(
            collect(models, loader(dataset, args.batch_size), device)
        )
    score_names = [
        "ensemble_confidence",
        "ensemble_negative_entropy",
        "ensemble_probability_margin",
        "negative_mutual_information",
    ]
    calibration_correct = np.concatenate(
        [part["correct"] for part in calibration_parts]
    )
    thresholds = {}
    for score_name in score_names:
        score = np.concatenate(
            [part[score_name] for part in calibration_parts]
        )
        thresholds[score_name] = calibrate_threshold(
            score, calibration_correct, args.target_risk
        )[0]

    conditions = ["clean", "snr_6", "snr_0", "snr_-6"]
    if args.extended_corruptions:
        conditions.extend(
            ["mixed_snr", "colored_-6", "impulse_-6", "mechanical_-6"]
        )
    rows = []
    for condition in conditions:
        result = collect(
            models,
            loader(
                dataset_for_condition(args, spec, condition),
                args.batch_size,
            ),
            device,
        )
        for score_name in score_names:
            row = {
                "condition": condition,
                "score": score_name,
                "accuracy": accuracy_score(
                    result["labels"], result["predictions"]
                ),
                "macro_f1": f1_score(
                    result["labels"],
                    result["predictions"],
                    average="macro",
                ),
            }
            row.update(
                selective_metrics(
                    result[score_name],
                    result["correct"],
                    thresholds[score_name],
                )
            )
            row.update(
                ranking_metrics(result[score_name], result["correct"])
            )
            rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "dataset": spec.name,
        "checkpoints": args.checkpoints,
        "member_count": len(models),
        "device": str(device),
        "evaluation_seconds": time.time() - start,
        "parameter_count_per_member": sum(
            parameter.numel() for parameter in models[0].parameters()
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    for row in rows:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=["CWRU", "JNU", "PADERBORN"]
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mid-channel", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-risk", type=float, default=0.05)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--extended-corruptions", action="store_true")
    parser.add_argument("--mixed-segment-length", type=int, default=16)
    main(parser.parse_args())
