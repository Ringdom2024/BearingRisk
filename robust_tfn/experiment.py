import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from robust_tfn.counterfactual import (
    counterfactual_evidence,
    counterfactual_ranking_loss,
)
from robust_tfn.data import CrossConditionDataset, get_dataset_spec
from robust_tfn.model import CounterfactualTFN
from robust_tfn.risk import calibrate_threshold, ranking_metrics, selective_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(
    dataset_name,
    data_root,
    batch_size,
    train_noise,
    train_seed=999,
    samples_per_class=None,
):
    spec = get_dataset_spec(dataset_name)
    train = CrossConditionDataset(
        data_root,
        spec.train_conditions,
        dataset_name=spec.name,
        train=True,
        random_noise_range=train_noise,
        seed=train_seed,
        samples_per_class=samples_per_class,
    )
    calibration = CrossConditionDataset(
        data_root,
        [spec.calibration_condition],
        dataset_name=spec.name,
        samples_per_class=samples_per_class,
    )
    tests = {
        "clean": CrossConditionDataset(
            data_root,
            [spec.test_condition],
            dataset_name=spec.name,
            samples_per_class=samples_per_class,
        ),
        "snr_6": CrossConditionDataset(
            data_root,
            [spec.test_condition],
            dataset_name=spec.name,
            noise_snr=6,
            samples_per_class=samples_per_class,
        ),
        "snr_0": CrossConditionDataset(
            data_root,
            [spec.test_condition],
            dataset_name=spec.name,
            noise_snr=0,
            samples_per_class=samples_per_class,
        ),
        "snr_-6": CrossConditionDataset(
            data_root,
            [spec.test_condition],
            dataset_name=spec.name,
            noise_snr=-6,
            samples_per_class=samples_per_class,
        ),
    }
    loader = lambda dataset, shuffle=False: DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader(train, True), loader(calibration), {
        name: loader(dataset) for name, dataset in tests.items()
    }


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    cf_weight,
    alignment_weight,
    class_weights,
    freeze_batchnorm=False,
):
    model.train()
    if freeze_batchnorm:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    totals = {"loss": 0.0, "correct": 0, "count": 0}
    for inputs, labels, rpms in loader:
        inputs, labels, rpms = inputs.to(device), labels.to(device), rpms.to(device)
        optimizer.zero_grad()
        if cf_weight > 0:
            output = counterfactual_ranking_loss(model, inputs, labels, rpms)
            classification = F.cross_entropy(
                output["logits"], labels, weight=class_weights
            )
            loss = (
                classification
                + cf_weight * output["ranking_loss"]
                + alignment_weight * output["alignment_loss"]
            )
            logits = output["logits"]
        else:
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
        loss.backward()
        optimizer.step()
        totals["loss"] += loss.item() * len(labels)
        totals["correct"] += (logits.argmax(1) == labels).sum().item()
        totals["count"] += len(labels)
    return totals["loss"] / totals["count"], totals["correct"] / totals["count"]


@torch.no_grad()
def basic_accuracy(model, loader, device):
    model.eval()
    correct = 0
    count = 0
    for inputs, labels, _ in loader:
        logits = model(inputs.to(device))
        correct += (logits.argmax(1).cpu() == labels).sum().item()
        count += len(labels)
    return correct / count


def signal_quality_features(inputs):
    signals = inputs[:, 0]
    power = torch.fft.rfft(signals, dim=1).abs().square()[:, 1:]
    power = power.clamp_min(1e-12)
    normalized_power = power / power.sum(dim=1, keepdim=True)
    frequencies = torch.linspace(
        0.0, 1.0, power.shape[1], device=inputs.device
    ).unsqueeze(0)
    spectral_flatness = power.log().mean(dim=1).exp() / power.mean(dim=1)
    spectral_entropy = -(
        normalized_power * normalized_power.log()
    ).sum(dim=1) / np.log(power.shape[1])
    high_frequency_ratio = normalized_power[
        :, power.shape[1] // 2 :
    ].sum(dim=1)
    spectral_centroid = (normalized_power * frequencies).sum(dim=1)
    spectral_bandwidth = torch.sqrt(
        (
            normalized_power
            * (frequencies - spectral_centroid.unsqueeze(1)).square()
        ).sum(dim=1)
    )
    absolute_difference = torch.diff(signals, dim=1).abs().mean(dim=1)
    zero_crossing_rate = (
        signals[:, 1:] * signals[:, :-1] < 0
    ).float().mean(dim=1)
    kurtosis = signals.pow(4).mean(dim=1)
    crest_factor = signals.abs().amax(dim=1)
    lag1_correlation = (signals[:, 1:] * signals[:, :-1]).mean(dim=1)
    lag4_correlation = (signals[:, 4:] * signals[:, :-4]).mean(dim=1)
    return torch.stack(
        [
            spectral_flatness,
            spectral_entropy,
            high_frequency_ratio,
            spectral_centroid,
            spectral_bandwidth,
            absolute_difference,
            zero_crossing_rate,
            kurtosis,
            crest_factor,
            lag1_correlation,
            lag4_correlation,
        ],
        dim=1,
    )


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    labels_all, predictions_all = [], []
    evidence_all, confidence_all = [], []
    entropy_all, margin_all = [], []
    quality_features_all = []
    related_drop_all, control_change_all = [], []
    for inputs, labels, rpms in loader:
        inputs, rpms = inputs.to(device), rpms.to(device)
        quality_features_all.append(
            signal_quality_features(inputs).cpu().numpy()
        )
        details = model(inputs, return_details=True)
        logits = details["logits"]
        probabilities = F.softmax(logits, dim=1)
        predicted = logits.argmax(1)
        evidence, confidence, related_drop, control_change = counterfactual_evidence(
            model,
            inputs,
            predicted,
            rpms,
            original_logits=logits,
            original_tf_response=details["tf_response"],
        )
        top_probabilities = probabilities.topk(k=2, dim=1).values
        negative_entropy = (
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=1)
        labels_all.append(labels.numpy())
        predictions_all.append(predicted.cpu().numpy())
        evidence_all.append(evidence.cpu().numpy())
        confidence_all.append(confidence.cpu().numpy())
        entropy_all.append(negative_entropy.cpu().numpy())
        margin_all.append(
            (top_probabilities[:, 0] - top_probabilities[:, 1]).cpu().numpy()
        )
        related_drop_all.append(related_drop.cpu().numpy())
        control_change_all.append(control_change.cpu().numpy())
    labels = np.concatenate(labels_all)
    predictions = np.concatenate(predictions_all)
    return {
        "labels": labels,
        "predictions": predictions,
        "evidence": np.concatenate(evidence_all),
        "confidence": np.concatenate(confidence_all),
        "negative_entropy": np.concatenate(entropy_all),
        "probability_margin": np.concatenate(margin_all),
        "signal_quality_features": np.concatenate(quality_features_all),
        "related_drop": np.concatenate(related_drop_all),
        "control_change": np.concatenate(control_change_all),
        "correct": (labels == predictions).astype(float),
    }


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_dataset_spec(args.dataset)
    train_loader, calibration_loader, test_loaders = make_loaders(
        spec.name,
        args.data_root,
        args.batch_size,
        None if args.train_noise_min is None else (args.train_noise_min, args.train_noise_max),
        args.seed,
        args.samples_per_class,
    )
    model = CounterfactualTFN(
        mid_channel=args.mid_channel,
        num_classes=spec.num_classes,
        sample_rate=spec.sample_rate,
        fault_ratios=spec.fault_ratios,
    ).to(device)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(state)
    if args.freeze_non_tfconv:
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.backbone.funconv.superparams.requires_grad = True
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
    counts = np.bincount(
        train_loader.dataset.labels, minlength=spec.num_classes
    )
    class_weights = torch.tensor(
        len(train_loader.dataset) / (spec.num_classes * counts),
        dtype=torch.float32,
        device=device,
    )

    best_state = None
    best_accuracy = -1.0
    history = []
    start = time.time()
    for epoch in range(args.epochs):
        loss, accuracy = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.cf_weight,
            args.alignment_weight,
            class_weights,
            args.freeze_batchnorm,
        )
        calibration_accuracy = basic_accuracy(model, calibration_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss,
                "train_accuracy": accuracy,
                "calibration_accuracy": calibration_accuracy,
            }
        )
        if calibration_accuracy > best_accuracy:
            best_accuracy = calibration_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        scheduler.step()
        print(
            f"epoch={epoch + 1:02d} loss={loss:.4f} "
            f"train_acc={accuracy:.4f} calibration_acc={calibration_accuracy:.4f}"
        )

    model.load_state_dict(best_state)
    calibration = collect_predictions(model, calibration_loader, device)
    evidence_threshold, evidence_coverage, evidence_risk = calibrate_threshold(
        calibration["evidence"], calibration["correct"], args.target_risk
    )
    confidence_threshold, confidence_coverage, confidence_risk = calibrate_threshold(
        calibration["confidence"], calibration["correct"], args.target_risk
    )

    rows = []
    for test_name, loader in test_loaders.items():
        result = collect_predictions(model, loader, device)
        base = {
            "condition": test_name,
            "accuracy": accuracy_score(result["labels"], result["predictions"]),
            "macro_f1": f1_score(
                result["labels"], result["predictions"], average="macro"
            ),
        }
        for score_name, threshold in (
            ("evidence", evidence_threshold),
            ("confidence", confidence_threshold),
        ):
            row = dict(base)
            row["score"] = score_name
            row.update(
                selective_metrics(result[score_name], result["correct"], threshold)
            )
            row.update(ranking_metrics(result[score_name], result["correct"]))
            rows.append(row)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output / "best_model.pth")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "args": vars(args),
        "device": str(device),
        "training_seconds": time.time() - start,
        "best_calibration_accuracy": best_accuracy,
        "evidence_threshold": evidence_threshold,
        "evidence_calibration_coverage": evidence_coverage,
        "evidence_calibration_risk": evidence_risk,
        "confidence_threshold": confidence_threshold,
        "confidence_calibration_coverage": confidence_coverage,
        "confidence_calibration_risk": confidence_risk,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    for row in rows:
        print(row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="CWRU", choices=["CWRU", "JNU", "PADERBORN"]
    )
    parser.add_argument("--data-root", default="./Datasets_dir/CWRU")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mid-channel", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cf-weight", type=float, default=0.0)
    parser.add_argument("--alignment-weight", type=float, default=0.0)
    parser.add_argument("--target-risk", type=float, default=0.05)
    parser.add_argument("--train-noise-min", type=float)
    parser.add_argument("--train-noise-max", type=float)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--freeze-non-tfconv", action="store_true")
    parser.add_argument("--freeze-batchnorm", action="store_true")
    parser.add_argument("--samples-per-class", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
