import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from robust_tfn.data import CrossConditionDataset, get_dataset_spec
from robust_tfn.experiment import collect_predictions, signal_quality_features
from robust_tfn.model import CounterfactualTFN
from robust_tfn.risk import calibrate_threshold, ranking_metrics, selective_metrics


def features(result):
    confidence = np.clip(result["confidence"], 1e-6, 1 - 1e-6)
    confidence_logit = np.log(confidence / (1 - confidence))
    related_drop = result["related_drop"]
    control_change = result["control_change"]
    return np.column_stack(
        [
            confidence_logit,
            related_drop,
            control_change,
            related_drop - control_change,
            related_drop * confidence,
        ]
    )


def adaptive_features(result, gate_probability):
    base = features(result)
    gate_probability = np.asarray(gate_probability).reshape(-1, 1)
    return np.column_stack(
        [
            base,
            gate_probability,
            base * gate_probability,
        ]
    )


def corruption_gate_features(result):
    return np.column_stack(
        [
            result["signal_quality_features"],
            result["confidence"],
            result["negative_entropy"],
            result["probability_margin"],
            np.abs(result["related_drop"]),
            result["control_change"],
        ]
    )


def loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def collect_quality(dataset, batch_size, device):
    parts = []
    for inputs, _, _ in loader(dataset, batch_size):
        parts.append(signal_quality_features(inputs.to(device)).cpu().numpy())
    return np.concatenate(parts)


def score_calibrator(scores, correct):
    model = LogisticRegression(max_iter=1000, random_state=999)
    model.fit(np.asarray(scores).reshape(-1, 1), correct)
    return model


def calibrated_score(model, scores):
    return model.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]


def select_gate_threshold(
    gate_probability,
    learned_scores,
    confidence_scores,
    calibration_parts,
    tolerance,
):
    lengths = [len(part["correct"]) for part in calibration_parts]
    offsets = np.cumsum([0] + lengths)
    candidates = np.unique(
        np.concatenate(
            [
                np.quantile(gate_probability, np.linspace(0.0, 1.0, 201)),
                np.asarray([0.5]),
            ]
        )
    )
    evaluations = []
    for threshold in candidates:
        aurcs = []
        confidence_aurcs = []
        for index, part in enumerate(calibration_parts):
            section = slice(offsets[index], offsets[index + 1])
            score = np.where(
                gate_probability[section] >= threshold,
                learned_scores[section],
                confidence_scores[section],
            )
            aurcs.append(ranking_metrics(score, part["correct"])["aurc"])
            confidence_aurcs.append(
                ranking_metrics(
                    confidence_scores[section], part["correct"]
                )["aurc"]
            )
        nonsevere_regret = np.asarray(aurcs[:3]) - np.asarray(
            confidence_aurcs[:3]
        )
        evaluations.append(
            {
                "threshold": float(threshold),
                "severe_aurc": aurcs[3],
                "max_nonsevere_regret": float(nonsevere_regret.max()),
                "mean_nonsevere_regret": float(nonsevere_regret.mean()),
            }
        )
    valid = [
        item
        for item in evaluations
        if item["max_nonsevere_regret"] <= tolerance
    ]
    pool = valid or evaluations
    selected = min(
        pool,
        key=lambda item: (
            item["severe_aurc"],
            item["max_nonsevere_regret"],
            item["mean_nonsevere_regret"],
        ),
    )
    selected["constraint_satisfied"] = bool(valid)
    return selected


def sequence_gate_scores(
    gate_probability,
    learned_scores,
    confidence_scores,
    threshold,
    sequence_length,
    anomaly_score=None,
    anomaly_threshold=None,
):
    scores = np.empty_like(learned_scores)
    decisions = np.zeros(len(learned_scores), dtype=bool)
    for start in range(0, len(scores), sequence_length):
        end = min(start + sequence_length, len(scores))
        severe = gate_probability[start:end].mean() >= threshold
        if anomaly_score is not None and anomaly_threshold is not None:
            severe = severe or (
                anomaly_score[start:end].mean() >= anomaly_threshold
            )
        decisions[start:end] = severe
        source = learned_scores if severe else confidence_scores
        scores[start:end] = source[start:end]
    return scores, decisions


def sequence_anomaly_threshold(scores, lengths, sequence_length):
    means = []
    offset = 0
    for length in lengths:
        end = offset + length
        section = scores[offset:end]
        for start in range(0, length, sequence_length):
            means.append(section[start : start + sequence_length].mean())
        offset = end
    return float(np.nextafter(max(means), np.inf))


def stabilized_gate_scores(
    decisions,
    learned_scores,
    confidence_scores,
    activation_rate=0.2,
):
    severe_rate = float(decisions.mean())
    if severe_rate < activation_rate:
        stable_decisions = np.zeros_like(decisions)
    else:
        stable_decisions = np.ones_like(decisions)
    return (
        np.where(stable_decisions, learned_scores, confidence_scores),
        stable_decisions,
    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_dataset_spec(args.dataset)
    model = CounterfactualTFN(
        mid_channel=args.mid_channel,
        num_classes=spec.num_classes,
        sample_rate=spec.sample_rate,
        fault_ratios=spec.fault_ratios,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    calibration_parts = []
    calibration_severe = []
    for snr in (None, 6, 0, -6):
        dataset = CrossConditionDataset(
            args.data_root,
            [spec.calibration_condition],
            dataset_name=spec.name,
            noise_snr=snr,
            samples_per_class=args.samples_per_class,
        )
        result = collect_predictions(model, loader(dataset, args.batch_size), device)
        calibration_parts.append(result)
        calibration_severe.append(
            np.full(len(result["correct"]), float(snr == -6))
        )
    calibration_features = np.concatenate([features(item) for item in calibration_parts])
    calibration_correct = np.concatenate([item["correct"] for item in calibration_parts])
    calibration_quality = np.concatenate(
        [item["signal_quality_features"] for item in calibration_parts]
    )
    calibration_gate_features = np.concatenate(
        [corruption_gate_features(item) for item in calibration_parts]
    )
    calibration_severe = np.concatenate(calibration_severe)
    gate_training_features = [calibration_gate_features]
    gate_training_labels = [calibration_severe]
    auxiliary_gate_corruptions = []
    if args.extended_corruptions:
        for corruption in ("uniform", "burst_dropout", "chirp"):
            dataset = CrossConditionDataset(
                args.data_root,
                [spec.calibration_condition],
                dataset_name=spec.name,
                corruption=corruption,
                corruption_snr=-6,
                samples_per_class=args.samples_per_class,
            )
            result = collect_predictions(
                model, loader(dataset, args.batch_size), device
            )
            gate_training_features.append(corruption_gate_features(result))
            gate_training_labels.append(
                np.ones(len(result["correct"]), dtype=float)
            )
            auxiliary_gate_corruptions.append(corruption)
    reference_quality_parts = [
        part["signal_quality_features"] for part in calibration_parts[:3]
    ]
    for condition in spec.train_conditions:
        reference_dataset = CrossConditionDataset(
            args.data_root,
            [condition],
            dataset_name=spec.name,
            samples_per_class=args.samples_per_class,
        )
        reference_quality_parts.append(
            collect_quality(reference_dataset, args.batch_size, device)
        )
    reference_quality = np.concatenate(reference_quality_parts)
    quality_scaler = StandardScaler().fit(reference_quality)
    quality_distance_model = LedoitWolf().fit(
        quality_scaler.transform(reference_quality)
    )
    calibration_anomaly_score = quality_distance_model.mahalanobis(
        quality_scaler.transform(calibration_quality)
    )
    reference_lengths = [len(part) for part in reference_quality_parts]
    reference_anomaly_score = quality_distance_model.mahalanobis(
        quality_scaler.transform(reference_quality)
    )

    calibrator = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=999),
    )
    calibrator.fit(calibration_features, calibration_correct)
    gate = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=999
        ),
    )
    gate.fit(
        np.concatenate(gate_training_features),
        np.concatenate(gate_training_labels),
    )
    calibration_gate_probability = gate.predict_proba(
        calibration_gate_features
    )[:, 1]
    adaptive_calibrator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=999
        ),
    )
    adaptive_calibrator.fit(
        adaptive_features(
            {
                key: np.concatenate([part[key] for part in calibration_parts])
                for key in (
                    "confidence",
                    "related_drop",
                    "control_change",
                )
            },
            calibration_gate_probability,
        ),
        calibration_correct,
    )
    nonsevere_confidence_calibrator = score_calibrator(
        np.concatenate(
            [part["confidence"] for part in calibration_parts[:3]]
        ),
        np.concatenate([part["correct"] for part in calibration_parts[:3]]),
    )
    severe_calibrator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=999
        ),
    )
    severe_calibrator.fit(
        features(calibration_parts[3]), calibration_parts[3]["correct"]
    )
    learned_calibration_score = calibrator.predict_proba(calibration_features)[:, 1]
    raw_confidence_calibration_score = np.concatenate(
        [item["confidence"] for item in calibration_parts]
    )
    learned_score_calibrator = score_calibrator(
        learned_calibration_score, calibration_correct
    )
    confidence_score_calibrator = score_calibrator(
        raw_confidence_calibration_score, calibration_correct
    )
    aligned_learned_calibration_score = calibrated_score(
        learned_score_calibrator, learned_calibration_score
    )
    aligned_confidence_calibration_score = calibrated_score(
        confidence_score_calibrator, raw_confidence_calibration_score
    )
    adaptive_calibration_score = adaptive_calibrator.predict_proba(
        adaptive_features(
            {
                key: np.concatenate([part[key] for part in calibration_parts])
                for key in (
                    "confidence",
                    "related_drop",
                    "control_change",
                )
            },
            calibration_gate_probability,
        )
    )[:, 1]
    nonsevere_expert_calibration_score = calibrated_score(
        nonsevere_confidence_calibrator,
        raw_confidence_calibration_score,
    )
    severe_expert_calibration_score = np.concatenate(
        [
            severe_calibrator.predict_proba(features(part))[:, 1]
            for part in calibration_parts
        ]
    )
    mixture_calibration_score = (
        calibration_gate_probability * severe_expert_calibration_score
        + (1.0 - calibration_gate_probability)
        * nonsevere_expert_calibration_score
    )
    calibration_gate_means = []
    offset = 0
    for part in calibration_parts:
        end = offset + len(part["correct"])
        calibration_gate_means.append(
            float(calibration_gate_probability[offset:end].mean())
        )
        offset = end
    sequence_gate_threshold = 0.5 * (
        max(calibration_gate_means[:3]) + calibration_gate_means[3]
    )
    sequence_gate_threshold = max(
        sequence_gate_threshold, args.sequence_probability_floor
    )
    sequence_lengths = sorted(
        {
            int(value)
            for value in args.gate_sequence_lengths.split(",")
            if value.strip()
        }
        | {args.gate_sequence_length}
    )
    sequence_calibration_scores = {}
    sequence_calibration_decisions = {}
    sequence_anomaly_thresholds = {}
    for sequence_length in sequence_lengths:
        anomaly_threshold = sequence_anomaly_threshold(
            reference_anomaly_score,
            reference_lengths,
            sequence_length,
        ) * args.anomaly_threshold_multiplier
        sequence_anomaly_thresholds[sequence_length] = anomaly_threshold
        sequence_calibration_parts = []
        offset = 0
        for part in calibration_parts:
            end = offset + len(part["correct"])
            score, decisions = sequence_gate_scores(
                calibration_gate_probability[offset:end],
                aligned_learned_calibration_score[offset:end],
                aligned_confidence_calibration_score[offset:end],
                sequence_gate_threshold,
                sequence_length,
                (
                    calibration_anomaly_score[offset:end]
                    if args.use_anomaly_gate
                    else None
                ),
                anomaly_threshold if args.use_anomaly_gate else None,
            )
            sequence_calibration_parts.append(score)
            sequence_calibration_decisions.setdefault(
                sequence_length, []
            ).append(decisions)
            offset = end
        sequence_calibration_scores[sequence_length] = np.concatenate(
            sequence_calibration_parts
        )
    gate_selection = None
    gate_threshold = args.gate_threshold
    if gate_threshold is None:
        gate_selection = select_gate_threshold(
            calibration_gate_probability,
            aligned_learned_calibration_score,
            aligned_confidence_calibration_score,
            calibration_parts,
            args.max_nonsevere_regret,
        )
        gate_threshold = gate_selection["threshold"]
    calibration_hard_gate = calibration_gate_probability >= gate_threshold
    calibration_scores = {
        "learned_evidence": learned_calibration_score,
        "confidence": raw_confidence_calibration_score,
        "negative_entropy": np.concatenate(
            [item["negative_entropy"] for item in calibration_parts]
        ),
        "probability_margin": np.concatenate(
            [item["probability_margin"] for item in calibration_parts]
        ),
        "adaptive_meta": adaptive_calibration_score,
        "mixture_experts": mixture_calibration_score,
    }
    for sequence_length, score in sequence_calibration_scores.items():
        calibration_scores[f"adaptive_sequence_{sequence_length}"] = score
    calibration_scores["adaptive_sequence"] = sequence_calibration_scores[
        args.gate_sequence_length
    ]
    stable_calibration_parts = []
    offset = 0
    stable_length = args.stable_sequence_length
    for part, decisions in zip(
        calibration_parts,
        sequence_calibration_decisions[stable_length],
    ):
        end = offset + len(part["correct"])
        long_decisions = sequence_calibration_decisions[
            args.gate_sequence_length
        ][len(stable_calibration_parts)]
        score, _ = stabilized_gate_scores(
            np.logical_or(decisions, long_decisions),
            aligned_learned_calibration_score[offset:end],
            aligned_confidence_calibration_score[offset:end],
            args.stable_activation_rate,
        )
        stable_calibration_parts.append(score)
        offset = end
    calibration_scores["adaptive_stable"] = np.concatenate(
        stable_calibration_parts
    )
    calibration_scores["adaptive_hard"] = np.where(
        calibration_hard_gate,
        aligned_learned_calibration_score,
        aligned_confidence_calibration_score,
    )
    calibration_scores["adaptive_soft"] = (
        calibration_gate_probability * aligned_learned_calibration_score
        + (1.0 - calibration_gate_probability)
        * aligned_confidence_calibration_score
    )
    thresholds = {
        name: calibrate_threshold(score, calibration_correct, args.target_risk)[0]
        for name, score in calibration_scores.items()
    }

    test_conditions = [
        ("clean", {"noise_snr": None}),
        ("snr_6", {"noise_snr": 6}),
        ("snr_0", {"noise_snr": 0}),
        ("snr_-6", {"noise_snr": -6}),
    ]
    if args.extended_corruptions:
        test_conditions.extend(
            [
                (
                    "mixed_snr",
                    {
                        "corruption": "mixed_gaussian",
                        "mixed_segment_length": args.mixed_segment_length,
                    },
                ),
                (
                    "colored_-6",
                    {"corruption": "colored", "corruption_snr": -6},
                ),
                (
                    "impulse_-6",
                    {"corruption": "impulse", "corruption_snr": -6},
                ),
                (
                    "mechanical_-6",
                    {"corruption": "mechanical", "corruption_snr": -6},
                ),
            ]
        )

    rows = []
    gate_rows = []
    for name, corruption_args in test_conditions:
        dataset = CrossConditionDataset(
            args.data_root,
            [spec.test_condition],
            dataset_name=spec.name,
            samples_per_class=args.samples_per_class,
            **corruption_args,
        )
        result = collect_predictions(model, loader(dataset, args.batch_size), device)
        gate_probability = gate.predict_proba(
            corruption_gate_features(result)
        )[:, 1]
        anomaly_score = quality_distance_model.mahalanobis(
            quality_scaler.transform(result["signal_quality_features"])
        )
        hard_gate = gate_probability >= gate_threshold
        learned_score = calibrator.predict_proba(features(result))[:, 1]
        adaptive_score = adaptive_calibrator.predict_proba(
            adaptive_features(result, gate_probability)
        )[:, 1]
        severe_expert_score = severe_calibrator.predict_proba(
            features(result)
        )[:, 1]
        nonsevere_expert_score = calibrated_score(
            nonsevere_confidence_calibrator, result["confidence"]
        )
        mixture_score = (
            gate_probability * severe_expert_score
            + (1.0 - gate_probability) * nonsevere_expert_score
        )
        aligned_learned_score = calibrated_score(
            learned_score_calibrator, learned_score
        )
        aligned_confidence_score = calibrated_score(
            confidence_score_calibrator, result["confidence"]
        )
        sequence_scores = {}
        sequence_gates = {}
        for sequence_length in sequence_lengths:
            sequence_scores[sequence_length], sequence_gates[sequence_length] = (
                sequence_gate_scores(
                    gate_probability,
                    aligned_learned_score,
                    aligned_confidence_score,
                    sequence_gate_threshold,
                    sequence_length,
                    anomaly_score if args.use_anomaly_gate else None,
                    (
                        sequence_anomaly_thresholds[sequence_length]
                        if args.use_anomaly_gate
                        else None
                    ),
                )
            )
        sequence_score = sequence_scores[args.gate_sequence_length]
        sequence_gate = sequence_gates[args.gate_sequence_length]
        stable_candidate_gate = np.logical_or(
            sequence_gates[args.stable_sequence_length],
            sequence_gates[args.gate_sequence_length],
        )
        stable_score, stable_gate = stabilized_gate_scores(
            stable_candidate_gate,
            aligned_learned_score,
            aligned_confidence_score,
            args.stable_activation_rate,
        )
        scores = {
            "learned_evidence": learned_score,
            "confidence": result["confidence"],
            "negative_entropy": result["negative_entropy"],
            "probability_margin": result["probability_margin"],
            "adaptive_meta": adaptive_score,
            "mixture_experts": mixture_score,
            "adaptive_sequence": sequence_score,
            "adaptive_stable": stable_score,
            "adaptive_hard": np.where(
                hard_gate, aligned_learned_score, aligned_confidence_score
            ),
            "adaptive_soft": (
                gate_probability * aligned_learned_score
                + (1.0 - gate_probability) * aligned_confidence_score
            ),
        }
        for sequence_length, score in sequence_scores.items():
            scores[f"adaptive_sequence_{sequence_length}"] = score
        for score_name, score in scores.items():
            row = {
                "condition": name,
                "score": score_name,
                "mean_gate_probability": float(gate_probability.mean()),
                "severe_gate_rate": float(hard_gate.mean()),
                "sequence_severe_rate": float(sequence_gate.mean()),
                "stable_severe_rate": float(stable_gate.mean()),
                "stable_candidate_rate": float(
                    stable_candidate_gate.mean()
                ),
                "mean_anomaly_score": float(anomaly_score.mean()),
            }
            row.update(selective_metrics(score, result["correct"], thresholds[score_name]))
            row.update(ranking_metrics(score, result["correct"]))
            rows.append(row)
        true_severe_value = int(
            name
            in ("snr_-6", "colored_-6", "impulse_-6", "mechanical_-6")
        )
        true_severe = np.full(len(hard_gate), true_severe_value)
        gate_rows.append(
            {
                "condition": name,
                "true_severe": true_severe_value,
                "mean_gate_probability": float(gate_probability.mean()),
                "severe_gate_rate": float(hard_gate.mean()),
                "sequence_severe_rate": float(sequence_gate.mean()),
                "stable_severe_rate": float(stable_gate.mean()),
                "stable_candidate_rate": float(
                    stable_candidate_gate.mean()
                ),
                "mean_anomaly_score": float(anomaly_score.mean()),
                "gate_accuracy": accuracy_score(true_severe, hard_gate),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    gate_output = output.with_name(f"{output.stem}_gate.csv")
    with gate_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=gate_rows[0].keys())
        writer.writeheader()
        writer.writerows(gate_rows)
    gate_metadata = {
        "gate_threshold": gate_threshold,
        "gate_selection": gate_selection,
        "max_nonsevere_regret": args.max_nonsevere_regret,
        "calibration_gate_accuracy": accuracy_score(
            calibration_severe, calibration_hard_gate
        ),
        "calibration_gate_roc_auc": roc_auc_score(
            calibration_severe, calibration_gate_probability
        ),
        "quality_feature_count": calibration_quality.shape[1],
        "gate_feature_count": calibration_gate_features.shape[1],
        "use_anomaly_gate": args.use_anomaly_gate,
        "anomaly_threshold_multiplier": args.anomaly_threshold_multiplier,
        "auxiliary_gate_corruptions": auxiliary_gate_corruptions,
        "sequence_gate_threshold": sequence_gate_threshold,
        "sequence_probability_floor": args.sequence_probability_floor,
        "sequence_length": args.gate_sequence_length,
        "sequence_lengths": sequence_lengths,
        "stable_sequence_length": args.stable_sequence_length,
        "stable_activation_rate": args.stable_activation_rate,
        "sequence_anomaly_thresholds": sequence_anomaly_thresholds,
        "calibration_gate_means": calibration_gate_means,
    }
    output.with_name(f"{output.stem}_gate.json").write_text(
        json.dumps(gate_metadata, indent=2), encoding="utf-8"
    )
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
    parser.add_argument("--gate-threshold", type=float)
    parser.add_argument("--max-nonsevere-regret", type=float, default=0.001)
    parser.add_argument("--gate-sequence-length", type=int, default=128)
    parser.add_argument("--gate-sequence-lengths", default="1,8,32,128")
    parser.add_argument("--extended-corruptions", action="store_true")
    parser.add_argument("--mixed-segment-length", type=int, default=16)
    parser.add_argument("--use-anomaly-gate", action="store_true")
    parser.add_argument(
        "--anomaly-threshold-multiplier", type=float, default=4.0
    )
    parser.add_argument(
        "--sequence-probability-floor", type=float, default=0.7
    )
    parser.add_argument("--stable-sequence-length", type=int, default=8)
    parser.add_argument(
        "--stable-activation-rate", type=float, default=0.3
    )
    parser.add_argument("--samples-per-class", type=int)
    main(parser.parse_args())
