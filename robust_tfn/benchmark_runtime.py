import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from robust_tfn.data import CrossConditionDataset, get_dataset_spec
from robust_tfn.experiment import collect_predictions
from robust_tfn.learned_risk import loader
from robust_tfn.model import CounterfactualTFN


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def confidence_pass(model, data_loader, device):
    for inputs, _, _ in data_loader:
        F.softmax(model(inputs.to(device)), dim=1)


@torch.no_grad()
def ensemble_pass(models, data_loader, device):
    for inputs, _, _ in data_loader:
        inputs = inputs.to(device)
        torch.stack(
            [F.softmax(model(inputs), dim=1) for model in models]
        ).mean(dim=0)


def measure(function, repeats, device):
    times = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        function()
        synchronize(device)
        times.append(time.perf_counter() - start)
    return min(times), sum(times) / len(times)


def main(args):
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
    dataset = CrossConditionDataset(
        args.data_root,
        [spec.test_condition],
        dataset_name=spec.name,
        samples_per_class=args.samples_per_class,
    )
    data_loader = loader(dataset, args.batch_size)

    confidence_pass(models[0], data_loader, device)
    collect_predictions(models[0], data_loader, device)
    ensemble_pass(models, data_loader, device)

    confidence_min, confidence_mean = measure(
        lambda: confidence_pass(models[0], data_loader, device),
        args.repeats,
        device,
    )
    physical_min, physical_mean = measure(
        lambda: collect_predictions(models[0], data_loader, device),
        args.repeats,
        device,
    )
    ensemble_min, ensemble_mean = measure(
        lambda: ensemble_pass(models, data_loader, device),
        args.repeats,
        device,
    )
    count = len(dataset)
    parameter_count = sum(
        parameter.numel() for parameter in models[0].parameters()
    )
    result = {
        "dataset": spec.name,
        "device": str(device),
        "sample_count": count,
        "single_model_parameters": parameter_count,
        "deep_ensemble_parameters": parameter_count * len(models),
        "ensemble_members": len(models),
        "confidence_seconds_mean": confidence_mean,
        "physical_evidence_seconds_mean": physical_mean,
        "deep_ensemble_seconds_mean": ensemble_mean,
        "confidence_ms_per_sample": 1000 * confidence_min / count,
        "physical_evidence_ms_per_sample": 1000 * physical_min / count,
        "deep_ensemble_ms_per_sample": 1000 * ensemble_min / count,
        "physical_over_confidence": physical_min / confidence_min,
        "ensemble_over_confidence": ensemble_min / confidence_min,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="CWRU", choices=["CWRU", "JNU", "PADERBORN"]
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mid-channel", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    main(parser.parse_args())
