import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from robust_tfn.data import get_dataset_spec
from robust_tfn.experiment import basic_accuracy, make_loaders, set_seed
from robust_tfn.model import build_baseline_model


def train_epoch(model, loader, optimizer, device, class_weights):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for inputs, labels, _ in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels, weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_count += len(labels)
    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels_all = []
    predictions_all = []
    for inputs, labels, _ in loader:
        predictions = model(inputs.to(device)).argmax(1).cpu().numpy()
        labels_all.append(labels.numpy())
        predictions_all.append(predictions)
    labels = np.concatenate(labels_all)
    predictions = np.concatenate(predictions_all)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def run(args):
    set_seed(args.seed)
    spec = get_dataset_spec(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, calibration_loader, test_loaders = make_loaders(
        spec.name,
        args.data_root,
        args.batch_size,
        (args.train_noise_min, args.train_noise_max),
        args.seed,
        args.samples_per_class,
    )
    model = build_baseline_model(args.model, spec.num_classes).to(device)
    counts = np.bincount(
        train_loader.dataset.labels, minlength=spec.num_classes
    )
    class_weights = torch.tensor(
        len(train_loader.dataset) / (spec.num_classes * counts),
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)

    best_state = None
    best_accuracy = -1.0
    history = []
    start = time.time()
    for epoch in range(args.epochs):
        loss, train_accuracy = train_epoch(
            model, train_loader, optimizer, device, class_weights
        )
        calibration_accuracy = basic_accuracy(model, calibration_loader, device)
        if calibration_accuracy > best_accuracy:
            best_accuracy = calibration_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss,
                "train_accuracy": train_accuracy,
                "calibration_accuracy": calibration_accuracy,
            }
        )
        scheduler.step()
        print(
            f"epoch={epoch + 1:02d} loss={loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"calibration_acc={calibration_accuracy:.4f}"
        )

    model.load_state_dict(best_state)
    rows = []
    for condition, loader in test_loaders.items():
        row = {"condition": condition, "model": args.model, "seed": args.seed}
        row.update(evaluate(model, loader, device))
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
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    for row in rows:
        print(row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=["CWRU", "JNU", "PADERBORN"]
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True, choices=["CNN", "ResNet1D"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-noise-min", type=float, default=0.0)
    parser.add_argument("--train-noise-max", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--samples-per-class", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
