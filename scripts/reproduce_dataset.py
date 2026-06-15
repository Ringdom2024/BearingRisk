import argparse
import subprocess
import sys
from pathlib import Path


SEEDS = (999, 1001, 1003)


def run(command, cwd):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train three TFNs and evaluate the stable corruption gate."
    )
    parser.add_argument(
        "--dataset", required=True, choices=("CWRU", "JNU", "PADERBORN")
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="./research_runs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    epochs = 1 if args.smoke else args.epochs
    samples = 12 if args.smoke else None

    for seed in SEEDS:
        run_dir = output_root / f"{args.dataset.lower()}_seed_{seed}"
        train = [
            sys.executable,
            "-m",
            "robust_tfn.experiment",
            "--dataset",
            args.dataset,
            "--data-root",
            str(data_root),
            "--output",
            str(run_dir),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(args.batch_size),
            "--mid-channel",
            "16",
            "--train-noise-min",
            "0",
            "--train-noise-max",
            "12",
            "--seed",
            str(seed),
        ]
        if samples is not None:
            train.extend(["--samples-per-class", str(samples)])
        run(train, root)

        risk = [
            sys.executable,
            "-m",
            "robust_tfn.learned_risk",
            "--dataset",
            args.dataset,
            "--data-root",
            str(data_root),
            "--checkpoint",
            str(run_dir / "best_model.pth"),
            "--output",
            str(run_dir / "extended_risk.csv"),
            "--mid-channel",
            "16",
            "--batch-size",
            str(args.batch_size),
            "--extended-corruptions",
            "--use-anomaly-gate",
            "--gate-sequence-length",
            "128",
            "--gate-sequence-lengths",
            "1,8,32,128",
            "--sequence-probability-floor",
            "0.7",
            "--anomaly-threshold-multiplier",
            "4",
            "--stable-sequence-length",
            "8",
            "--stable-activation-rate",
            "0.3",
        ]
        if samples is not None:
            risk.extend(["--samples-per-class", str(samples)])
        run(risk, root)


if __name__ == "__main__":
    main()

