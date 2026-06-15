from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robust_tfn.data import (  # noqa: E402
    add_colored_noise,
    add_impulse_noise,
    add_mechanical_interference,
    add_noise_at_snr,
    get_dataset_spec,
)
from robust_tfn.model import CounterfactualTFN  # noqa: E402
from robust_tfn.risk import ranking_metrics  # noqa: E402


def main():
    rng = np.random.default_rng(999)
    signal = np.sin(np.linspace(0, 20 * np.pi, 1024)).astype(np.float32)

    corruptions = [
        add_noise_at_snr(signal, -6, rng),
        add_colored_noise(signal, -6, rng),
        add_impulse_noise(signal, -6, rng),
        add_mechanical_interference(signal, -6, rng, 1730, 48_000),
    ]
    assert all(array.shape == (1024,) for array in corruptions)
    assert all(np.isfinite(array).all() for array in corruptions)

    spec = get_dataset_spec("CWRU")
    model = CounterfactualTFN(
        mid_channel=16,
        num_classes=spec.num_classes,
        sample_rate=spec.sample_rate,
        fault_ratios=spec.fault_ratios,
    )
    inputs = torch.from_numpy(np.stack([signal, corruptions[0]])[:, None, :])
    with torch.no_grad():
        details = model(inputs, return_details=True)
    assert details["logits"].shape == (2, 10)
    assert details["tf_response"].shape[1] == 16

    metrics = ranking_metrics(
        np.array([0.9, 0.8, 0.2, 0.1]),
        np.array([1.0, 1.0, 0.0, 0.0]),
    )
    assert 0.0 <= metrics["aurc"] <= 1.0
    print("Smoke test passed.")


if __name__ == "__main__":
    main()

