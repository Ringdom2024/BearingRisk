# Corruption-aware selective risk assessment for bearing fault diagnosis

Official research code for:

> **Corruption-aware selective risk assessment for cross-condition bearing
> fault diagnosis under severe noise**

Author: **Yifan Wang**  
Affiliation: **Shandong Women's University, Jinan, Shandong, China**

This repository implements a post-hoc reliability score for a
time-frequency network (TFN). The method challenges frequency channels near
the predicted bearing-fault harmonics, learns a correctness score on an
independent calibration condition, and activates that score through a
dual-scale corruption gate.

## Main result

Across CWRU, JNU, and Paderborn with three seeds, the final score:

- reduces AURC by `0.0365` on average under `-6 dB` Gaussian noise;
- improves `8/9` paired runs (`p=0.0195`, one-sided Wilcoxon test);
- exactly preserves the confidence ranking under clean, `6 dB`, and `0 dB`;
- does not consistently improve colored noise, which is a reported limitation.

The committed result tables are in [`results/`](results/), with an index in
[`RESULTS.md`](RESULTS.md).

## Repository structure

```text
Models/                 Upstream TFN model components
utils/                  Minimal upstream model-summary dependency
robust_tfn/             Corruption, intervention, risk, and evaluation code
scripts/                Reproduction helpers
results/                Compact tables reported in the manuscript
tests/                   Dependency-free data and model smoke test
DATASET_SOURCES.md       Dataset sources and cross-condition protocol
NOTICE.md                Upstream attribution and code ownership
```

Raw datasets, trained weights, logs, and per-sample prediction files are not
stored in Git because of size and dataset licensing constraints.

## Installation

Python 3.10 was used for the reported experiments.

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the smoke test:

```bash
python tests/smoke_test.py
python scripts/check_repository.py
```

## Data preparation

Download the datasets from the links in
[`DATASET_SOURCES.md`](DATASET_SOURCES.md). Expected roots are:

```text
data/
  CWRU/
    Normal Baseline Data/
    48k Drive End Bearing Fault Data/
  JNU/
    n600_3_2.csv
    ib600_2.csv
    ...
  Paderborn/
    K001/
    KA01/
    KI01/
```

Paderborn files may be downloaded with:

```powershell
.\scripts\download_paderborn.ps1
```

## Reproduce one dataset

The helper trains the three published seeds and evaluates the final stable
gate. Replace the data path as needed.

```bash
python scripts/reproduce_dataset.py \
  --dataset CWRU \
  --data-root ./data/CWRU \
  --output-root ./research_runs
```

For a quick pipeline check:

```bash
python scripts/reproduce_dataset.py \
  --dataset CWRU \
  --data-root ./data/CWRU \
  --output-root ./research_runs \
  --smoke
```

The full experiment uses 20 epochs for each seed. CWRU seed 999 in the
original run used 12 epochs; this difference is configurable through
`--epochs`.

## Direct commands

Train a noise-augmented TFN:

```bash
python -m robust_tfn.experiment \
  --dataset CWRU \
  --data-root ./data/CWRU \
  --output ./research_runs/cwru_seed_999 \
  --epochs 20 \
  --batch-size 128 \
  --mid-channel 16 \
  --train-noise-min 0 \
  --train-noise-max 12 \
  --seed 999
```

Evaluate the proposed risk score and unseen corruptions:

```bash
python -m robust_tfn.learned_risk \
  --dataset CWRU \
  --data-root ./data/CWRU \
  --checkpoint ./research_runs/cwru_seed_999/best_model.pth \
  --output ./research_runs/cwru_seed_999/extended_risk.csv \
  --mid-channel 16 \
  --batch-size 128 \
  --extended-corruptions \
  --use-anomaly-gate \
  --gate-sequence-length 128 \
  --gate-sequence-lengths 1,8,32,128 \
  --sequence-probability-floor 0.7 \
  --anomaly-threshold-multiplier 4 \
  --stable-sequence-length 8 \
  --stable-activation-rate 0.3
```

## Reproducibility notes

- Training, calibration, and test conditions are disjoint.
- Seeds are `999`, `1001`, and `1003`.
- The test SNR and corruption identity are not supplied to the gate.
- GPU kernels may cause small run-to-run differences.
- The committed CSV files are compact summaries, not substitutes for raw
  prediction artifacts.

## Upstream TFN

The TFN backbone is adapted from Chen et al.:

- Paper: <https://doi.org/10.1016/j.ymssp.2023.110952>
- Code: <https://github.com/ChenQian0618/TFN>

See [`NOTICE.md`](NOTICE.md) and [`LICENSE`](LICENSE).

## Citation

The manuscript is under preparation. A complete citation will be added after
publication. Repository metadata are provided in [`CITATION.cff`](CITATION.cff).
