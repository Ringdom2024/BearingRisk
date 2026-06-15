# Dataset Sources

## CWRU

- Source: Case Western Reserve University Bearing Data Center.
- Local protocol: loads 0 and 1 for training, load 2 for calibration, load 3
  for testing.
- Integrity: `Datasets_dir/CWRU/SHA256SUMS_ALL_LOADS.csv`.

## JNU

- Source mirror:
  `https://github.com/ClarkGableWang/JNU-Bearing-Dataset`.
- Sampling frequency: 50 kHz.
- Local protocol: 600 rpm training, 800 rpm calibration, 1000 rpm testing.
- Classes: healthy, inner race, roller, outer race.
- Integrity: `Datasets_dir/JNU/SHA256SUMS.csv`.

## Paderborn

- Official source:
  `https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/`
- Reproducible mirror:
  `https://zenodo.org/records/15845309`
- Mirror license: CC BY 4.0.
- Selected bearings: K001 healthy, KA01 outer race, KI01 inner race.
- Sampling frequency: 64 kHz vibration channel `vibration_1`.
- Local protocol:
  - Training: `N09_M07_F10`, `N15_M01_F10`
  - Calibration: `N15_M07_F04`
  - Testing: `N15_M07_F10`
- Each class and condition uses 400 windows sampled evenly from 20 independent
  measurement files.
- Integrity: `Datasets_dir/Paderborn/ARCHIVE_CHECKSUMS.csv`.
