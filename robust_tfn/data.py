from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


SIGNAL_SIZE = 1024


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    sample_rate: float
    fault_ratios: dict
    train_conditions: tuple
    calibration_condition: int
    test_condition: int
    samples_per_class: int


DATASET_SPECS = {
    "CWRU": DatasetSpec(
        name="CWRU",
        num_classes=10,
        sample_rate=48_000.0,
        fault_ratios={
            1: 5.4152,
            2: 2.3570,
            3: 3.5848,
            4: 5.4152,
            5: 2.3570,
            6: 3.5848,
            7: 5.4152,
            8: 2.3570,
            9: 3.5848,
        },
        train_conditions=(0, 1),
        calibration_condition=2,
        test_condition=3,
        samples_per_class=450,
    ),
    "JNU": DatasetSpec(
        name="JNU",
        num_classes=4,
        sample_rate=50_000.0,
        # Labels: normal, inner race (NU205), roller, outer race (N205).
        fault_ratios={1: 6.5, 2: 2.6591, 3: 4.0909},
        train_conditions=(0,),
        calibration_condition=1,
        test_condition=2,
        samples_per_class=400,
    ),
    "PADERBORN": DatasetSpec(
        name="PADERBORN",
        num_classes=3,
        sample_rate=64_000.0,
        # Labels: healthy, outer race, inner race for the 6203 bearing.
        fault_ratios={1: 3.0530, 2: 4.9469},
        train_conditions=(0, 1),
        calibration_condition=2,
        test_condition=3,
        samples_per_class=400,
    ),
}

CWRU_RPM_BY_CONDITION = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
CWRU_NORMAL_FILES = ["97.mat", "98.mat", "99.mat", "100.mat"]
CWRU_FAULT_FILES = [
    [
        "109.mat",
        "122.mat",
        "135.mat",
        "174.mat",
        "189.mat",
        "201.mat",
        "213.mat",
        "226.mat",
        "238.mat",
    ],
    [
        "110.mat",
        "123.mat",
        "136.mat",
        "175.mat",
        "190.mat",
        "202.mat",
        "214.mat",
        "227.mat",
        "239.mat",
    ],
    [
        "111.mat",
        "124.mat",
        "137.mat",
        "176.mat",
        "191.mat",
        "203.mat",
        "215.mat",
        "228.mat",
        "240.mat",
    ],
    [
        "112.mat",
        "125.mat",
        "138.mat",
        "177.mat",
        "192.mat",
        "204.mat",
        "217.mat",
        "229.mat",
        "241.mat",
    ],
]

JNU_RPM_BY_CONDITION = {0: 600, 1: 800, 2: 1000}
JNU_SPEED_NAMES = {0: "600", 1: "800", 2: "1000"}
JNU_FILE_PATTERNS = {
    0: "n{speed}_3_2.csv",
    1: "ib{speed}_2.csv",
    2: "tb{speed}_2.csv",
    3: "ob{speed}_2.csv",
}

PADERBORN_CONDITIONS = {
    0: "N09_M07_F10",
    1: "N15_M01_F10",
    2: "N15_M07_F04",
    3: "N15_M07_F10",
}
PADERBORN_RPM_BY_CONDITION = {0: 900, 1: 1500, 2: 1500, 3: 1500}
PADERBORN_BEARINGS = {0: "K001", 1: "KA01", 2: "KI01"}


def get_dataset_spec(name):
    key = name.upper()
    if key not in DATASET_SPECS:
        raise ValueError(
            f"Unknown dataset {name!r}; choose from {sorted(DATASET_SPECS)}"
        )
    return DATASET_SPECS[key]


def _windows(signal, limit):
    count = min(len(signal) // SIGNAL_SIZE, limit)
    if count == 0:
        raise ValueError("Signal is shorter than one analysis window")
    return np.stack(
        [
            signal[index * SIGNAL_SIZE : (index + 1) * SIGNAL_SIZE]
            for index in range(count)
        ]
    )


def _load_drive_end_signal(path):
    values = loadmat(path)
    key = next((name for name in values if "_DE_time" in name), None)
    if key is None:
        raise ValueError(f"No drive-end signal found in {path}")
    return values[key].squeeze().astype(np.float32)


def load_cwru_condition(root, condition, samples_per_class):
    root = Path(root)
    normal_root = root / "Normal Baseline Data"
    fault_root = root / "48k Drive End Bearing Fault Data"
    arrays = []
    labels = []

    normal = _windows(
        _load_drive_end_signal(normal_root / CWRU_NORMAL_FILES[condition]),
        samples_per_class,
    )
    arrays.append(normal)
    labels.extend([0] * len(normal))

    for label, filename in enumerate(CWRU_FAULT_FILES[condition], start=1):
        fault = _windows(
            _load_drive_end_signal(fault_root / filename), samples_per_class
        )
        arrays.append(fault)
        labels.extend([label] * len(fault))

    data = np.concatenate(arrays).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    rpms = np.full(
        len(labels), CWRU_RPM_BY_CONDITION[condition], dtype=np.float32
    )
    conditions = np.full(len(labels), condition, dtype=np.int64)
    return data, labels, rpms, conditions


def _load_csv_signal(path):
    return np.loadtxt(path, delimiter=",", dtype=np.float32).reshape(-1)


def load_jnu_condition(root, condition, samples_per_class):
    root = Path(root)
    speed = JNU_SPEED_NAMES[condition]
    arrays = []
    labels = []
    for label, pattern in JNU_FILE_PATTERNS.items():
        signal = _load_csv_signal(root / pattern.format(speed=speed))
        windows = _windows(signal, samples_per_class)
        arrays.append(windows)
        labels.extend([label] * len(windows))

    data = np.concatenate(arrays).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    rpms = np.full(
        len(labels), JNU_RPM_BY_CONDITION[condition], dtype=np.float32
    )
    conditions = np.full(len(labels), condition, dtype=np.int64)
    return data, labels, rpms, conditions


def _load_paderborn_signal(path):
    root = loadmat(path, simplify_cells=True)[path.stem]
    vibration = next(
        channel for channel in root["Y"] if channel["Name"] == "vibration_1"
    )
    return np.asarray(vibration["Data"], dtype=np.float32).reshape(-1)


def _balanced_file_windows(paths, limit):
    per_file = max(int(np.ceil(limit / len(paths))), 1)
    arrays = []
    for path in paths:
        signal = _load_paderborn_signal(path)
        available = len(signal) // SIGNAL_SIZE
        count = min(per_file, available)
        indices = np.linspace(0, available - 1, count, dtype=int)
        arrays.extend(
            signal[index * SIGNAL_SIZE : (index + 1) * SIGNAL_SIZE]
            for index in indices
        )
    return np.stack(arrays[:limit])


def load_paderborn_condition(root, condition, samples_per_class):
    root = Path(root)
    condition_name = PADERBORN_CONDITIONS[condition]
    arrays = []
    labels = []
    for label, bearing in PADERBORN_BEARINGS.items():
        paths = sorted(
            (root / bearing).glob(f"{condition_name}_{bearing}_*.mat"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )
        if len(paths) != 20:
            raise ValueError(
                f"Expected 20 files for {bearing}/{condition_name}, got {len(paths)}"
            )
        windows = _balanced_file_windows(paths, samples_per_class)
        arrays.append(windows)
        labels.extend([label] * len(windows))

    data = np.concatenate(arrays).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    rpms = np.full(
        len(labels), PADERBORN_RPM_BY_CONDITION[condition], dtype=np.float32
    )
    conditions = np.full(len(labels), condition, dtype=np.int64)
    return data, labels, rpms, conditions


@lru_cache(maxsize=32)
def _load_condition_cached(dataset_name, root, condition, samples_per_class):
    spec = get_dataset_spec(dataset_name)
    if spec.name == "CWRU":
        return load_cwru_condition(root, condition, samples_per_class)
    if spec.name == "JNU":
        return load_jnu_condition(root, condition, samples_per_class)
    if spec.name == "PADERBORN":
        return load_paderborn_condition(root, condition, samples_per_class)
    raise AssertionError(f"No loader implemented for {spec.name}")


def load_condition(dataset_name, root, condition, samples_per_class=None):
    spec = get_dataset_spec(dataset_name)
    limit = samples_per_class or spec.samples_per_class
    resolved_root = str(Path(root).resolve())
    return _load_condition_cached(spec.name, resolved_root, condition, limit)


def add_noise_at_snr(signal, snr_db, rng):
    if snr_db is None:
        return signal
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise.astype(np.float32)


def _scale_interference(signal, interference, snr_db):
    signal_power = max(float(np.mean(signal**2)), 1e-12)
    interference_power = max(float(np.mean(interference**2)), 1e-12)
    target_power = signal_power / (10 ** (snr_db / 10))
    return interference * np.sqrt(target_power / interference_power)


def add_colored_noise(signal, snr_db, rng, coefficient=0.9):
    white = rng.normal(size=signal.shape).astype(np.float32)
    colored = np.empty_like(white)
    colored[0] = white[0]
    for index in range(1, len(white)):
        colored[index] = coefficient * colored[index - 1] + white[index]
    colored -= colored.mean()
    return signal + _scale_interference(signal, colored, snr_db).astype(
        np.float32
    )


def add_impulse_noise(signal, snr_db, rng, probability=0.01):
    impulses = np.zeros_like(signal)
    mask = rng.random(len(signal)) < probability
    if not mask.any():
        mask[rng.integers(0, len(signal))] = True
    impulses[mask] = rng.normal(size=mask.sum())
    return signal + _scale_interference(signal, impulses, snr_db).astype(
        np.float32
    )


def add_mechanical_interference(signal, snr_db, rng, rpm, sample_rate):
    time = np.arange(len(signal), dtype=np.float32) / sample_rate
    shaft_frequency = rpm / 60.0
    harmonics = np.zeros_like(signal)
    for order, amplitude in ((1, 1.0), (2, 0.7), (3, 0.45), (5, 0.25)):
        phase = rng.uniform(0, 2 * np.pi)
        harmonics += amplitude * np.sin(
            2 * np.pi * order * shaft_frequency * time + phase
        )
    carrier_frequency = rng.uniform(0.08, 0.22) * sample_rate
    modulation = 1.0 + 0.6 * np.sin(2 * np.pi * shaft_frequency * time)
    harmonics += 0.5 * modulation * np.sin(
        2 * np.pi * carrier_frequency * time + rng.uniform(0, 2 * np.pi)
    )
    return signal + _scale_interference(signal, harmonics, snr_db).astype(
        np.float32
    )


def add_uniform_noise(signal, snr_db, rng):
    noise = rng.uniform(-1.0, 1.0, size=signal.shape).astype(np.float32)
    return signal + _scale_interference(signal, noise, snr_db).astype(
        np.float32
    )


def add_burst_dropout(signal, rng, fraction=0.2):
    corrupted = signal.copy()
    total = max(int(round(len(signal) * fraction)), 1)
    burst_count = 4
    burst_length = max(total // burst_count, 1)
    for _ in range(burst_count):
        start = rng.integers(0, max(len(signal) - burst_length, 1))
        corrupted[start : start + burst_length] = 0.0
    return corrupted


def add_chirp_interference(signal, snr_db, rng, sample_rate):
    time = np.arange(len(signal), dtype=np.float32) / sample_rate
    start_frequency = rng.uniform(0.02, 0.08) * sample_rate
    end_frequency = rng.uniform(0.25, 0.45) * sample_rate
    duration = len(signal) / sample_rate
    slope = (end_frequency - start_frequency) / duration
    phase = 2 * np.pi * (
        start_frequency * time + 0.5 * slope * time**2
    )
    chirp = np.sin(phase + rng.uniform(0, 2 * np.pi))
    return signal + _scale_interference(signal, chirp, snr_db).astype(
        np.float32
    )


class CrossConditionDataset(Dataset):
    def __init__(
        self,
        root,
        conditions,
        dataset_name="CWRU",
        train=False,
        noise_snr=None,
        random_noise_range=None,
        corruption=None,
        corruption_snr=-6.0,
        mixed_snr_values=(None, 6.0, 0.0, -6.0),
        mixed_segment_length=16,
        seed=999,
        samples_per_class=None,
    ):
        loaded = [
            load_condition(dataset_name, root, condition, samples_per_class)
            for condition in conditions
        ]
        self.dataset_name = dataset_name.upper()
        self.data = np.concatenate([item[0] for item in loaded])
        self.labels = np.concatenate([item[1] for item in loaded])
        self.rpms = np.concatenate([item[2] for item in loaded])
        self.conditions = np.concatenate([item[3] for item in loaded])
        self.train = train
        self.noise_snr = noise_snr
        self.random_noise_range = random_noise_range
        self.corruption = corruption
        self.corruption_snr = corruption_snr
        self.mixed_snr_values = tuple(mixed_snr_values)
        self.mixed_segment_length = int(mixed_segment_length)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        signal = self.data[index].copy()
        rng = self.rng if self.train else np.random.default_rng(self.seed + index)
        snr = self.noise_snr
        if self.train and self.random_noise_range is not None:
            low, high = self.random_noise_range
            snr = rng.uniform(low, high)
        if self.corruption == "mixed_gaussian":
            segment = index // self.mixed_segment_length
            sequence = segment // len(self.mixed_snr_values)
            position = segment % len(self.mixed_snr_values)
            order = np.random.default_rng(self.seed + sequence).permutation(
                len(self.mixed_snr_values)
            )
            snr = self.mixed_snr_values[order[position]]
        if self.corruption in (None, "mixed_gaussian"):
            signal = add_noise_at_snr(signal, snr, rng)
        elif self.corruption == "colored":
            signal = add_colored_noise(
                signal, self.corruption_snr, rng
            )
        elif self.corruption == "impulse":
            signal = add_impulse_noise(
                signal, self.corruption_snr, rng
            )
        elif self.corruption == "mechanical":
            spec = get_dataset_spec(self.dataset_name)
            signal = add_mechanical_interference(
                signal,
                self.corruption_snr,
                rng,
                self.rpms[index],
                spec.sample_rate,
            )
        elif self.corruption == "uniform":
            signal = add_uniform_noise(
                signal, self.corruption_snr, rng
            )
        elif self.corruption == "burst_dropout":
            signal = add_burst_dropout(signal, rng)
        elif self.corruption == "chirp":
            spec = get_dataset_spec(self.dataset_name)
            signal = add_chirp_interference(
                signal, self.corruption_snr, rng, spec.sample_rate
            )
        else:
            raise ValueError(f"Unknown corruption: {self.corruption}")
        std = max(float(signal.std()), 1e-8)
        signal = (signal - signal.mean()) / std
        return (
            torch.from_numpy(signal[None, :].astype(np.float32)),
            torch.tensor(self.labels[index], dtype=torch.long),
            torch.tensor(self.rpms[index], dtype=torch.float32),
        )
