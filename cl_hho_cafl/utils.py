from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


FEET_TO_METERS = 0.3048
SPEED_OF_LIGHT = 299_792_458.0


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def db_to_linear(value_db: float) -> float:
    return 10.0 ** (value_db / 10.0)


def linear_to_db(value_linear: float, eps: float = 1e-12) -> float:
    return 10.0 * math.log10(max(value_linear, eps))


def dbm_to_watts(value_dbm: float) -> float:
    return 10.0 ** ((value_dbm - 30.0) / 10.0)


def watts_to_dbm(value_watts: float, eps: float = 1e-15) -> float:
    return 10.0 * math.log10(max(value_watts, eps)) + 30.0


def noise_power_watts(noise_psd_dbm_hz: float, bandwidth_hz: float) -> float:
    return dbm_to_watts(noise_psd_dbm_hz + linear_to_db(bandwidth_hz))


def jains_fairness(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    denom = arr.size * np.square(arr).sum()
    if denom <= 0.0:
        return 0.0
    return float(np.square(arr.sum()) / denom)


def bessel_j0(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64)
    return float(torch.special.bessel_j0(tensor).item())


def erfc(value: float) -> float:
    return float(math.erfc(value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def to_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
