from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_protocol(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        protocol = yaml.safe_load(config_file)
    if protocol["protocol_version"] != "ACDC_MINDEG_V1":
        raise ValueError("Only ACDC_MINDEG_V1 is supported by this generator")
    return protocol


def source_relative_path(raw_root: str | Path, source_path: str | Path) -> str:
    return Path(source_path).resolve().relative_to(Path(raw_root).resolve()).as_posix()


def sample_seed(protocol: dict[str, Any], split: str, relative_path: str) -> int:
    payload = "|".join(
        [str(protocol["protocol_version"]), split, relative_path, str(protocol["global_seed"])]
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def _uniform(generator: np.random.Generator, values: dict[str, float]) -> float:
    return float(generator.uniform(values["min"], values["max"]))


def sample_parameters(protocol: dict[str, Any], split: str, relative_path: str) -> tuple[int, dict[str, float]]:
    stable_seed = sample_seed(protocol, split, relative_path)
    generator = np.random.Generator(np.random.PCG64(stable_seed))
    illumination = protocol["illumination_parameter_range"]
    parameters = {
        "global_exposure": _uniform(generator, protocol["exposure_range"]),
        "local_attenuation_strength": _uniform(generator, illumination["local_attenuation_strength"]),
        "center_x": _uniform(generator, illumination["center_x"]),
        "center_y": _uniform(generator, illumination["center_y"]),
        "spread": _uniform(generator, illumination["spread"]),
        "shot_noise_coefficient": float(protocol["noise_parameter_range"]["shot_noise_coefficient"]),
        "read_noise_floor": float(protocol["noise_parameter_range"]["read_noise_floor"]),
    }
    return stable_seed, parameters


def apply_degradation(normal_rgb: np.ndarray, parameters: dict[str, float], stable_seed: int) -> np.ndarray:
    if normal_rgb.dtype != np.float32:
        raise TypeError("normal_rgb must be float32")
    if normal_rgb.ndim != 3 or normal_rgb.shape[2] != 3:
        raise ValueError("normal_rgb must have shape HxWx3")
    height, width, _ = normal_rgb.shape
    y_coord = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    x_coord = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    radial = ((y_coord - parameters["center_y"]) ** 2 + (x_coord - parameters["center_x"]) ** 2) / (parameters["spread"] ** 2)
    smooth_field = np.exp(-radial, dtype=np.float32)[..., None]
    attenuation = parameters["global_exposure"] * (1.0 - parameters["local_attenuation_strength"] * smooth_field)
    dimmed = normal_rgb * attenuation
    generator = np.random.Generator(np.random.PCG64(stable_seed))
    noise_std = parameters["shot_noise_coefficient"] * np.sqrt(np.maximum(dimmed, 1e-4)) + parameters["read_noise_floor"]
    noise = generator.standard_normal(dimmed.shape, dtype=np.float32) * noise_std
    return np.clip(dimmed + noise, 0.0, 1.0).astype(np.float32, copy=False)


def manifest_parameter_fields(parameters: dict[str, float]) -> dict[str, str]:
    return {
        "exposure_parameters": json.dumps({"global_exposure": parameters["global_exposure"]}, sort_keys=True, separators=(",", ":")),
        "illumination_parameters": json.dumps({"center_x": parameters["center_x"], "center_y": parameters["center_y"], "local_attenuation_strength": parameters["local_attenuation_strength"], "spread": parameters["spread"]}, sort_keys=True, separators=(",", ":")),
        "noise_parameters": json.dumps({"read_noise_floor": parameters["read_noise_floor"], "shot_noise_coefficient": parameters["shot_noise_coefficient"]}, sort_keys=True, separators=(",", ":")),
    }
