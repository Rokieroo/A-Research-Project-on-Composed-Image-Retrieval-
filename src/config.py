"""Load, validate and apply experiment configs.

Every run starts from one YAML file in configs/:

    from src.config import load_config, resolve_device, set_seed

    cfg = load_config("configs/baseline_clip.yaml")
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

torch is imported lazily so this module (and its tests) work without it.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DATASETS = {"fashioniq", "cirr", "circo"}
SPLITS = {"train", "val", "test"}
FUSION_MODES = {"sum", "text_only", "image_only", "textual_inversion"}
DEVICES = {"auto", "cpu", "cuda", "mps"}
FASHIONIQ_CATEGORIES = ["dress", "shirt", "toptee"]


class ConfigError(ValueError):
    """Raised when a config file is missing a field or has a bad value."""


@dataclass
class ModelConfig:
    backbone: str = "ViT-B-32"  # open_clip model name, see open_clip.list_pretrained()
    pretrained: str | None = "openai"  # open_clip pretrained tag; None = random weights
    fusion: str = "sum"
    mapping_network_ckpt: str | None = None  # textual inversion only
    prompt_template: str = "a photo of {token} that {text}"  # textual inversion only


@dataclass
class EvalConfig:
    batch_size: int = 64
    recall_ks: list[int] = field(default_factory=lambda: [1, 5, 10, 50])
    subset_ks: list[int] = field(default_factory=lambda: [1, 2, 3])  # CIRR only
    map_ks: list[int] = field(default_factory=lambda: [5, 10, 25, 50])  # CIRCO only


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 0.01
    temperature: float = 0.07


@dataclass
class Config:
    name: str
    dataset: str
    split: str = "val"
    data_root: str = "data/"
    output_dir: str = "outputs/"
    device: str = "auto"
    seed: int = 42
    num_workers: int = 4
    categories: list[str] = field(default_factory=lambda: list(FASHIONIQ_CATEGORIES))  # FashionIQ only
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    train: TrainConfig | None = None

    @property
    def dataset_dir(self) -> Path:
        return Path(self.data_root) / self.dataset

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls, values: dict[str, Any] | None, where: str):
    values = values or {}
    if not isinstance(values, dict):
        raise ConfigError(f"{where} must be a mapping")
    known = set(cls.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"unknown key(s) in {where}: {sorted(unknown)}")
    return cls(**values)


def _check_in(value, allowed, where: str):
    if value not in allowed:
        raise ConfigError(f"{where} must be one of {sorted(allowed)}, got {value!r}")


def _check_positive_ints(values, where: str):
    if not values or not all(isinstance(k, int) and k > 0 for k in values):
        raise ConfigError(f"{where} must be a non-empty list of positive integers")


def validate(cfg: Config) -> Config:
    _check_in(cfg.dataset, DATASETS, "dataset")
    _check_in(cfg.split, SPLITS, "split")
    _check_in(cfg.device, DEVICES, "device")
    _check_in(cfg.model.fusion, FUSION_MODES, "model.fusion")
    if not cfg.model.backbone:
        raise ConfigError("model.backbone is required")
    if cfg.eval.batch_size <= 0:
        raise ConfigError("eval.batch_size must be positive")
    if cfg.num_workers < 0:
        raise ConfigError("num_workers must be >= 0")
    _check_positive_ints(cfg.eval.recall_ks, "eval.recall_ks")
    _check_positive_ints(cfg.eval.subset_ks, "eval.subset_ks")
    _check_positive_ints(cfg.eval.map_ks, "eval.map_ks")
    if cfg.dataset == "cirr" and max(cfg.eval.subset_ks) > 5:
        raise ConfigError("CIRR subsets have 5 candidates, so eval.subset_ks must be <= 5")
    for c in cfg.categories:
        _check_in(c, set(FASHIONIQ_CATEGORIES), "categories")
    if cfg.train is not None and (cfg.train.epochs <= 0 or cfg.train.batch_size <= 0 or cfg.train.lr <= 0):
        raise ConfigError("train.epochs, train.batch_size and train.lr must be positive")
    return cfg


def config_from_dict(raw: dict[str, Any]) -> Config:
    if not isinstance(raw, dict):
        raise ConfigError("config must be a YAML mapping")
    raw = dict(raw)
    for key in ("name", "dataset"):
        if key not in raw:
            raise ConfigError(f"missing required key: {key}")
    model = _build(ModelConfig, raw.pop("model", None), "model")
    eval_cfg = _build(EvalConfig, raw.pop("eval", None), "eval")
    train = _build(TrainConfig, raw.pop("train"), "train") if raw.get("train") is not None else None
    raw.pop("train", None)
    cfg = _build(Config, {**raw, "model": model, "eval": eval_cfg, "train": train}, "config")
    return validate(cfg)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Read a YAML config, apply optional top-level overrides, and validate it."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if overrides:
        raw = {**raw, **overrides}
    return config_from_dict(raw)


def resolve_device(device: str = "auto") -> str:
    """Pick "cuda", then "mps" (Apple Silicon), then "cpu" when device is "auto"."""
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested but MPS is not available")
    return device


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
