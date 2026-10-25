"""Shared command-line handling for train.py and evaluate.py."""

from __future__ import annotations

import argparse

from src.config import Config, load_config, resolve_device, set_seed


def setup_run(description: str) -> tuple[Config, str]:
    """Parse --config/--device, load and validate the config, seed and pick a device."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="path to a YAML file in configs/")
    parser.add_argument("--device", help="override the config: auto | cpu | cuda | mps")
    args = parser.parse_args()

    overrides = {"device": args.device} if args.device else None
    cfg = load_config(args.config, overrides)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"[{cfg.name}] dataset={cfg.dataset}/{cfg.split} model={cfg.model.backbone} device={device}")
    return cfg, device
