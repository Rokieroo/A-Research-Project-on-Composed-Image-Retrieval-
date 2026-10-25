"""Train a CIR model from a config file.

Usage: python -m src.train --config configs/<name>.yaml [--device cpu]
"""

from src.cli import setup_run


def main():
    cfg, device = setup_run(__doc__)
    if cfg.train is None:
        raise SystemExit(f"{cfg.name} has no 'train' section; it is an evaluation-only config")
    raise NotImplementedError(f"TODO: training loop for {cfg.name}")


if __name__ == "__main__":
    main()
