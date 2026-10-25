"""Evaluate a CIR model from a config file.

Usage: python -m src.evaluate --config configs/<name>.yaml [--device cpu]

Steps to implement:
1. Load the dataset split and the model from the config.
2. Encode every gallery image once.
3. For each query, fuse reference image + text into a query embedding.
4. Score = query @ gallery.T, then rank with src.metrics.rank_gallery.
5. Report metrics from src.metrics and add a row to experiments/README.md.
"""

from src.cli import setup_run


def main():
    cfg, device = setup_run(__doc__)
    raise NotImplementedError(f"TODO: evaluation for {cfg.name}")


if __name__ == "__main__":
    main()
