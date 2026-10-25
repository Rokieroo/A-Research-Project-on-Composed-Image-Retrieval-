"""Evaluate a CIR model from a config file.

Usage: python -m src.evaluate --config configs/<name>.yaml

Steps to implement:
1. Load the dataset split and the model from the config.
2. Encode every gallery image once.
3. For each query, fuse reference image + text into a query embedding.
4. Score = query @ gallery.T, then rank with src.metrics.rank_gallery.
5. Report metrics from src.metrics and add a row to experiments/README.md.
"""

import argparse

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    raise NotImplementedError(f"TODO: evaluation for {cfg['name']}")


if __name__ == "__main__":
    main()
