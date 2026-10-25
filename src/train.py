"""Train a CIR model from a config file.

Usage: python -m src.train --config configs/<name>.yaml
"""

import argparse

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    raise NotImplementedError(f"TODO: training loop for {cfg['name']}")


if __name__ == "__main__":
    main()
