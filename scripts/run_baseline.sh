#!/usr/bin/env bash
# Evaluate the zero-shot CLIP baseline.
set -euo pipefail
python -m src.evaluate --config configs/baseline_clip.yaml "$@"
