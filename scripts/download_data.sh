#!/usr/bin/env bash
# Download FashionIQ + CIRR (with images) from the BUAADreamer Hugging Face mirror.
# Usage: bash scripts/download_data.sh [target_dir]
set -euo pipefail

TARGET="${1:-data/hf_cir_dataset}"
mkdir -p "$TARGET"

huggingface-cli download BUAADreamer/cir_dataset \
  --repo-type dataset \
  --local-dir "$TARGET"

echo "Downloaded to $TARGET"
echo "Next: move or symlink the FashionIQ and CIRR folders into data/fashioniq and data/cirr."
