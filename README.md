# A Research Project on Composed Image Retrieval

[![CI](https://github.com/Rokieroo/A-Research-Project-on-Composed-Image-Retrieval-/actions/workflows/ci.yml/badge.svg)](https://github.com/Rokieroo/A-Research-Project-on-Composed-Image-Retrieval-/actions/workflows/ci.yml)

SC203 Computational Thinking research project.

**Composed Image Retrieval (CIR)** searches for a target image using a
reference image plus a text that describes how to change it, for example
*(photo of a red dress) + "make it blue and sleeveless"*. A model fuses the
two inputs into one query embedding and ranks a gallery of images by
similarity to it.

## Goal

1. Understand the main families of CIR methods (supervised fusion, zero-shot
   textual inversion, LLM/MLLM reasoning, generative).
2. Reproduce a zero-shot CLIP baseline on CIRR and FashionIQ.
3. Try one improvement and measure it with the standard metrics.

New to Git? Start with the [Git workflow guide](docs/GIT_WORKFLOW.md).

## Repository layout

```
configs/            one YAML per experiment
data/               datasets (not committed, see data/README.md)
src/
  datasets/         FashionIQ, CIRR, CIRCO loaders
  models/           encoders, fusion (phi), textual inversion
  losses.py         InfoNCE, triplet
  metrics.py        Recall@K, Recall_subset@K, mAP@K
  config.py         load and validate configs, pick device, set seed
  train.py
  evaluate.py
  check_env.py      check that the installed libraries work together
scripts/            download_data.sh, run_baseline.sh
notebooks/          exploration and visualisation
experiments/        results table (small files only)
checkpoints/        model weights (not committed, links only)
docs/
  reading_notes/    one .md per paper
  report/           SC203 write-up and figures
tests/              unit tests (start with metrics)
```

## Setup

Requires Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.check_env            # prints library versions, device, and runs a CLIP forward pass
```

All library versions in `requirements.txt` are pinned and tested together. For a
CPU-only or specific CUDA build of torch, see the note at the top of that file.

## Data

FashionIQ and CIRR, with images, are available from the Hugging Face mirror
[`BUAADreamer/cir_dataset`](https://huggingface.co/datasets/BUAADreamer/cir_dataset)
(about 13.6 GB):

```bash
bash scripts/download_data.sh
```

See [`data/README.md`](data/README.md) for the folder layout, and
[`docs/DATASETS.md`](docs/DATASETS.md) for every CIR benchmark and training set
with official sources, mirrors, metrics and gotchas.

## Running

```bash
pytest                         # metric tests and config checks (also run by CI)
bash scripts/run_baseline.sh   # zero-shot CLIP baseline (evaluate.py is still a stub)
python -m src.evaluate --config configs/baseline_clip.yaml --device cpu   # override the device
```

## Metrics

| Dataset | Metric |
|---|---|
| FashionIQ | Recall@10, Recall@50 per category and averaged |
| CIRR | Recall@1/5/10/50 and Recall_subset@1/2/3 |
| CIRCO | mAP@5/10/25/50 |

## Reproducibility

- Data and weights stay out of Git. Commit download scripts and links instead.
- One config per experiment in `configs/`. Every field and its default is defined in
  `src/config.py`, and unknown or invalid fields are rejected before a run starts.
- Log every result in `experiments/README.md` with its config name and commit hash.

## Resources

- Survey: Song et al., *A Comprehensive Survey on Composed Image Retrieval*, [arXiv 2502.18495](https://arxiv.org/abs/2502.18495)
- Reading list (2024 to 2026 papers): https://claude.ai/artifact/TLK2fNRpdCiLJgDhuHuoic
- Dataset download guide: https://claude.ai/artifact/Pbq78dNYFuUTAGBYB3JUi6
