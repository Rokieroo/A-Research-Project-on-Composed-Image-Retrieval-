# Data

Nothing in this folder is committed to Git (see `.gitignore`). Download the
datasets here so the layout looks like this:

```
data/
├── fashioniq/
├── cirr/
└── circo/
```

## FashionIQ and CIRR (easiest route)

The Hugging Face mirror [`BUAADreamer/cir_dataset`](https://huggingface.co/datasets/BUAADreamer/cir_dataset)
contains FashionIQ and CIRR together with their images (about 13.6 GB).

```bash
bash scripts/download_data.sh
```

or by hand:

```bash
pip install huggingface_hub
hf download BUAADreamer/cir_dataset --repo-type dataset --local-dir data/hf_cir_dataset
```

After downloading, check the folder structure and move or symlink the
FashionIQ and CIRR parts into `data/fashioniq/` and `data/cirr/`.

## Official sources

| Dataset | Annotations | Images |
|---|---|---|
| FashionIQ | https://github.com/XiaoxiaoGuo/fashion-iq | Image URLs in the repo (many are dead links, so prefer the mirror) |
| CIRR | `cirr_dataset` branch of https://github.com/Cuberick-Orion/CIRR | NLVR2 images (request access through the NLVR2 Google Form) |
| CIRCO | https://github.com/miccunifi/CIRCO | COCO 2017 unlabeled split |

Note: CIRR test-set labels are hidden. Test results must be submitted to the
official CIRR evaluation server, so report validation results while developing.

For every other benchmark and training set (Shoes, Fashion200k, GeneCIS,
MIT-States, CSS, SynthTriplets18M, WebVid-CoVR, CC3M, LAION, MagicLens), with
mirrors, metrics and gotchas, see [`docs/DATASETS.md`](../docs/DATASETS.md).
