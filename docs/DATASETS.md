# CIR datasets and benchmarks

Where to get every dataset mentioned in this project. All links were checked
on 2026-10-25.

Most CIR benchmarks are not hosted on Hugging Face by their authors. The usual
route is a GitHub repo with the annotations, plus images you fetch from the
original source. A few community mirrors make this much easier.

## Evaluation benchmarks

| Dataset | Official source | Mirror | Images from | Metrics | Gotchas |
|---|---|---|---|---|---|
| [FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq/) | GitHub repo with captions and splits (CDLA licence) | [BUAADreamer/cir_dataset](https://huggingface.co/datasets/BUAADreamer/cir_dataset) (13.6 GB, FashionIQ + CIRR with images). Val images only: [royokong/fashioniq_val_imgs](https://huggingface.co/datasets/royokong/fashioniq_val_imgs) (207 MB) | Amazon product URLs in a separate metadata repo | R@10, R@50 per category (dress, shirt, toptee) and averaged | Many original URLs are dead. Fixed copies are shared in [issue #18](https://github.com/XiaoxiaoGuo/fashion-iq/issues/18), but commenters say the Yandex copy there is broken or incomplete. Test labels are not public, so report val |
| [CIRR](https://www.zheyuanliu.me/CIRR/) | `cirr_dataset` branch of the [CIRR repo](https://github.com/Cuberick-Orion/CIRR) (MIT for annotations) | [BUAADreamer/cir_dataset](https://huggingface.co/datasets/BUAADreamer/cir_dataset) | NLVR2 images: fill in the [Google Form](https://goo.gl/forms/yS29stWnFWzrDBFH3) and agree to NLVR2's terms | R@1/5/10/50, R_subset@1/2/3 | If NLVR2 does not reply, email the CIRR authors. Don't download images URL by URL, many are broken. Test scores come from the [evaluation server](https://cirr.zheyuanliu.me) |
| [CIRCO](https://github.com/miccunifi/CIRCO) | GitHub repo with `val.json` and `test.json` (CC BY-NC 4.0) | None found | COCO 2017 *unlabeled* images from cocodataset.org | mAP@5/10/25/50 | Val labels public, test hidden: submit to the [CIRCO server](https://circo.micc.unifi.it/) or [Codabench](https://www.codabench.org/competitions/10055/). Folder layout is in the repo README |
| Shoes | Captions in the [fashion-retrieval repo](https://github.com/XiaoxiaoGuo/fashion-retrieval/tree/master/dataset) (per the [ARTEMIS README](https://github.com/naver/artemis)) | None found | Berg et al. attribute dataset at tamaraberg.com | R@1/10/50 | The image page did not load when checked, so images may need a mirror or an email to the authors |
| [Fashion200k](https://github.com/xthan/fashion-200k) | [Google Drive folder](https://drive.google.com/drive/folders/0B4Eo9mft9jwoamlYWFZBSHFzV3c?resourcekey=0-2s7M82p8Bn7riqxWVlgctw) with labels, cropped images and URLs | None found | `women.tar.gz` in the Drive folder | R@1/10/50 | Test queries are a separate file from the [TIRG repo](https://github.com/google/tirg): [test_queries.txt](https://storage.googleapis.com/image_retrieval_css/test_queries.txt) |
| [GeneCIS](https://github.com/facebookresearch/genecis) | Annotation JSONs in the repo, see [DOWNLOAD.md](https://github.com/facebookresearch/genecis/blob/main/DOWNLOAD.md) (CC BY-NC 4.0) | None found | COCO val2017 (5,000 images) and Visual Genome 1.2 (108,249 images) | R@1/2/3 per task | Evaluation only. Optional training triplets: [1.6M CC3M triplets](https://dl.fbaipublicfiles.com/genecis/cc3m_training_triplets_1.6M.pt) |
| MIT-States | [MIT project page](http://web.mit.edu/phillipi/Public/states_and_transformations/index.html), linked from the TIRG README | Community copy on [HyperAI](https://hyper.ai/en/datasets/19033) (770 MB zip, torrent) | Included in the zip | R@1/5/10 | Could not confirm the MIT download link still works; the HyperAI copy is user-contributed |
| CSS | [Google Drive file](https://drive.google.com/file/d/1wPqMw-HKmXUG2qTgYBiTNUnjz83hA2tY/view?usp=sharing) linked from the [TIRG README](https://github.com/google/tirg) | None found | Synthetic renders, included | R@1 | Small toy dataset. Fine for quick experiments, rarely reported in new papers |

## Training sets

| Dataset | Where | Size | Gotchas |
|---|---|---|---|
| [SynthTriplets18M](https://huggingface.co/datasets/navervision/SynthTriplets18M) (CompoDiff) | Hugging Face, official, linked from the [CompoDiff repo](https://github.com/navervision/CompoDiff) | 4.46 TB of tar shards | Source and target images plus CLIP features. No licence stated. Stream a few shards instead of downloading it all |
| WebVid-CoVR (CoVR) | Annotations: [lucas-ventura/WebVid-CoVR](https://huggingface.co/datasets/lucas-ventura/WebVid-CoVR) (MIT). Videos: [lucas-ventura/WebVid](https://huggingface.co/datasets/lucas-ventura/WebVid/tree/main/) | Annotations small, videos 9.65 GB | Video, not images (composed *video* retrieval). The [CoVR README](https://github.com/lucas-ventura/CoVR/) warns many original WebVid URLs now return 403, so use the HF copy |
| CC3M (Conceptual Captions) | [pixparse/cc3m-wds](https://huggingface.co/datasets/pixparse/cc3m-wds) | About 2.91M train + 13.4k val image-caption pairs | Images included in webdataset format. Image-caption data, not CIR triplets; zero-shot methods such as Pic2Word train on it |
| LAION (Re-LAION-5B) | Gated Hugging Face collections, see the [Re-LAION-5B post](https://laion.ai/blog/relaion-5b/) | Billions of rows | Links and metadata only: download images yourself with a tool like img2dataset. Access needs your affiliation and a research-use agreement |
| MagicLens 36.7M triplets | [google-deepmind/magiclens](https://github.com/google-deepmind/magiclens) has inference code and checkpoints only | Not released | Training data never released, and the repo does not support training. A community [PyTorch port](https://huggingface.co/mat0k/magiclens-base-pytorch) adds training |

## Where to start

1. Download [BUAADreamer/cir_dataset](https://huggingface.co/datasets/BUAADreamer/cir_dataset)
   with `bash scripts/download_data.sh`. It gives FashionIQ and CIRR with images
   in one 13.6 GB download, which skips the dead Amazon URLs and the NLVR2 form.
   Cite the original papers and note in the report that you used a community mirror.
2. For a zero-shot setup with no training, add CIRCO: annotations from GitHub,
   COCO 2017 unlabeled images from cocodataset.org, and score on val locally.

Put each dataset under `data/<name>/` (see [`data/README.md`](../data/README.md)).
