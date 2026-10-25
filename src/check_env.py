"""Check that the installed libraries work together.

Usage:
    python -m src.check_env               # versions, device, CLIP forward pass with random weights
    python -m src.check_env --pretrained  # also download the real CLIP weights (about 350 MB)
"""

import argparse
import importlib

LIBRARIES = [
    "torch",
    "torchvision",
    "open_clip",
    "transformers",
    "datasets",
    "huggingface_hub",
    "numpy",
    "PIL",
    "yaml",
    "tqdm",
    "pandas",
    "matplotlib",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="ViT-B-32")
    parser.add_argument("--pretrained", action="store_true", help="download and use the openai weights")
    args = parser.parse_args()

    for name in LIBRARIES:
        module = importlib.import_module(name)
        print(f"{name:16s} {getattr(module, '__version__', '?')}")

    import torch
    from PIL import Image

    from src.config import resolve_device
    from src.models.encoders import load_clip
    from src.models.fusion import fuse

    device = resolve_device("auto")
    print(f"device           {device}")

    model, preprocess, tokenizer = load_clip(args.backbone, "openai" if args.pretrained else None, device)
    image = preprocess(Image.new("RGB", (256, 256), "red")).unsqueeze(0).to(device)
    text = tokenizer(["make it blue"]).to(device)
    with torch.no_grad():
        query = fuse(model.encode_image(image), model.encode_text(text))
    print(f"CLIP {args.backbone} query embedding {tuple(query.shape)}: OK")


if __name__ == "__main__":
    main()
