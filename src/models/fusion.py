"""Fusion function phi(reference image, modification text) -> query embedding."""

import torch
import torch.nn.functional as F


def fuse(image_emb: torch.Tensor, text_emb: torch.Tensor, mode: str = "sum") -> torch.Tensor:
    """Training-free fusion used by the zero-shot baseline."""
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    if mode == "sum":
        query = image_emb + text_emb
    elif mode == "text_only":
        query = text_emb
    elif mode == "image_only":
        query = image_emb
    else:
        raise ValueError(f"unknown fusion mode: {mode}")
    return F.normalize(query, dim=-1)
