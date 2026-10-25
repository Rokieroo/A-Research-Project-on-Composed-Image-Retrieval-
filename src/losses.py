"""Training losses for supervised CIR."""

import torch
import torch.nn.functional as F


def info_nce(query: torch.Tensor, target: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Batch-wise contrastive loss: query i should match target i.

    Both inputs have shape (batch, dim); they are L2-normalised here.
    """
    query = F.normalize(query, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = query @ target.t() / temperature
    labels = torch.arange(query.size(0), device=query.device)
    return F.cross_entropy(logits, labels)


def triplet(query: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """Cosine-distance triplet loss."""
    pos = 1 - F.cosine_similarity(query, positive)
    neg = 1 - F.cosine_similarity(query, negative)
    return F.relu(pos - neg + margin).mean()
