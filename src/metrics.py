"""Retrieval metrics for Composed Image Retrieval.

All functions take *ranked* lists of gallery ids (best match first), so they
work the same whatever model produced the scores. Use `rank_gallery` to turn a
score matrix into ranked ids.

- Recall@K          FashionIQ, CIRR   one target per query
- Recall_subset@K   CIRR              ranking restricted to the query's subset
- mAP@K             CIRCO             several valid targets per query
"""

from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np


def rank_gallery(
    scores: np.ndarray,
    gallery_ids: Sequence[Hashable],
    exclude_ids: Sequence[Hashable] | None = None,
) -> list[list[Hashable]]:
    """Sort the gallery for each query by descending score.

    Args:
        scores: array of shape (num_queries, num_gallery), higher is better.
        gallery_ids: id of each gallery column.
        exclude_ids: optional id per query to drop from its ranking. CIRR
            removes the reference image this way, since it would otherwise
            often be the top match.
    """
    scores = np.asarray(scores)
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked = []
    for i, row in enumerate(order):
        ids = [gallery_ids[j] for j in row]
        if exclude_ids is not None:
            ids = [g for g in ids if g != exclude_ids[i]]
        ranked.append(ids)
    return ranked


def recall_at_k(
    ranked_ids: Sequence[Sequence[Hashable]],
    target_ids: Sequence[Hashable],
    k: int,
) -> float:
    """Percentage of queries whose target appears in the top k results."""
    if len(ranked_ids) != len(target_ids):
        raise ValueError("ranked_ids and target_ids must have the same length")
    hits = [target in list(ranked[:k]) for ranked, target in zip(ranked_ids, target_ids)]
    return 100.0 * float(np.mean(hits))


def recall_subset_at_k(
    ranked_ids: Sequence[Sequence[Hashable]],
    target_ids: Sequence[Hashable],
    subsets: Sequence[Sequence[Hashable]],
    k: int,
    reference_ids: Sequence[Hashable] | None = None,
) -> float:
    """CIRR Recall_subset@K.

    Each CIRR query comes with a small group of visually similar images (its
    subset). The full-gallery ranking is filtered to those images, keeping
    their order, and Recall@K is computed on the result. The reference image
    is removed from the subset when `reference_ids` is given.
    """
    filtered = []
    for i, (ranked, subset) in enumerate(zip(ranked_ids, subsets)):
        members = set(subset)
        if reference_ids is not None:
            members.discard(reference_ids[i])
        filtered.append([g for g in ranked if g in members])
    return recall_at_k(filtered, target_ids, k)


def average_precision_at_k(
    ranked: Sequence[Hashable],
    relevant: Sequence[Hashable],
    k: int,
) -> float:
    """AP@K for one query, normalised by min(k, number of relevant items)."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant_set:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(k, len(relevant_set))


def map_at_k(
    ranked_ids: Sequence[Sequence[Hashable]],
    relevant_ids: Sequence[Sequence[Hashable]],
    k: int,
) -> float:
    """CIRCO mAP@K, as a percentage."""
    if len(ranked_ids) != len(relevant_ids):
        raise ValueError("ranked_ids and relevant_ids must have the same length")
    aps = [average_precision_at_k(r, rel, k) for r, rel in zip(ranked_ids, relevant_ids)]
    return 100.0 * float(np.mean(aps))
