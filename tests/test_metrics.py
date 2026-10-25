import numpy as np
import pytest

from src.metrics import (
    average_precision_at_k,
    map_at_k,
    rank_gallery,
    recall_at_k,
    recall_subset_at_k,
)


def test_rank_gallery_orders_and_excludes_reference():
    scores = np.array([[0.1, 0.9, 0.5], [0.8, 0.2, 0.3]])
    ids = ["a", "b", "c"]
    assert rank_gallery(scores, ids) == [["b", "c", "a"], ["a", "c", "b"]]
    assert rank_gallery(scores, ids, exclude_ids=["b", "c"]) == [["c", "a"], ["a", "b"]]


def test_recall_at_k():
    ranked = [["a", "b", "c"], ["c", "a", "b"]]
    targets = ["a", "b"]
    assert recall_at_k(ranked, targets, 1) == 50.0
    assert recall_at_k(ranked, targets, 2) == 50.0
    assert recall_at_k(ranked, targets, 3) == 100.0


def test_recall_subset_at_k():
    ranked = [["x", "s3", "y", "s1", "s2"]]
    subsets = [["ref", "s1", "s2", "s3"]]
    # Within the subset the order is s3, s1, s2, so target s1 is at rank 2.
    assert recall_subset_at_k(ranked, ["s1"], subsets, 1, reference_ids=["ref"]) == 0.0
    assert recall_subset_at_k(ranked, ["s1"], subsets, 2, reference_ids=["ref"]) == 100.0


def test_average_precision_at_k():
    # Relevant at ranks 1 and 3: (1/1 + 2/3) / min(5, 2)
    assert average_precision_at_k(["a", "x", "b", "y"], ["a", "b"], 5) == pytest.approx((1 + 2 / 3) / 2)
    # Only one hit fits inside k=2, normalised by min(2, 3) = 2
    assert average_precision_at_k(["a", "x", "b"], ["a", "b", "c"], 2) == pytest.approx(0.5)
    assert average_precision_at_k(["x"], [], 5) == 0.0


def test_map_at_k():
    ranked = [["a", "b"], ["x", "a"]]
    relevant = [["a"], ["a"]]
    assert map_at_k(ranked, relevant, 2) == pytest.approx(100 * (1.0 + 0.5) / 2)
