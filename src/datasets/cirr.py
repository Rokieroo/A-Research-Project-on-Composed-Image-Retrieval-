"""CIRR loader.

Each sample: (reference image, modification text, target image, subset of
visually similar images). Evaluated with Recall@K and Recall_subset@K.
"""

from torch.utils.data import Dataset


class CIRR(Dataset):
    def __init__(self, root: str, split: str, preprocess=None):
        raise NotImplementedError("TODO: read captions/cap.rc2.<split>.json and image_splits/")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
