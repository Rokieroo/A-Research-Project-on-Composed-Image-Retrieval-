"""CIRCO loader.

Each query has several valid target images drawn from COCO 2017 unlabeled.
Evaluated with mAP@K (K = 5, 10, 25, 50).
"""

from torch.utils.data import Dataset


class CIRCO(Dataset):
    def __init__(self, root: str, split: str, preprocess=None):
        raise NotImplementedError("TODO: read annotations/<split>.json")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
