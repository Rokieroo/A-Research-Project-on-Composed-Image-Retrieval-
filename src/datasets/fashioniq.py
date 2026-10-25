"""FashionIQ loader.

Each sample: (reference image, two relative captions, target image), split by
category: dress, shirt, toptee. Evaluated with Recall@10 and Recall@50.
"""

from torch.utils.data import Dataset


class FashionIQ(Dataset):
    def __init__(self, root: str, split: str, category: str, preprocess=None):
        raise NotImplementedError("TODO: read captions/cap.<category>.<split>.json and image_splits/")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
