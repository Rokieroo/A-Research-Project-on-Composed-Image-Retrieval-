"""Textual inversion for zero-shot CIR (Pic2Word / SEARLE style).

A small mapping network turns the image embedding into a pseudo-word token
S*, which is inserted into a prompt like "a photo of S* that <text>".
"""

import torch.nn as nn


class ImageToToken(nn.Module):
    def __init__(self, embed_dim: int, token_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, image_emb):
        return self.mlp(image_emb)
