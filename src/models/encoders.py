"""Image and text encoders (CLIP via open_clip)."""


def load_clip(backbone: str = "ViT-B-32", pretrained: str | None = "openai", device: str = "cpu"):
    """Return (model, preprocess, tokenizer). pretrained=None gives random weights."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(backbone)
    model.eval()
    return model, preprocess, tokenizer
