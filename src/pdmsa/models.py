from __future__ import annotations

from pathlib import Path
from typing import Any


def create_vit(model_config: dict[str, Any]):
    """Create the fold-shared Hugging Face ViT classifier."""
    from transformers import ViTForImageClassification

    name_or_path = str(model_config["name_or_path"])
    kwargs = {
        "num_labels": int(model_config.get("num_labels", 2)),
        "ignore_mismatched_sizes": bool(model_config.get("ignore_mismatched_sizes", True)),
        "local_files_only": bool(model_config.get("local_files_only", False)),
    }
    if model_config.get("revision"):
        kwargs["revision"] = str(model_config["revision"])
    model = ViTForImageClassification.from_pretrained(name_or_path, **kwargs)

    checkpoint = model_config.get("checkpoint")
    if checkpoint:
        import torch

        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return model
