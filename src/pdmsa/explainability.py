from __future__ import annotations

import math
from typing import Any


def vit_reshape_transform(tokens):
    """Convert patch tokens `(B, 1+N, C)` to Grad-CAM feature maps `(B, C, H, W)`."""
    if tokens.ndim != 3:
        raise ValueError(f"Expected a 3D token tensor, found shape {tuple(tokens.shape)}")
    patch_tokens = tokens[:, 1:, :]
    patch_count = int(patch_tokens.shape[1])
    side = int(math.isqrt(patch_count))
    if side * side != patch_count:
        raise ValueError(f"Patch-token count {patch_count} is not a square grid")
    return patch_tokens.reshape(tokens.size(0), side, side, tokens.size(2)).permute(0, 3, 1, 2)


def resolve_vit_target_layer(model: Any, target_layer_index: int):
    """Return a Hugging Face ViT encoder layer norm after validating its index."""
    if not hasattr(model, "vit") or not hasattr(model.vit, "encoder"):
        raise TypeError("Expected a Hugging Face ViTForImageClassification model")

    layers = model.vit.encoder.layer
    layer_count = len(layers)
    resolved_index = (
        target_layer_index if target_layer_index >= 0 else layer_count + target_layer_index
    )
    if resolved_index < 0 or resolved_index >= layer_count:
        raise ValueError(
            f"target_layer_index {target_layer_index} is outside the {layer_count}-layer encoder"
        )
    target_layer = layers[resolved_index]
    if not hasattr(target_layer, "layernorm_before"):
        raise TypeError("The selected ViT encoder layer has no layernorm_before module")
    return target_layer.layernorm_before


def generate_vit_gradcam(
    model,
    input_tensor,
    target_class: int,
    target_layer_index: int = -1,
):
    """Generate class-specific Grad-CAM maps from a Hugging Face ViT classifier.

    ``pytorch-grad-cam`` expects a tensor-returning model, whereas
    ``ViTForImageClassification`` returns an object with a ``logits`` member. The
    lightweight wrapper bridges that interface without changing the trained model.
    """
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    class LogitsWrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, pixels):
            return self.wrapped(pixel_values=pixels).logits

    if int(target_class) < 0:
        raise ValueError("target_class must be non-negative")
    target_layer = resolve_vit_target_layer(model, int(target_layer_index))
    wrapped_model = LogitsWrapper(model)
    model.zero_grad(set_to_none=True)
    with GradCAM(
        model=wrapped_model,
        target_layers=[target_layer],
        reshape_transform=vit_reshape_transform,
    ) as cam:
        maps = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(int(target_class))] * input_tensor.size(0),
        )
    return maps
