from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve_config_path
from .explainability import generate_vit_gradcam
from .models import create_vit
from .reproducibility import file_sha256


def _load_rgb_image(path: Path, image_size: int):
    """Return both the display RGB array and the normalized ViT input tensor."""
    from PIL import Image
    from torchvision import transforms

    with Image.open(path) as loaded:
        image = loaded.convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    display = np.asarray(image, dtype=np.float32) / 255.0
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return image, display, transform(image).unsqueeze(0)


def _extract_state_dict(payload: Any) -> Mapping[str, Any]:
    """Accept the plain state dict used by the source code and common wrappers."""
    if not isinstance(payload, Mapping):
        raise ValueError("The checkpoint does not contain a PyTorch state dictionary")

    for key in ("state_dict", "model_state_dict", "network_weights"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            payload = candidate
            break
    if not payload or not all(isinstance(key, str) for key in payload):
        raise ValueError("The checkpoint state dictionary is empty or has invalid keys")

    state = dict(payload)
    for prefix in ("module.", "model."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix) :]: value for key, value in state.items()}
    return state


def _load_checkpoint(model: Any, checkpoint_path: Path) -> None:
    """Load a source-code or wrapped checkpoint without executing pickled code."""
    import torch

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as error:  # pragma: no cover - only relevant to unsupported old PyTorch
        raise RuntimeError(
            "Grad-CAM checkpoint loading requires PyTorch with weights_only support"
        ) from error
    state = _extract_state_dict(payload)
    model.load_state_dict(state, strict=True)


def _public_model_metadata(model_config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove local filesystem locations before serializing model provenance."""
    def sanitize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        if isinstance(value, (str, Path)):
            text = str(value)
            candidate = Path(text).expanduser()
            if candidate.is_absolute() or text.startswith("."):
                return candidate.name
        return value

    metadata = sanitize(dict(model_config))
    metadata.pop("checkpoint", None)
    metadata.pop("cache_dir", None)

    name_or_path = model_config.get("name_or_path")
    if name_or_path is not None:
        candidate = Path(str(name_or_path)).expanduser()
        if candidate.is_absolute() or str(name_or_path).startswith("."):
            metadata["name_or_path_is_local"] = True
    return metadata


def create_gradcam_artifacts(
    config: dict[str, Any],
    checkpoint: str | Path,
    input_image: str | Path,
    output_dir: str | Path,
    target_class: int | None = None,
    target_layer_index: int | None = None,
) -> Path:
    """Create auditable Grad-CAM artifacts for one pre-fused RGB slice.

    The retained source selected encoder layer 8 and class 0. Those values remain
    configurable because the visualized class and layer must be reported with each image.
    """
    import torch
    from PIL import Image
    from pytorch_grad_cam.utils.image import show_cam_on_image

    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=False)
    image_path = Path(input_image).expanduser().resolve(strict=False)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    gradcam_config = config.get("gradcam", {})
    if target_class is None:
        target_class = int(gradcam_config.get("target_class", 0))
    if target_layer_index is None:
        target_layer_index = int(gradcam_config.get("target_layer_index", 8))

    data_config = config.get("data", {})
    image_size = int(data_config.get("image_size", 224))
    if image_size <= 0:
        raise ValueError("data.image_size must be positive")

    original, display_rgb, tensor = _load_rgb_image(image_path, image_size)

    model_config = dict(config["model"])
    model_config.pop("checkpoint", None)
    if model_config.get("name_or_path"):
        configured_name = str(model_config["name_or_path"])
        configured_candidate = resolve_config_path(config, configured_name)
        if configured_name.startswith(".") or configured_candidate.exists():
            model_config["name_or_path"] = str(configured_candidate)
    model = create_vit(model_config)
    _load_checkpoint(model, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    tensor = tensor.to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(pixel_values=tensor).logits, dim=1)[0].cpu().numpy()
    predicted_class = int(np.argmax(probabilities))
    if target_class < 0 or target_class >= len(probabilities):
        raise ValueError(
            f"target_class {target_class} is outside the model's {len(probabilities)} classes"
        )
    maps = generate_vit_gradcam(
        model,
        tensor,
        target_class=target_class,
        target_layer_index=target_layer_index,
    )
    cam = np.clip(np.asarray(maps[0], dtype=np.float32), 0.0, 1.0)
    if cam.ndim != 2 or not np.isfinite(cam).all():
        raise RuntimeError(f"Grad-CAM returned an invalid map with shape {cam.shape}")
    overlay = show_cam_on_image(display_rgb, cam, use_rgb=True)
    heatmap = np.clip(cam * 255.0, 0, 255).astype(np.uint8)

    destination.mkdir(parents=True, exist_ok=True)
    original.save(destination / "input.png")
    np.save(destination / "gradcam.npy", cam)
    Image.fromarray(heatmap).save(destination / "gradcam.png")
    Image.fromarray(overlay).save(destination / "overlay.png")

    metadata = {
        "input_file": image_path.name,
        "input_sha256": file_sha256(image_path),
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "image_size": image_size,
        "target_class": int(target_class),
        "target_layer_index": int(target_layer_index),
        "predicted_class": predicted_class,
        "p_class_0": float(probabilities[0]),
        "p_class_1": float(probabilities[1]),
        "label_names": config.get("label_names", {"0": "MSA-P", "1": "PD"}),
        "model": _public_model_metadata(model_config),
    }
    with (destination / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return destination
