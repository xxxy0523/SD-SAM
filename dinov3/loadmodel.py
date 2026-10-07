"""Load a complete DINOv3 ViT-B/16 backbone on CPU."""

from pathlib import Path
import torch

from .vision_transformer import vit_base


def extract_state_dict(checkpoint):
    while isinstance(checkpoint, dict):
        nested = next((checkpoint[key] for key in
                       ("state_dict", "model", "backbone", "teacher", "student", "ema", "model_ema")
                       if isinstance(checkpoint.get(key), dict)), None)
        if nested is None:
            break
        checkpoint = nested
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dictionary.")
    return checkpoint


def strip_prefixes(state_dict):
    result = {}
    prefixes = ("module.", "backbone.", "model.", "encoder.")
    for key, value in state_dict.items():
        while any(key.startswith(prefix) for prefix in prefixes):
            key = next(key[len(prefix):] for prefix in prefixes if key.startswith(prefix))
        if key in result:
            raise ValueError(f"Duplicate checkpoint key after removing prefixes: {key}")
        result[key] = value
    return result


def load_dinov3_vitb16(path):
    """Load a local, trusted weight file; never select a device implicitly."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {path}")
    state = strip_prefixes(extract_state_dict(torch.load(path, map_location="cpu", weights_only=True)))
    if any(".mlp.fc1.weight" in key for key in state):
        ffn_layer = "mlp"
    elif any(".mlp.w1.weight" in key for key in state):
        ffn_layer = "swiglu"
    else:
        raise ValueError("Unrecognized DINOv3 feed-forward weights; expected a ViT-B/16 backbone.")
    storage = state.get("storage_tokens")
    n_storage = storage.shape[1] if isinstance(storage, torch.Tensor) else 0
    model = vit_base(
        patch_size=16,
        img_size=224,
        ffn_layer=ffn_layer,
        n_storage_tokens=n_storage,
        qkv_bias="blocks.0.attn.qkv.bias" in state,
        mask_k_bias=any(key.endswith("attn.qkv.bias_mask") for key in state),
        layerscale_init=1e-6 if any(key.endswith("ls1.gamma") for key in state) else None,
        untie_cls_and_patch_norms=any(key.startswith("cls_norm.") for key in state),
        untie_global_and_local_cls_norm=any(key.startswith("local_cls_norm.") for key in state),
    )
    # Classification heads are not part of the feature backbone. Every other
    # parameter/buffer must match: a partial load would freeze random weights.
    state = {key: value for key, value in state.items() if not key.startswith("head.")}
    model.load_state_dict(state, strict=True)
    return model.eval()
