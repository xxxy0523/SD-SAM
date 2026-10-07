# Modified for SD-SAM: explicit configuration and strict pretrained-weight validation.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. See licenses/SAM_LICENSE.

"""Construct the SD-SAM ViT-B backbone and load complete SAM weights."""

from functools import partial
import torch
from torch.nn import functional as F

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from dinov3.loadmodel import extract_state_dict, strip_prefixes


def load_sam_checkpoint_with_interpolation(sam_model, checkpoint_path):
    state = extract_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    # Official SAM keys start with image_encoder.*, prompt_encoder.*, mask_decoder.*.
    state = {key.removeprefix("module.").removeprefix("sam."): value for key, value in state.items()}
    current = sam_model.state_dict()
    adapted = {}
    unexpected = []
    for key, value in state.items():
        if key not in current:
            unexpected.append(key)
            continue
        target = current[key]
        if value.shape == target.shape:
            adapted[key] = value
        elif key == "image_encoder.pos_embed" and value.shape[-1] == target.shape[-1]:
            adapted[key] = F.interpolate(value.permute(0, 3, 1, 2), target.shape[1:3],
                                         mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
        elif "rel_pos" in key and value.ndim == 2 and value.shape[1] == target.shape[1]:
            adapted[key] = F.interpolate(value.t().unsqueeze(0), target.shape[0],
                                         mode="linear", align_corners=False).squeeze(0).t()
        else:
            raise RuntimeError(f"SAM checkpoint shape mismatch for {key}: {value.shape} vs {target.shape}")

    def is_added(key):
        return (key.startswith("image_encoder.dinomodel.") or
                any(token in key for token in (".SimpleFusion.", ".layer_adapter.",
                                               ".attn.fc_q.", ".attn.fc_k.", ".attn.fc_v.")) or
                key.endswith(".alpha"))

    missing = [key for key in current if key not in adapted and not is_added(key)]
    if missing or unexpected:
        raise RuntimeError(f"Incomplete/incompatible SAM checkpoint. Missing: {missing[:8]}; "
                           f"unexpected: {unexpected[:8]}")
    sam_model.load_state_dict(adapted, strict=False)


def build_sam_vit_b(*, checkpoint, dino_backbone_ckpt, image_size, shallow_layers,
                    adapter_bottleneck, fusion_bottleneck, feature_layers, window_size):
    if image_size % 16:
        raise ValueError("image_size must be divisible by 16.")
    prompt_dim = 256
    grid = image_size // 16
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=12, embed_dim=768, img_size=image_size, mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_heads=12,
            patch_size=16, qkv_bias=True, use_rel_pos=True,
            global_attn_indexes=(2, 5, 8, 11), window_size=window_size,
            out_chans=prompt_dim, dino_backbone_ckpt=dino_backbone_ckpt,
            shallow_layers=shallow_layers, adapter_bottleneck=adapter_bottleneck,
            fusion_bottleneck=fusion_bottleneck, feature_layers=feature_layers,
        ),
        prompt_encoder=PromptEncoder(embed_dim=prompt_dim, image_embedding_size=(grid, grid),
                                     input_image_size=(image_size, image_size), mask_in_chans=16),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=prompt_dim, mlp_dim=2048, num_heads=8),
            transformer_dim=prompt_dim, iou_head_depth=3, iou_head_hidden_dim=256,
        ),
    )
    if checkpoint is not None:
        load_sam_checkpoint_with_interpolation(sam, checkpoint)
    return sam.eval()


def build_sam_vit_h(*args, **kwargs):
    raise ValueError("SD-SAM's layerwise DINO fusion is implemented for SAM ViT-B only.")


def build_sam_vit_l(*args, **kwargs):
    raise ValueError("SD-SAM's layerwise DINO fusion is implemented for SAM ViT-B only.")


build_sam = build_sam_vit_b
sam_model_registry = {"default": build_sam_vit_b, "vit_b": build_sam_vit_b,
                      "vit_h": build_sam_vit_h, "vit_l": build_sam_vit_l}
