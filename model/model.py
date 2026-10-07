"""Stage-two segmentation model with a frozen DINO teacher and frozen APG."""

import torch
from torch import nn

from segment_anything import sam_model_registry
from selfprompt.selfprompt import DinoPromptDecoder
from dinov3.loadmodel import extract_state_dict


class PolypSAM(nn.Module):
    def __init__(self, *, sam_type, sam_ckpt, dino_backbone_ckpt, prompt_ckpt,
                 image_size, shallow_layers, adapter_bottleneck, fusion_bottleneck,
                 feature_layers, prompt_config, window_size=0):
        super().__init__()
        if sam_type != "vit_b":
            raise ValueError("This SD-SAM implementation requires SAM ViT-B and DINOv3 ViT-B/16.")
        if not sam_ckpt or not dino_backbone_ckpt or not prompt_ckpt:
            raise ValueError("SAM, DINO and stage-one prompt checkpoints are required.")
        self.sam = sam_model_registry[sam_type](
            checkpoint=sam_ckpt, dino_backbone_ckpt=dino_backbone_ckpt,
            image_size=image_size, shallow_layers=shallow_layers,
            adapter_bottleneck=adapter_bottleneck, fusion_bottleneck=fusion_bottleneck,
            feature_layers=feature_layers, window_size=window_size,
        )
        self.auto_prompt_gen = DinoPromptDecoder(**prompt_config)
        checkpoint = torch.load(prompt_ckpt, map_location="cpu", weights_only=True)
        self.auto_prompt_gen.load_state_dict(extract_state_dict(checkpoint), strict=True)
        self.requires_grad_(False)
        # Dense Q/K/V fusion projections from the supplied implementation.
        for name, parameter in self.sam.image_encoder.named_parameters():
            if any(part in name for part in
                   ("SimpleFusion.", "layer_adapter.", ".attn.fc_q.", ".attn.fc_k.", ".attn.fc_v.")):
                parameter.requires_grad_(True)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Frozen parameters alone do not freeze BatchNorm running statistics.
        self.auto_prompt_gen.eval()
        self.sam.image_encoder.dinomodel.eval()
        self.sam.prompt_encoder.eval()
        self.sam.mask_decoder.eval()
        return self

    def forward(self, images, return_distill=True):
        image_embeddings, dino_features, distill_features = self.sam.image_encoder(
            images, return_interm=return_distill)
        with torch.no_grad():
            _, points, labels = self.auto_prompt_gen(dino_features, input_size=images.shape[-2:])
            sparse, dense = self.sam.prompt_encoder(points=(points, labels), boxes=None, masks=None)
        # Preserve autograd through the frozen decoder to the trainable encoder.
        masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        logits = masks[:, 0]
        if return_distill:
            return logits, distill_features, image_embeddings
        return logits, image_embeddings
