import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from torch import nn

from segment_anything import sam_model_registry
from dinov3.loadmodel import load_dinov3_vitb16, CKPT
from selfprompt.selfprompt import DinoPromptDecoder


class PolypSAM(nn.Module):
    def __init__(self, sam_type, sam_ckpt,
                 dino_ckpt_path="/root/autodl-tmp/SAMdino/selfprompt/poly_best_dino_prompt_decoder.pth", W=3):
        super().__init__()

        # 1. 初始化基础 SAM 模型 [cite: 35, 36]
        self.sam = sam_model_registry[sam_type](checkpoint=sam_ckpt, w=W)

        # --- 全局冻结策略 ---
        # 默认冻结所有参数 [cite: 127]
        for p in self.sam.parameters():
            p.requires_grad = False

        # 冻结 Mask Decoder，保持原始分割精度 [cite: 166]
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = False

        # 冻结 Image Encoder 内部的 DINO 主干 [cite: 166]
        img_encoder = self.sam.image_encoder
        if hasattr(img_encoder, 'dinomodel') and img_encoder.dinomodel is not None:
            for p in img_encoder.dinomodel.parameters():
                p.requires_grad = False
            img_encoder.dinomodel.eval()

        # 2. 在内部导入并初始化 APG 分支 (DinoPromptDecoder) [cite: 56, 157]
        # 注意：此处直接实例化，不再通过参数传入
        self.auto_prompt_gen = DinoPromptDecoder(in_dim=768, embed_dim=256)

        if dino_ckpt_path:
            print(f"📦 Loading pre-trained APG weights from: {dino_ckpt_path}")
            state_dict = torch.load(dino_ckpt_path, map_location='cpu')
            self.auto_prompt_gen.load_state_dict(state_dict)

        for p in self.auto_prompt_gen.parameters():
            p.requires_grad = False
        self.auto_prompt_gen.eval()
        print(" ✅ APG-Branch (DinoPromptDecoder) is LOADED and FROZEN.")

        print("🔥 Unfreezing specific SAM layers for fine-tuning...")
        for name, p in self.named_parameters():
            # 确保不解冻已加载权重的 APG 分支
            if "auto_prompt_gen" in name:
                continue

            if any(k in name for k in
                   ["SimpleFusion", "layer_adapter", ".attn.fc_q", ".attn.fc_k", ".attn.fc_v", "gate", "dino_proj"]):
                p.requires_grad = True
                print(f" -> Trainable: {name}")

    def forward(self, images, return_distill=True):
        """
        前向传播：利用冻结的 APG 分支自动生成提示 [cite: 57, 58, 137]
        """
        # A. 提取多尺度特征
        ret = self.sam.image_encoder(images, return_interm=True)

        if isinstance(ret, tuple) and len(ret) == 3:
            img_emb, dinode, distill_feats = ret
        else:
            img_emb = ret[0] if isinstance(ret, tuple) else ret
            distill_feats = {}

        with torch.no_grad():
            _, auto_pts, auto_lbls = self.auto_prompt_gen(
                dinode,
                input_size=(images.shape[-2], images.shape[-1])
            )

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=(auto_pts, auto_lbls),
            boxes=None,
            masks=None
        )

        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=img_emb,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        logits_256 = low_res_masks[:, 0, :, :]

        if return_distill:
            return logits_256, distill_feats, img_emb
        return logits_256, img_emb
