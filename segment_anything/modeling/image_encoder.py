# Modified for SD-SAM: frozen DINO fusion, aligned spatial tokens and configurable adapters.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. Licensed under the SAM license; see THIRD_PARTY_NOTICES.md.

from typing import Optional, Tuple, Type
import torch
from torch import nn
from torch.nn import functional as F

from .common import LayerNorm2d, MLPBlock, SimpleFusion, LayerAdapter
from dinov3.loadmodel import load_dinov3_vitb16


class ImageEncoderViT(nn.Module):
    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            *,
            dino_backbone_ckpt,
            shallow_layers,
            adapter_bottleneck,
            fusion_bottleneck,
            feature_layers,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        if not 0 <= shallow_layers <= depth:
            raise ValueError("shallow_layers must be between zero and encoder depth.")
        if (len(feature_layers) != 4 or list(feature_layers) != sorted(set(feature_layers))
                or any(i < 0 or i >= depth for i in feature_layers)):
            raise ValueError("feature_layers must contain four unique ascending zero-based DINO indices.")
        self.shallow_layers = shallow_layers
        self.feature_layers = tuple(feature_layers)
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.dinomodel = load_dinov3_vitb16(dino_backbone_ckpt)
        self.dinomodel.requires_grad_(False)
        if (self.dinomodel.embed_dim != embed_dim or self.dinomodel.num_heads != num_heads
                or len(self.dinomodel.blocks) != depth or self.dinomodel.patch_size != patch_size):
            raise ValueError("SAM and DINO depth, embedding width, head count and patch size must match.")
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                i=i,
                shallow_layers=shallow_layers,
                adapter_bottleneck=adapter_bottleneck,
                fusion_bottleneck=fusion_bottleneck,
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def train(self, mode=True):
        super().train(mode)
        self.dinomodel.eval()
        return self

    def forward(self, x: torch.Tensor, return_interm=True):
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError("Input image dimensions must match the configured image_size.")
        with torch.no_grad():
            dino_data = self.dinomodel(x)
        all_dino_output = dino_data["intermediate_features"]
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        distill_feats = {"student": [], "teacher": []} if return_interm else {}
        for i, block in enumerate(self.blocks):
            x = block(x, dino_data, all_dino_output)
            if return_interm and i < self.shallow_layers:
                distill_feats["student"].append(x)
                distill_feats["teacher"].append(all_dino_output[i])
        embeddings = self.neck(x.permute(0, 3, 1, 2))
        prompt_features = [all_dino_output[i] for i in self.feature_layers]
        return embeddings, prompt_features, distill_feats


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            i: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int]] = None,
            *,
            shallow_layers,
            adapter_bottleneck,
            fusion_bottleneck,
    ) -> None:
        super().__init__()
        self.i = i
        self.w = shallow_layers
        self.use_dino_fusion = (self.i >= self.w)
        self.use_layer_adapter = (self.i < self.w)

        if self.use_dino_fusion:
            self.SimpleFusion = SimpleFusion(sam_dim=dim, dino_dim=dim, bottleneck_dim=fusion_bottleneck)
        else:
            self.SimpleFusion = None

        if self.use_layer_adapter:
            self.layer_adapter = LayerAdapter(dim=dim, bottleneck=adapter_bottleneck)
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.layer_adapter = None

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            i=self.i,
            w=self.w,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor, dino_data, all_dino_output) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        attention_data = dino_data
        if self.window_size > 0 and self.use_dino_fusion:
            window_qkv = []
            for value in dino_data["allqkv"][self.i]:
                batch, heads, tokens, head_dim = value.shape
                grid = value.permute(0, 2, 1, 3).reshape(batch, H, W, heads * head_dim)
                grid, _ = window_partition(grid, self.window_size)
                window_qkv.append(grid.reshape(-1, self.window_size ** 2, heads, head_dim)
                                  .permute(0, 2, 1, 3))
            attention_data = {"allqkv": {self.i: window_qkv}}
        x = self.attn(x, attention_data)

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x

        x = x + self.mlp(self.norm2(x))

        if self.use_layer_adapter:
            x = self.layer_adapter(x)
        elif self.use_dino_fusion:
            dinooutput = all_dino_output[self.i]
            x = self.SimpleFusion(x, dinooutput)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            i: int,
            w: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.i = i
        self.w = w
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_fusion = (self.i >= self.w)

        if self.use_fusion:
            self.fc_q = nn.Linear(self.head_dim * 2, self.head_dim)
            self.fc_k = nn.Linear(self.head_dim * 2, self.head_dim)
            self.fc_v = nn.Linear(self.head_dim * 2, self.head_dim)
        else:
            self.fc_q = None
            self.fc_k = None
            self.fc_v = None

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (input_size is not None), "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim))

    def forward(self, x: torch.Tensor, dino_data) -> torch.Tensor:
        B, H, W, C = x.shape
        N_sam = H * W

        qkv = self.qkv(x).reshape(B, N_sam, 3, self.num_heads, self.head_dim)
        q_sam = qkv[:, :, 0].permute(0, 2, 1, 3)
        k_sam = qkv[:, :, 1].permute(0, 2, 1, 3)
        v_sam = qkv[:, :, 2].permute(0, 2, 1, 3)
        if self.use_fusion and dino_data is not None:
            DINOqkv = dino_data["allqkv"][self.i]
            DINOq, DINOk, DINOv = DINOqkv[0], DINOqkv[1], DINOqkv[2]

            expected = (B, self.num_heads, N_sam, self.head_dim)
            if any(value.shape != expected for value in (DINOq, DINOk, DINOv)):
                raise ValueError(f"DINO patch Q/K/V must align with SAM tokens: expected {expected}.")

            q_cat = torch.cat([q_sam, DINOq], dim=-1)
            k_cat = torch.cat([k_sam, DINOk], dim=-1)
            v_cat = torch.cat([v_sam, DINOv], dim=-1)
            q_cat = self.fc_q(q_cat) + q_sam
            k_cat = self.fc_k(k_cat) + k_sam
            v_cat = self.fc_v(v_cat) + v_sam

            q = q_cat.reshape(B * self.num_heads, N_sam, self.head_dim)
            k = k_cat.reshape(B * self.num_heads, N_sam, self.head_dim)
            v = v_cat.reshape(B * self.num_heads, N_sam, self.head_dim)
        else:
            q = q_sam.contiguous().view(B * self.num_heads, N_sam, self.head_dim)
            k = k_sam.contiguous().view(B * self.num_heads, N_sam, self.head_dim)
            v = v_sam.contiguous().view(B * self.num_heads, N_sam, self.head_dim)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, N_sam, self.head_dim)
        x = x.permute(0, 2, 1, 3).reshape(B, H, W, C)
        x = self.proj(x)
        return x


# ==========================================
# 4. PatchEmbed & Helpers (保持不变)
# ==========================================

class PatchEmbed(nn.Module):
    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16),
            stride: Tuple[int, int] = (16, 16),
            padding: Tuple[int, int] = (0, 0),
            in_chans: int = 3,
            embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
        windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn = (
            attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)
    return attn
