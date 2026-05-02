# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.

# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import *
from dinov3.loadmodel import load_dinov3_vitb16, CKPT
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from dinov3.loadmodel import load_dinov3_vitb16, CKPT

    HAS_DINO = True
except ImportError:
    print("Warning: dinov3 not found. DINO fusion will be disabled.")
    HAS_DINO = False


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
            w=None
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.w = w
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

        if HAS_DINO:
            self.dinomodel = load_dinov3_vitb16(CKPT).to("cuda")
        else:
            self.dinomodel = None
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
                w=w
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

    def forward(self, x: torch.Tensor, return_interm=True) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        # 1. 获取 DINO 特征
        if self.dinomodel is not None:
            dino_data = self.dinomodel(x)
            all_dino_output = self.dinomodel.get_intermediate_layers(x, n=12, reshape=True)
            v_norm = dino_data["x_norm_patchtokens"]
            B, L, C = v_norm.shape
            v_norm = v_norm.view(B, 64, 64, C).permute(0, 3, 1, 2)
        else:
            dino_data = None
            all_dino_output = [None] * 12
            v_norm = None

        x = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        distill_feats = {"student": [], "teacher": []}

        for i, blk in enumerate(self.blocks):

            x = blk(x, dino_data, all_dino_output)
            if i < self.w:
                if return_interm:
                    distill_feats["student"].append(x)
                    distill_feats["teacher"].append(all_dino_output[i])

        x = self.neck(x.permute(0, 3, 1, 2))
        if not return_interm:
            distill_feats = {}
        dinode = []
        dinode.append(all_dino_output[2])
        dinode.append(all_dino_output[5])
        dinode.append(all_dino_output[8])
        dinode.append(all_dino_output[11])
        return x, dinode, distill_feats


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
            w=None
    ) -> None:
        super().__init__()
        self.i = i
        self.w = w
        self.use_dino_fusion = (self.i >= self.w)
        self.use_layer_adapter = (self.i < self.w)

        if self.use_dino_fusion:
            self.SimpleFusion = SimpleFusion(sam_dim=dim, dino_dim=dim, i=self.i)
        else:
            self.SimpleFusion = None

        if self.use_layer_adapter:
            self.layer_adapter = LayerAdapter(dim=dim, bottleneck=64)
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

        x = self.attn(x, dino_data)

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

            B_d, n_heads_d, N_dino, d_head_d = DINOq.shape

            if B != B_d:
                windows_per_image = B // B_d
                DINOq = DINOq.unsqueeze(1).expand(-1, windows_per_image, -1, -1, -1).reshape(B, n_heads_d, N_dino,
                                                                                             d_head_d)
                DINOk = DINOk.unsqueeze(1).expand(-1, windows_per_image, -1, -1, -1).reshape(B, n_heads_d, N_dino,
                                                                                             d_head_d)
                DINOv = DINOv.unsqueeze(1).expand(-1, windows_per_image, -1, -1, -1).reshape(B, n_heads_d, N_dino,
                                                                                             d_head_d)

            if n_heads_d != self.num_heads:
                if n_heads_d < self.num_heads and self.num_heads % n_heads_d == 0:
                    repeat = self.num_heads // n_heads_d
                    DINOq = DINOq.repeat(1, repeat, 1, 1)
                    DINOk = DINOk.repeat(1, repeat, 1, 1)
                    DINOv = DINOv.repeat(1, repeat, 1, 1)
                else:
                    DINOq = DINOq.mean(dim=1, keepdim=True).repeat(1, self.num_heads, 1, 1)
                    DINOk = DINOk.mean(dim=1, keepdim=True).repeat(1, self.num_heads, 1, 1)
                    DINOv = DINOv.mean(dim=1, keepdim=True).repeat(1, self.num_heads, 1, 1)

            if N_dino != N_sam:
                # --- 修复 4D 插值问题 ---
                q_temp = DINOq.permute(0, 1, 3, 2).reshape(B * self.num_heads, d_head_d, N_dino)
                q_temp = F.interpolate(q_temp, size=N_sam, mode='linear')
                DINOq = q_temp.reshape(B, self.num_heads, d_head_d, N_sam).permute(0, 1, 3, 2)

                k_temp = DINOk.permute(0, 1, 3, 2).reshape(B * self.num_heads, d_head_d, N_dino)
                k_temp = F.interpolate(k_temp, size=N_sam, mode='linear')
                DINOk = k_temp.reshape(B, self.num_heads, d_head_d, N_sam).permute(0, 1, 3, 2)

                v_temp = DINOv.permute(0, 1, 3, 2).reshape(B * self.num_heads, d_head_d, N_dino)
                v_temp = F.interpolate(v_temp, size=N_sam, mode='linear')
                DINOv = v_temp.reshape(B, self.num_heads, d_head_d, N_sam).permute(0, 1, 3, 2)

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
