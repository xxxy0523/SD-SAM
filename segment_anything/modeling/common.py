# Modified for SD-SAM: explicit fusion shapes and residual LayerAdapter correction.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        y = self.weight[:, None, None] * x
        # y = torch.mul(self.weight[:, None, None], x)
        x = y + self.bias[:, None, None]
        return x

class SimpleFusion(nn.Module):
    def __init__(self, sam_dim, dino_dim, bottleneck_dim):
        super().__init__()
        self.dino_dim = dino_dim
        self.reducer = nn.Sequential(
            nn.Linear(sam_dim + dino_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, sam_dim),
        )

    def forward(self, x_sam, x_dino):
        if x_dino.ndim != 4 or x_dino.shape[1] != self.dino_dim:
            raise ValueError("DINO fusion expects a BCHW feature map.")
        x_dino = x_dino.permute(0, 2, 3, 1)
        if x_sam.shape[:3] != x_dino.shape[:3]:
            raise ValueError("SAM and DINO feature grids must match.")
        return x_sam + self.reducer(torch.cat((x_sam, x_dino), dim=-1))


class LayerAdapter(nn.Module):
    def __init__(self, dim=768, bottleneck=64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim)
        )
        # 零初始化：保证初始状态下是一个恒等映射，不破坏预训练特征
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x):
        # Zero-initialized residual branch must preserve the pretrained stream.
        return x + self.adapter(x)


class BottleneckAdapter(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, in_dim)
        )

    def forward(self, x):
        return self.net(x)

class MultiAdapterModule(nn.Module):
    def __init__(self, in_dim=768, num_adapters=3, reduction_factor=4):
        """
        Args:
            in_dim: 输入特征维度 (768)
            num_adapters: 适配器数量 (n)
            reduction_factor: 降维倍数 (决定中间层维度)
        """
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        
        
        self.adapters = nn.ModuleList([
            BottleneckAdapter(in_dim, 64) 
            for _ in range(num_adapters)
        ])

    def forward(self, x):
        x_norm = self.norm(x)
        outputs = []
        for adapter in self.adapters:
            out = adapter(x_norm)
            outputs.append(out)
        return outputs


def get_spectrum_img(x):
    fft_x = torch.fft.fft2(x, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft_x, dim=(-2, -1))
    magnitude = fft_shifted.abs()
    magnitude = torch.log(magnitude + 1.0)
    B, C, H, W = magnitude.shape
    mag_flat = magnitude.view(B, C, -1)
    val_min = mag_flat.min(dim=2, keepdim=True)[0].view(B, C, 1, 1)
    val_max = mag_flat.max(dim=2, keepdim=True)[0].view(B, C, 1, 1)
    spectrum = (magnitude - val_min) / (val_max - val_min + 1e-8)
    return spectrum
