# Modified for SD-SAM: explicit Q/K/V output contract and crop-safe metadata.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Callable, List, Optional

import torch
from torch import Tensor, nn

from dinov3.utils import cat_keep_shapes, uncat_with_shapes

from .attention import CausalSelfAttention, SelfAttention
from .ffn_layers import Mlp
from .layer_scale import LayerScale  # , DropPath


class SelfAttentionBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            ffn_ratio: float = 4.0,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            ffn_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            init_values=None,
            drop_path: float = 0.0,
            act_layer: Callable[..., nn.Module] = nn.GELU,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            attn_class: Callable[..., nn.Module] = SelfAttention,
            ffn_layer: Callable[..., nn.Module] = Mlp,
            mask_k_bias: bool = False,
            device=None,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

    def _forward_tensor(self, x: Tensor, rope=None):
        # Compute Q/K/V for every image so the metadata always has batch size B.
        attention_output, qkv = self.attn(self.norm1(x), rope=rope)
        residual = self.ls1(attention_output)
        if self.training and self.sample_drop_ratio > 0:
            batch = x.shape[0]
            subset = max(int(batch * (1 - self.sample_drop_ratio)), 1)
            indices = torch.randperm(batch, device=x.device)[:subset]
            x = torch.index_add(x, 0, indices, residual[indices], alpha=batch / subset)
            indices = torch.randperm(batch, device=x.device)[:subset]
            residual = self.ls2(self.mlp(self.norm2(x[indices])))
            x = torch.index_add(x, 0, indices, residual, alpha=batch / subset)
        else:
            x = x + residual
            x = x + self.ls2(self.mlp(self.norm2(x)))
        return x, qkv, attention_output

    def forward(self, x_or_x_list, rope_or_rope_list=None):
        if isinstance(x_or_x_list, Tensor):
            return self._forward_tensor(x_or_x_list, rope_or_rope_list)
        if isinstance(x_or_x_list, list):
            ropes = rope_or_rope_list or [None] * len(x_or_x_list)
            if len(ropes) != len(x_or_x_list):
                raise ValueError("Each input crop requires its own positional embedding.")
            outputs = [self._forward_tensor(x, rope) for x, rope in zip(x_or_x_list, ropes)]
            return tuple([output[i] for output in outputs] for i in range(3))
        raise TypeError("Expected a tensor or list of tensors.")


class CausalSelfAttentionBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            ffn_ratio: float = 4.0,
            ls_init_value: Optional[float] = None,
            is_causal: bool = True,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.is_causal = is_causal
        self.ls1 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()
        self.attention_norm = norm_layer(dim)
        self.attention = CausalSelfAttention(dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob)

        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()

    def init_weights(
            self,
            init_attn_std: float | None = None,
            init_proj_std: float | None = None,
            init_fc_std: float | None = None,
            factor: float = 1.0,
    ) -> None:
        init_attn_std = init_attn_std or (self.dim ** -0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        init_fc_std = init_fc_std or (2 * self.dim) ** -0.5
        self.attention.init_weights(init_attn_std, init_proj_std)
        self.attention_norm.reset_parameters()
        nn.init.normal_(self.feed_forward.fc1.weight, std=init_fc_std)
        nn.init.normal_(self.feed_forward.fc2.weight, std=init_proj_std)
        self.ffn_norm.reset_parameters()

    def forward(
            self,
            x: torch.Tensor,
    ):
        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn
