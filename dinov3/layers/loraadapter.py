import torch
import torch.nn as nn
import math


class LoRAAdapter(nn.Module):
    def __init__(self, in_dim=768, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.scale = alpha / rank
        # 可选: 给 LoRA 分支加一点 dropout 防止过拟合
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # 1. 降维 [..., 768] -> [..., 4]
        self.lora_a = nn.Linear(in_dim, rank, bias=False)
        # 2. 升维 [..., 4] -> [..., 768]
        self.lora_b = nn.Linear(rank, in_dim, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
            # A 使用 Kaiming 初始化
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            # B 必须初始化为 0，确保初始状态下 output = input
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        residual = x
        out = self.dropout(x)
        out = self.lora_a(out)
        out = self.lora_b(out)
        out = out * self.scale
        return residual + out