import torch
from torch import nn
from typing import Tuple, List
import torch.nn.functional as F


def dice_coeff(logits: torch.Tensor, target: torch.Tensor, eps=1e-6):
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


def iou_score(logits: torch.Tensor, target: torch.Tensor, eps=1e-6):
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2))
    union = (pred + target).clamp(0, 1).sum(dim=(1, 2))
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w = bce_weight

    def soft_dice(self, logits, targets, eps=1e-6):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(1, 2)) + eps
        den = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) + eps
        return 1 - (num / den).mean()

    def forward(self, logits, targets):
        return self.w * self.bce(logits, targets) + (1 - self.w) * self.soft_dice(logits, targets)


# -----------------------------
# 3.5 对齐蒸馏损失（AlignedDistillationLoss）
# -----------------------------
class BottleneckProj(nn.Module):
    """
    低秩通道投影: C -> r -> C
    参数量 ~ 2 * C * r，比直接 CxC 小很多
    """

    def __init__(self, dim=768, bottleneck_dim=128):
        super().__init__()
        self.down = nn.Conv2d(dim, bottleneck_dim, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.up = nn.Conv2d(bottleneck_dim, dim, kernel_size=1, bias=False)

        # 可选：初始化为接近恒等映射（不是必须）
        # nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        # nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


class AlignedDistillationLoss(nn.Module):
    def __init__(self, dim=768, use_proj=True, sample_ratio=0.25, bottleneck_dim=128):
        super().__init__()
        self.use_proj = use_proj
        self.sample_ratio = sample_ratio
        self.dim = dim
        if use_proj:
            self.proj_s = BottleneckProj(dim=dim, bottleneck_dim=bottleneck_dim)
            self.proj_t = BottleneckProj(dim=dim, bottleneck_dim=bottleneck_dim)
        self.cosine = nn.CosineSimilarity(dim=-1)

    def _to_bchw(self, feat: torch.Tensor) -> torch.Tensor:

        assert feat.ndim == 4, f"Expect 4D tensor, got {feat.shape}"
        # 已经是 [B,C,H,W]
        if feat.shape[1] == self.dim:
            return feat
        # [B,H,W,C]
        if feat.shape[-1] == self.dim:
            return feat.permute(0, 3, 1, 2).contiguous()
        # [B,H,C,W]
        if feat.shape[2] == self.dim:
            return feat.permute(0, 2, 1, 3).contiguous()
        if feat.shape[3] == self.dim:
            return feat.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Cannot infer channel dim={self.dim} from shape {feat.shape}")

    def forward(self, student_feats: List[torch.Tensor], teacher_feats: List[torch.Tensor]):
        loss = 0.0
        count = 0
        for s_feat, t_feat in zip(student_feats, teacher_feats):
            # ---- 1. 统一成 [B,C,H,W] ----
            s_feat = self._to_bchw(s_feat)
            t_feat = self._to_bchw(t_feat).detach()

            if self.use_proj:
                s_feat = self.proj_s(s_feat)
                t_feat = self.proj_t(t_feat)

            B, C, H, W = s_feat.shape
            N = H * W
            s = s_feat.reshape(B, C, N).permute(0, 2, 1)  # [B,N,C]
            t = t_feat.reshape(B, C, N).permute(0, 2, 1)  # [B,N,C]

            n_sample = max(1, int(N * self.sample_ratio))
            idx = torch.randperm(N, device=s.device)[:n_sample]
            s = s[:, idx, :]  # [B,n,C]
            t = t[:, idx, :]  # [B,n,C]

            # ---- 5. L2 归一化后做 cosine KD ----
            s = F.normalize(s, dim=-1)
            t = F.normalize(t, dim=-1)
            cos_sim = self.cosine(s, t)  # [B, n]
            loss += (1 - cos_sim.mean())
            count += 1

        return loss / max(1, count)
