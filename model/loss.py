"""Segmentation, prompt regression and feature-distillation objectives."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _as_bhw(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [B,H,W] or [B,1,H,W], got {tuple(tensor.shape)}")
    return tensor


def _matching_masks(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = _as_bhw(logits, "logits")
    target = _as_bhw(target, "target")
    if logits.shape != target.shape:
        raise ValueError(f"Logit/target shapes differ: {tuple(logits.shape)} and {tuple(target.shape)}")
    return logits, target.to(device=logits.device, dtype=logits.dtype)


def metrics_per_image(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    """Return one Dice and IoU value per image, including smaller final batches."""
    logits, target = _matching_masks(logits, target)
    prediction = (logits.sigmoid() > threshold).float()
    target = (target > 0.5).float()
    intersection = (prediction * target).sum(dim=(1, 2))
    total = prediction.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return {"dice": (2 * intersection + eps) / (total + eps), "iou": (intersection + eps) / (total - intersection + eps)}


def dice_coeff(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    return metrics_per_image(logits, target, eps=eps)["dice"].mean().item()


def iou_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    return metrics_per_image(logits, target, eps=eps)["iou"].mean().item()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        if not 0 <= bce_weight <= 1:
            raise ValueError("bce_weight must be between zero and one")
        self.w = bce_weight

    def soft_dice(self, logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        logits, targets = _matching_masks(logits, targets)
        probabilities = logits.sigmoid()
        numerator = 2 * (probabilities * targets).sum(dim=(1, 2)) + eps
        denominator = probabilities.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) + eps
        return 1 - (numerator / denominator).mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = _matching_masks(logits, targets)
        return self.w * F.binary_cross_entropy_with_logits(logits, targets) + (1 - self.w) * self.soft_dice(logits, targets)


def gaussian_heatmap_targets(mask: torch.Tensor, sigma: float, output_size: tuple[int, int] | None = None) -> torch.Tensor:
    """Centroid Gaussian clipped to the foreground; sigma is in output pixels."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    mask = _as_bhw(mask, "mask").float()
    if output_size is not None and mask.shape[-2:] != tuple(output_size):
        mask = F.interpolate(mask[:, None], size=output_size, mode="nearest")[:, 0]
    mask = (mask > 0.5).float()
    _, height, width = mask.shape
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=mask.device, dtype=mask.dtype),
        torch.arange(width, device=mask.device, dtype=mask.dtype),
        indexing="ij",
    )
    foreground_count = mask.sum(dim=(1, 2)).clamp_min(1)
    center_y = (mask * y_grid).sum(dim=(1, 2)) / foreground_count
    center_x = (mask * x_grid).sum(dim=(1, 2)) / foreground_count
    distance_squared = (x_grid - center_x[:, None, None]).square() + (y_grid - center_y[:, None, None]).square()
    return torch.exp(-distance_squared / (2 * sigma**2)) * mask


def heatmap_regression_loss(logits: torch.Tensor, mask: torch.Tensor, sigma: float = 20.0, mse_weight: float = 1.0, bce_weight: float = 1.0) -> torch.Tensor:
    """Gaussian probability MSE plus binary-mask BCE; weights are set by entrypoints."""
    if mse_weight < 0 or bce_weight < 0 or mse_weight + bce_weight <= 0:
        raise ValueError("Loss weights must be nonnegative with at least one positive weight")
    logits = _as_bhw(logits, "logits")
    mask = _as_bhw(mask, "mask").to(device=logits.device, dtype=logits.dtype)
    if mask.shape[0] != logits.shape[0]:
        raise ValueError("Logit and mask batch sizes differ")
    if mask.shape[-2:] != logits.shape[-2:]:
        mask = F.interpolate(mask[:, None], size=logits.shape[-2:], mode="nearest")[:, 0]
    mask = (mask > 0.5).to(dtype=logits.dtype)
    target = gaussian_heatmap_targets(mask, sigma).to(dtype=logits.dtype)
    return mse_weight * F.mse_loss(logits.sigmoid(), target) + bce_weight * F.binary_cross_entropy_with_logits(logits, mask)


class BottleneckProj(nn.Module):
    """Shared low-rank channel projection C -> r -> C."""

    def __init__(self, dim: int = 768, bottleneck_dim: int = 128):
        super().__init__()
        if dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("Projection dimensions must be positive")
        self.down = nn.Conv2d(dim, bottleneck_dim, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.up = nn.Conv2d(bottleneck_dim, dim, kernel_size=1, bias=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(tensor)))


class AlignedDistillationLoss(nn.Module):
    """Mean sampled cosine loss after trainable student/teacher projections."""

    def __init__(self, dim: int = 768, use_proj: bool = True, sample_ratio: float = 0.25, bottleneck_dim: int = 128):
        super().__init__()
        if dim <= 0 or not 0 < sample_ratio <= 1:
            raise ValueError("dim must be positive and sample_ratio must be in (0, 1]")
        self.use_proj = use_proj
        self.sample_ratio = sample_ratio
        self.dim = dim
        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)
        if use_proj:
            self.proj_s = BottleneckProj(dim, bottleneck_dim)
            self.proj_t = BottleneckProj(dim, bottleneck_dim)

    def _to_bchw(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(f"Expected a 4-D feature, got {tuple(feature.shape)}")
        candidates = [axis for axis in (1, 2, 3) if feature.shape[axis] == self.dim]
        if len(candidates) != 1:
            raise ValueError(f"Cannot unambiguously identify {self.dim} channels in {tuple(feature.shape)}")
        channel_axis = candidates[0]
        if channel_axis == 1:
            return feature
        if channel_axis == 3:
            return feature.permute(0, 3, 1, 2).contiguous()
        return feature.permute(0, 2, 1, 3).contiguous()

    def forward(self, student_feats: Sequence[torch.Tensor], teacher_feats: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(student_feats) != len(teacher_feats):
            raise ValueError(f"Student/teacher feature counts differ: {len(student_feats)} versus {len(teacher_feats)}")
        if not student_feats:
            return self._zero.clone()
        losses = []
        for index, (student, teacher) in enumerate(zip(student_feats, teacher_feats)):
            student = self._to_bchw(student)
            # Detach the backbone feature, while allowing the teacher projection to learn.
            teacher = self._to_bchw(teacher).detach()
            if student.shape != teacher.shape:
                raise ValueError(f"Feature pair {index} has mismatched shapes: {tuple(student.shape)} versus {tuple(teacher.shape)}")
            if student.device != teacher.device:
                raise ValueError(f"Feature pair {index} is on different devices")
            if self.use_proj:
                student, teacher = self.proj_s(student), self.proj_t(teacher)
            batch, channels, height, width = student.shape
            if min(batch, channels, height, width) <= 0:
                raise ValueError("Distillation features cannot have empty dimensions")
            student = student.flatten(2).transpose(1, 2)
            teacher = teacher.flatten(2).transpose(1, 2)
            num_positions = height * width
            sample_count = max(1, int(num_positions * self.sample_ratio))
            positions = torch.randperm(num_positions, device=student.device)[:sample_count]
            similarity = F.cosine_similarity(student[:, positions], teacher[:, positions], dim=-1)
            losses.append(1 - similarity.mean())
        return torch.stack(losses).mean()
