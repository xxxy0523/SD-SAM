"""Automatic point-prompt generator shared by both training stages."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur


class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding,
                      dilation=dilation, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DinoPromptDecoder(nn.Module):
    """Fuse four DINO patch maps and regress a point-confidence heatmap.

    All experiment settings are supplied by the stage training script. Module
    names retain compatibility with the original APG state dictionaries.
    """

    def __init__(self, *, in_dim, embed_dim, patch_size, mask_size,
                 num_points, nms_kernel, blur_kernel, blur_sigma, num_classes=1):
        super().__init__()
        if num_classes != 1:
            raise ValueError("Point generation requires a single confidence channel.")
        if patch_size < 1 or num_points < 1 or blur_sigma <= 0:
            raise ValueError("patch_size, num_points and blur_sigma must be positive.")
        if any(k < 1 or k % 2 == 0 for k in (nms_kernel, blur_kernel)):
            raise ValueError("NMS and Gaussian kernels must be positive odd integers.")
        self.patch_size = patch_size
        self.mask_size = (mask_size, mask_size) if isinstance(mask_size, int) else tuple(mask_size)
        self.num_points = num_points
        self.nms_kernel = nms_kernel
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.proj0 = DSConv(in_dim, embed_dim)
        self.proj1 = DSConv(in_dim, embed_dim)
        self.proj2 = DSConv(in_dim, embed_dim)
        self.proj3 = DSConv(in_dim, embed_dim)
        self.fuse23 = DSConv(embed_dim * 2, embed_dim)
        self.fuse12 = DSConv(embed_dim * 2, embed_dim)
        self.fuse01 = DSConv(embed_dim * 2, embed_dim)
        self.mlp_head = nn.Sequential(
            DSConv(embed_dim, 64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    @torch.no_grad()
    def get_topk_confident_points(self, logits, target_size):
        batch, _, height, width = logits.shape
        target_h, target_w = (target_size, target_size) if isinstance(target_size, int) else target_size
        if self.num_points > height * width:
            raise ValueError("num_points exceeds the number of heatmap pixels.")
        if min(height, width) <= self.blur_kernel // 2:
            raise ValueError("Heatmap is too small for Gaussian reflect padding; reduce blur_kernel.")
        smoothed = gaussian_blur(torch.sigmoid(logits),
                                 [self.blur_kernel, self.blur_kernel],
                                 [self.blur_sigma, self.blur_sigma])
        pooled = F.max_pool2d(smoothed, self.nms_kernel, stride=1,
                             padding=self.nms_kernel // 2)
        scores = torch.where(smoothed == pooled, smoothed, torch.zeros_like(smoothed))
        indices = torch.topk(scores.reshape(batch, -1), self.num_points, dim=1).indices
        px = (indices % width + 0.5) * (target_w / width)
        py = (indices // width + 0.5) * (target_h / height)
        labels = torch.ones((batch, self.num_points), dtype=torch.int64, device=logits.device)
        return torch.stack((px, py), dim=-1), labels

    def forward(self, features_list, input_size):
        if len(features_list) != 4:
            raise ValueError("APG requires exactly four feature maps.")
        height, width = (int(v) for v in input_size)
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Input size must be divisible by the DINO patch size.")
        grid_h, grid_w = height // self.patch_size, width // self.patch_size

        def reshape_feat(x):
            if x.ndim == 3:
                if x.shape[1] != grid_h * grid_w:
                    raise ValueError("Expected patch tokens only; remove CLS/storage tokens first.")
                x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], grid_h, grid_w)
            if x.ndim != 4 or x.shape[-2:] != (grid_h, grid_w):
                raise ValueError("DINO feature grid does not match input_size / patch_size.")
            return x

        f0, f1, f2, f3 = [getattr(self, f"proj{i}")(reshape_feat(feat))
                          for i, feat in enumerate(features_list)]
        f2 = self.fuse23(torch.cat((f2, f3), dim=1))
        f1 = self.fuse12(torch.cat((f1, f2), dim=1))
        f0 = self.fuse01(torch.cat((f0, f1), dim=1))
        logits = self.mlp_head(f0)
        points, labels = self.get_topk_confident_points(logits, (height, width))
        heatmap = F.interpolate(logits, self.mask_size, mode="bilinear", align_corners=False)[:, 0]
        return heatmap, points, labels
