import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# ==============================================================================
# 1. 定位器：DinoPromptDecoder (含 Top-K NMS)
# ==============================================================================
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            # Depthwise: 负责空间特征提取
            nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding,
                      dilation=dilation, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            # Pointwise: 负责通道融合
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# ==============================================================================
# 轻量化解码器 (Max 2M Params)
# ==============================================================================
class DinoPromptDecoder(nn.Module):
    def __init__(self, in_dim=768, embed_dim=128, num_classes=1):
        super().__init__()

        # 1. 投影层：将 768 维降至 128 维 (使用 DSConv)
        # 参数量从 7M 降至约 0.4M
        self.proj0 = DSConv(in_dim, embed_dim)
        self.proj1 = DSConv(in_dim, embed_dim)
        self.proj2 = DSConv(in_dim, embed_dim)
        self.proj3 = DSConv(in_dim, embed_dim)

        # 2. 逐层融合路径 (Hierarchical Fusion)
        # 使用轻量化融合，参数量极小
        self.fuse23 = DSConv(embed_dim * 2, embed_dim)
        self.fuse12 = DSConv(embed_dim * 2, embed_dim)
        self.fuse01 = DSConv(embed_dim * 2, embed_dim)

        # 3. 增强预测头：由深变浅的压制，强化中心点回归
        self.mlp_head = nn.Sequential(
            DSConv(embed_dim, 64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def get_topk_confident_points(self, logits, target_size=1024, k=1, nms_kernel=15):
        B, _, H, W = logits.shape

        probs = torch.sigmoid(logits)
        smoothed = TF.gaussian_blur(probs, kernel_size=[11, 11], sigma=[3.0])

        pad = nms_kernel // 2
        max_pooled = F.max_pool2d(smoothed, kernel_size=nms_kernel, stride=1, padding=pad)
        keep_mask = (smoothed == max_pooled)

        # 提取逻辑
        nms_logits = torch.where(keep_mask, smoothed, torch.zeros_like(smoothed))
        flat_logits = nms_logits.view(B, -1)
        _, topk_indices = torch.topk(flat_logits, k=k, dim=1)

        py = topk_indices // W
        px = topk_indices % W

        real_px = (px.float() + 0.5) * (target_size / W)
        real_py = (py.float() + 0.5) * (target_size / H)

        return torch.stack([real_px, real_py], dim=2), torch.ones((B, k), dtype=torch.int64, device=logits.device)

    def forward(self, features_list, input_size=(1024, 1024)):
        h_feat, w_feat = input_size[0] // 14, input_size[1] // 14

        def reshape_feat(x):
            if x.dim() == 3:
                B, N, C = x.shape
                return x[:, -(h_feat * w_feat):, :].transpose(1, 2).view(B, C, h_feat, w_feat)
            return x

        # 多尺度投影
        f0 = self.proj0(reshape_feat(features_list[0]))
        f1 = self.proj1(reshape_feat(features_list[1]))
        f2 = self.proj2(reshape_feat(features_list[2]))
        f3 = self.proj3(reshape_feat(features_list[3]))

        # 逐级融合
        f2 = self.fuse23(torch.cat([f2, f3], dim=1))
        f1 = self.fuse12(torch.cat([f1, f2], dim=1))
        f0 = self.fuse01(torch.cat([f0, f1], dim=1))

        logits = self.mlp_head(f0)

        with torch.no_grad():
            point_coords, point_labels = self.get_topk_confident_points(logits, target_size=input_size[0])

        pred_mask_256 = F.interpolate(logits, size=(256, 256), mode='bilinear', align_corners=False)[:, 0, :, :]
        return pred_mask_256, point_coords, point_labels