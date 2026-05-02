
import os
import random
import numpy as np
from typing import Tuple, List
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image, UnidentifiedImageError
import cv2
# 假设环境已安装 segment_anything
from segment_anything.build_sam import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide




def list_images_masks(root: str) -> List[Tuple[str, str]]:
    img_dir = os.path.join(root, "images")
    msk_dir = os.path.join(root, "masks")
    if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
        raise ValueError(f"Expect {root}/images and {root}/masks")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    items = []
    for fn in sorted(os.listdir(img_dir)):
        name, ext = os.path.splitext(fn)
        if ext.lower() not in exts:
            continue
        m = None
        for me in exts:
            cand = os.path.join(msk_dir, name + me)
            if os.path.exists(cand):
                m = cand
                break
        if m is None:
            continue
        items.append((os.path.join(img_dir, fn), m))

    if len(items) == 0:
        raise ValueError("No image/mask pairs found.")
    return items


def split_train_val(pairs: List[Tuple[str, str]], train_ratio=0.8, seed=42):
    rng = random.Random(seed)
    pairs = pairs.copy()
    rng.shuffle(pairs)
    n_train = int(len(pairs) * train_ratio + 0.5)
    return pairs[:n_train], pairs[n_train:]


def mask_to_pos_point(mask: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return w // 2, h // 2
    idx = np.random.randint(0, len(xs))
    return int(xs[idx]), int(ys[idx])


# -----------------------------
# 2. 数据集
# -----------------------------
class KvasirSegDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], img_size_long=1024):
        self.pairs = pairs

        # 仅保留 Resize 和 Normalization
        self.resize_long = ResizeLongestSide(img_size_long)
        self.img_norm = T.Normalize(mean=[123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0],
                                    std=[58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                new_idx = random.randint(0, len(self.pairs) - 1)
                return self.__getitem__(new_idx)
            img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        # 2. 读取 Mask (优先 PIL，失败转 OpenCV)
        try:
            msk = Image.open(msk_path).convert("L")
        except (UnidentifiedImageError, OSError):
            msk_cv = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            if msk_cv is None:
                new_idx = random.randint(0, len(self.pairs) - 1)
                return self.__getitem__(new_idx)
            msk = Image.fromarray(msk_cv)

        orig_size_hw = np.array(img.size[::-1])
        img_np = np.array(img)
        msk_np = (np.array(msk) > 0).astype(np.uint8)
        px, py = mask_to_pos_point(msk_np)
        pos_pt = np.array([[px, py]], dtype=np.float32)
        img_resized = self.resize_long.apply_image(img_np)
        msk_resized = self.resize_long.apply_image(msk_np)
        pos_pt_resized = self.resize_long.apply_coords(pos_pt, orig_size_hw)
        h, w = img_resized.shape[:2]
        pad_h = 1024 - h
        pad_w = 1024 - w
        img_padded = np.pad(img_resized, ((0, pad_h), (0, pad_w), (0, 0)),
                            mode='constant', constant_values=0)
        msk_padded = np.pad(msk_resized, ((0, pad_h), (0, pad_w)),
                            mode='constant', constant_values=0)
        img_t = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
        img_t = self.img_norm(img_t)
        msk_t = torch.from_numpy(msk_padded).float()
        point_coord = torch.from_numpy(pos_pt_resized).float()
        point_label = torch.tensor([1], dtype=torch.int64)

        return {
            "image": img_t,
            "mask": msk_t,
            "pt": point_coord,
            "lbl": point_label,
            "orig_size": torch.from_numpy(orig_size_hw),
        }


def collate_fn(batch):
    out = {}
    for k in batch[0].keys():
        if k in ["image", "mask", "pt", "lbl", "orig_size"]:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out
