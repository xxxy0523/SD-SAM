import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2  # 关键：用于处理特殊的 .tif 文件
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 假设您的项目结构中包含这些模块
from .selfprompt import DinoPromptDecoder
from dinov3.loadmodel import load_dinov3_vitb16, CKPT

# ================= 0. 配置参数 =================
CONFIG = {
    "img_size": 1024,
    "batch_size": 2,
    "lr": 1e-4,
    "epochs": 50,
    "val_interval": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 0,
    "dino_dim": 768,
    "seed": 42
}

DATASETS_CONFIG = {
    "Kvasir": r"/root/code/KvasirSEG",
    "ClinicDB": r"/root/code/CVC-ClinicDB",
    "ColonDB": r"/root/code/CVC-ColonDB",
    "EndoScene": r"/root/code/CVC-300",
    "ETIS": r"/root/code/ETIS"
}


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to: {seed}")


# ================= 1. 数据集定义与鲁棒读取 =================
def get_image_mask_pairs(root_path):
    img_dir = os.path.join(root_path, "images")
    mask_dir = os.path.join(root_path, "masks")
    # 支持多种扩展名
    exts = ['*.jpg', '*.png', '*.tif', '*.tiff', '*.bmp']
    img_paths = []
    for ex in exts:
        img_paths.extend(glob.glob(os.path.join(img_dir, ex)))
    img_paths = sorted(img_paths)

    # 匹配 mask (假设文件名一致)
    pairs = []
    for img_p in img_paths:
        base_name = os.path.basename(img_p)
        mask_p = os.path.join(mask_dir, base_name)
        if os.path.exists(mask_p):
            pairs.append((img_p, mask_p))
    return pairs


class MedicalDataset(Dataset):
    def __init__(self, pairs, img_size):
        self.pairs = pairs
        self.img_size = img_size
        self.transform_img = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        try:
            # 使用 cv2 读取以解决 PIL 处理部分 TIFF 报错的问题
            cv_img = cv2.imread(img_path)
            if cv_img is None: raise ValueError("Empty Image")
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(cv_img)

            cv_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if cv_mask is None: raise ValueError("Empty Mask")
            mask = Image.fromarray(cv_mask)
        except Exception as e:
            # 容错处理
            return self.__getitem__(random.randint(0, len(self.pairs) - 1))

        image = self.transform_img(image)
        mask = self.transform_mask(mask)
        mask = (mask > 0.5).float()
        return image, mask


# ================= 2. 损失函数与评价指标 =================
def generate_gaussian_target(gt_mask, sigma=20):
    B, H, W = gt_mask.shape
    device = gt_mask.device
    gaussian_targets = torch.zeros_like(gt_mask)
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device),
                                    torch.arange(W, device=device), indexing='ij')
    for i in range(B):
        mask = gt_mask[i]
        if mask.sum() == 0: continue
        y_indices, x_indices = torch.where(mask > 0.5)
        center_y, center_x = y_indices.float().mean(), x_indices.float().mean()
        dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
        gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))
        gaussian_targets[i] = gaussian * mask
    return gaussian_targets


def calc_loss(pred_logits, gt_mask):
    gt_resized = F.interpolate(gt_mask, size=(256, 256), mode="nearest").squeeze(1)
    gaussian_gt = generate_gaussian_target(gt_resized, sigma=20)
    pred_probs = torch.sigmoid(pred_logits)
    mse_loss = F.mse_loss(pred_probs, gaussian_gt)
    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_resized)
    return 10.0 * mse_loss + 1.0 * bce


def calculate_hit_rate(point_coords, gt_mask):
    B = point_coords.shape[0]
    hits = 0
    pts = point_coords.detach().cpu().numpy()
    masks = gt_mask.detach().cpu().numpy()
    for i in range(B):
        px, py = pts[i, 0]
        ix, iy = int(round(px)), int(round(py))
        H, W = masks.shape[2], masks.shape[3]
        ix, iy = min(max(0, ix), W - 1), min(max(0, iy), H - 1)
        if masks[i, 0, iy, ix] > 0.5:
            hits += 1
    return hits, B


# ================= 3. 主程序 =================
def main():
    set_seed(CONFIG["seed"])

    # A. 数据收集与划分
    train_pairs_all = []
    test_loaders = {}

    for name, path in DATASETS_CONFIG.items():
        pairs = get_image_mask_pairs(path)
        if name in ["Kvasir", "ClinicDB"]:
            tr, te = train_test_split(pairs, test_size=0.1, random_state=CONFIG["seed"])
            train_pairs_all.extend(tr)
            test_loaders[name] = DataLoader(MedicalDataset(te, CONFIG["img_size"]),
                                            batch_size=CONFIG["batch_size"], shuffle=False)
            print(f"📁 {name}: Train={len(tr)}, Test={len(te)}")
        else:
            test_loaders[name] = DataLoader(MedicalDataset(pairs, CONFIG["img_size"]),
                                            batch_size=CONFIG["batch_size"], shuffle=False)
            print(f"📁 {name}: Test={len(pairs)} (All)")

    train_loader = DataLoader(MedicalDataset(train_pairs_all, CONFIG["img_size"]),
                              batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])

    # B. 模型与优化器
    dino_model = load_dinov3_vitb16(CKPT).to(CONFIG["device"]).eval()
    decoder = DinoPromptDecoder(in_dim=CONFIG["dino_dim"], embed_dim=256).to(CONFIG["device"])
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)

    # C. 训练循环
    best_avg_hit_rate = 0.0

    for epoch in range(CONFIG["epochs"]):
        decoder.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}")

        for images, masks in pbar:
            images, masks = images.to(CONFIG["device"]), masks.to(CONFIG["device"])
            with torch.no_grad():
                all_layers = dino_model.get_intermediate_layers(images, n=12, reshape=True)
                features = [all_layers[i] for i in [2, 5, 8, 11]]

            pred_mask, _, _ = decoder(features, input_size=(CONFIG["img_size"], CONFIG["img_size"]))
            loss = calc_loss(pred_mask, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # D. 验证阶段
        if (epoch + 1) % CONFIG["val_interval"] == 0:
            decoder.eval()
            print(f"\n--- Evaluation (Epoch {epoch + 1}) ---")
            current_hrs = []

            for name, loader in test_loaders.items():
                total_hits, total_samples = 0, 0
                with torch.no_grad():
                    for images, masks in loader:
                        images, masks = images.to(CONFIG["device"]), masks.to(CONFIG["device"])
                        all_layers = dino_model.get_intermediate_layers(images, n=12, reshape=True)
                        features = [all_layers[i] for i in [2, 5, 8, 11]]
                        _, point_coords, _ = decoder(features, input_size=(CONFIG["img_size"], CONFIG["img_size"]))

                        hits, B = calculate_hit_rate(point_coords, masks)
                        total_hits += hits
                        total_samples += B

                hr = total_hits / total_samples if total_samples > 0 else 0
                current_hrs.append(hr)
                print(f"  > {name:<10}: {hr:.2%}")

            avg_hr = np.mean(current_hrs)
            print(f"⭐ Global Average Hit Rate: {avg_hr:.2%}")

            if avg_hr >= best_avg_hit_rate:
                best_avg_hit_rate = avg_hr
                torch.save(decoder.state_dict(), "poly_best_dino_prompt_decoder.pth")
                print("🏆 Best Model Saved!")
            print("-" * 35)


if __name__ == "__main__":
    main()