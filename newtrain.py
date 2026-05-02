
from model.datasets import *
from model.loss import *
from model.model import *


def dice_coeff(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    targets = (targets > 0.5).float()
    smooth = 1e-6
    intersection = (preds * targets).sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) + smooth)
    return dice.mean().item()


def iou_score(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    targets = (targets > 0.5).float()
    smooth = 1e-6
    intersection = (preds * targets).sum(dim=(1, 2))
    union = (preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) - intersection) + smooth
    return (intersection / union).mean().item()


# =========================
# 配置部分
# =========================
DATASETS_CONFIG = {
    "Kvasir": r"/root/code/KvasirSEG",
    "ClinicDB": r"/root/code/CVC-ClinicDB",
    "ColonDB": r"/root/code/CVC-ColonDB",
    "EndoScene": r"/root/code/CVC-300",
    "ETIS": r"/root/code/ETIS"
}

SAM_CKPT = r"/root/code/sam_vit_b_01ec64.pth"
SAM_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 1
WORKERS = 8
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 0.0
EVAL_INTERVAL = 4
OUT_DIR = "./outputs_sam_distill_aug"
SEED = 42
DISTILL_ALPHA = 0.05
W = 5


# -----------------------------
# 1. 通用工具与修复后的 Collate
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_images_masks(root: str) -> List[Tuple[str, str]]:
    img_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    if not os.path.exists(img_dir): return []
    pairs = []
    for f in sorted(os.listdir(img_dir)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            mask_path = os.path.join(mask_dir, f)
            if os.path.exists(mask_path):
                pairs.append((os.path.join(img_dir, f), mask_path))
    return pairs


def split_train_val(pairs, train_ratio=0.9, seed=42):
    random.seed(seed)
    random.shuffle(pairs)
    n = int(len(pairs) * train_ratio)
    return pairs[:n], pairs[n:]


def collate_fn(batch):
    res = {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch])
    }
    if "name" in batch[0]:
        res["name"] = [item["name"] for item in batch]
    return res


# -----------------------------
# 5. 训练与验证 (已修改：分开打印 Loss)
# -----------------------------
def train_one_epoch(model, loader, optimizer, seg_loss_fn, distill_loss_fn, epoch_idx):
    model.train()
    total_loss, total_seg, total_dist = 0.0, 0.0, 0.0
    n = 0

    pbar = tqdm(loader, desc=f"Train [{epoch_idx}/{EPOCHS}]", ncols=150)
    for batch in pbar:
        images = batch["image"].to(DEVICE)
        masks = batch["mask"].to(DEVICE)
        B = images.size(0)

        logits, feats, _ = model(images, return_distill=True)
        gt = F.interpolate(masks.unsqueeze(1), size=(256, 256), mode="nearest").squeeze(1)

        # 1. 计算分割损失
        loss_seg = seg_loss_fn(logits, gt)

        # 2. 计算蒸馏损失
        # 改进点：确保 loss_dist 始终是可微分的 Tensor，即使为 0
        loss_dist = torch.tensor(0.0, device=DEVICE, requires_grad=True)

        if feats and "student" in feats and "teacher" in feats:
            # 只有当确实有特征产出时才计算损失
            loss_dist = distill_loss_fn(feats["student"], feats["teacher"])

        # 3. 总损失
        loss = loss_seg + DISTILL_ALPHA * loss_dist

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # 累计损失值
        total_loss += loss.item() * B
        total_seg += loss_seg.item() * B

        # 改进点：使用 float() 转换，它同时兼容 Tensor 和 Python 数字
        # 或者先检查类型，这样最安全
        dist_val = loss_dist.item() if isinstance(loss_dist, torch.Tensor) else loss_dist
        total_dist += dist_val * B

        n += B

        pbar.set_postfix({
            "Loss": f"{total_loss / n:.4f}",
            "Seg": f"{total_seg / n:.4f}",
            "Dist": f"{total_dist / n:.4f}"
        })

    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    dices, ious = [], []
    for batch in loader:
        images = batch["image"].to(DEVICE)
        masks = batch["mask"].to(DEVICE)

        logits, _ = model(images, return_distill=False)
        gt = F.interpolate(masks.unsqueeze(1), size=(256, 256), mode="nearest").squeeze(1)

        dices.append(dice_coeff(logits, gt))
        ious.append(iou_score(logits, gt))

    return {
        "dice": np.mean(dices) if dices else 0.0,
        "iou": np.mean(ious) if ious else 0.0
    }


# -----------------------------
# 6. 主程序
# -----------------------------
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Collecting data from 5 datasets...")
    data_lists = {k: list_images_masks(v) for k, v in DATASETS_CONFIG.items()}

    train_pairs_all = []
    test_loaders = {}

    for name in ["Kvasir", "ClinicDB"]:
        if len(data_lists[name]) > 0:
            tr, te = split_train_val(data_lists[name], train_ratio=0.9, seed=SEED)
            train_pairs_all.extend(tr)
            test_loaders[name] = DataLoader(PolypDataset(te, is_train=False),
                                            batch_size=BATCH, shuffle=False,
                                            num_workers=WORKERS, collate_fn=collate_fn)
            print(f" -> {name}: Train {len(tr)}, Test {len(te)}")

    for name in ["ColonDB", "EndoScene", "ETIS"]:
        if len(data_lists[name]) > 0:
            test_loaders[name] = DataLoader(PolypDataset(data_lists[name], is_train=False),
                                            batch_size=BATCH, shuffle=False,
                                            num_workers=WORKERS, collate_fn=collate_fn)
            print(f" -> {name}: All Test {len(data_lists[name])}")

    random.shuffle(train_pairs_all)
    train_loader = DataLoader(PolypDataset(train_pairs_all, is_train=True),
                              batch_size=BATCH, shuffle=True,
                              num_workers=WORKERS, collate_fn=collate_fn, drop_last=True)

    model = PolypSAM(SAM_TYPE, SAM_CKPT, W=W).to(DEVICE)
    loss_seg = BCEDiceLoss()
    loss_dist = AlignedDistillationLoss().to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad] + list(loss_dist.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Loss Params: {sum(p.numel() for p in loss_dist.parameters()):,}")

    print(f"Total Trainable Params: {model_trainable + sum(p.numel() for p in loss_dist.parameters()):,}")

    best_avg_dice = 0.0
    best_stats = {name: {"dice": 0.0, "iou": 0.0} for name in test_loaders.keys()}

    print("\nStart Training...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_seg, loss_dist, epoch)

        if epoch % EVAL_INTERVAL == 0 or epoch == EPOCHS:
            print(f"--- Validation Epoch {epoch} ---")
            current_dices = []
            current_ious = []

            for name, loader in test_loaders.items():
                metrics = evaluate(model, loader)
                d, i = metrics["dice"], metrics["iou"]
                current_dices.append(d)
                current_ious.append(i)

                print(f"  > {name:<10}: Dice {d:.4f} | IoU {i:.4f}")

                if d > best_stats[name]["dice"]:
                    best_stats[name]["dice"] = d
                    best_stats[name]["iou"] = i

            avg_dice = float(np.mean(current_dices))
            avg_iou = float(np.mean(current_ious))
            print(f"  >> Epoch {epoch} AVG: Dice {avg_dice:.4f}, IoU {avg_iou:.4f}")

            if avg_dice > best_avg_dice:
                best_avg_dice = avg_dice
                torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))
                print(f"[*] New Best Avg Dice: {best_avg_dice:.4f} (Saved)")

    print("\n" + "=" * 65)
    print(f"{'Dataset':<15} | {'Best Dice':<12} | {'Corresponding IoU':<12}")
    print("-" * 65)
    for name, m in best_stats.items():
        print(f"{name:<15} | {m['dice']:<12.4f} | {m['iou']:<12.4f}")
    print("-" * 65)
    print(f"Final Best Average Dice: {best_avg_dice:.4f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()