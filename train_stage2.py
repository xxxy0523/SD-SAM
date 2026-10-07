"""Stage 2: freeze the prompt network and adapt SAM using DINOv3.

Edit CONFIG below, then run `python train_stage2.py` after Stage 1.
Use `--check` to validate paths, split and prompt metadata without loading SAM.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model.loss import AlignedDistillationLoss, BCEDiceLoss, metrics_per_image
from model.model import PolypSAM
from utils.training import (make_loaders, parameter_counts, prepare_data, require_files,
                            resolve_config, save_checkpoint, save_json, set_seed,
                            validate_prompt_metadata)


# All Stage 2 experiment settings. Shared settings must agree with Stage 1.
CONFIG = {
    "datasets": {
        "Kvasir": "data/Kvasir-SEG", "ClinicDB": "data/CVC-ClinicDB",
        "ColonDB": "data/CVC-ColonDB", "CVC-300": "data/CVC-300",
    },
    "train_ratios": {"Kvasir": 0.9, "ClinicDB": 0.9, "ColonDB": 0.0,
                     "CVC-300": 0.0, "ETIS": 0.0},
    "split_manifest": "outputs/splits/five_datasets.json",
    "dino_checkpoint": "checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "sam_checkpoint": "checkpoints/sam_vit_b_01ec64.pth",
    "prompt_checkpoint": "outputs/stage1/prompt_decoder.pth",
    "output_dir": "outputs/stage2",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "image_size": 1024,
    "mask_size": 256,
    "mask_threshold": 0.5,
    "suppress_opencv_warnings": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "batch_size": 1,
    "num_workers": 8,
    "epochs": 50,
    "eval_interval": 5,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "sam_type": "vit_b",
    "W": 5,  # Blocks i < W use shallow distillation; i >= W use deep fusion.
    "adapter_bottleneck": 64,
    "fusion_bottleneck": 64,
    "sam_window_size": 0,  # Preserves supplied global attention in all blocks.
    "feature_layers": [2, 5, 8, 11],
    "prompt_config": {"in_dim": 768, "embed_dim": 256, "patch_size": 16,
                      "mask_size": 256, "num_points": 1, "nms_kernel": 15,
                      "blur_kernel": 11, "blur_sigma": 3.0},
    "bce_weight": 0.5,
    "distill_weight": 0.05,
    "distill_dim": 768,
    "distill_use_projection": True,
    "distill_bottleneck": 128,
    "distill_sample_ratio": 0.25,
    "metric_threshold": 0.5,
    "augmentation": {"horizontal_flip_prob": 0.0, "vertical_flip_prob": 0.0,
                     "rotation90_prob": 0.0},
}


def train_one_epoch(model, loader, optimizer, segmentation_loss, distillation_loss, config):
    model.train()  # Frozen DINO/APG branches remain in eval mode.
    distillation_loss.train()
    totals = {"loss": 0.0, "segmentation": 0.0, "distillation": 0.0}
    samples = 0
    progress = tqdm(loader, desc="Stage 2 train")
    for batch in progress:
        images = batch["image"].to(config["device"])
        masks = batch["mask"].to(config["device"])
        logits, features, _ = model(images, return_distill=True)
        target = F.interpolate(masks.unsqueeze(1), size=logits.shape[-2:], mode="nearest")[:, 0]
        loss_seg = segmentation_loss(logits, target)
        loss_dist = (distillation_loss(features["student"], features["teacher"])
                     if features["student"] else logits.sum() * 0.0)
        loss = loss_seg + config["distill_weight"] * loss_dist
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite Stage 2 loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        samples += len(images)
        for key, value in (("loss", loss), ("segmentation", loss_seg), ("distillation", loss_dist)):
            totals[key] += value.item() * len(images)
        progress.set_postfix({key: value / samples for key, value in totals.items()})
    return {key: value / samples for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, loader, config):
    model.eval()
    totals = {"dice": 0.0, "iou": 0.0}
    samples = 0
    for batch in tqdm(loader, desc="Stage 2 test", leave=False):
        images = batch["image"].to(config["device"])
        masks = batch["mask"].to(config["device"])
        logits, _ = model(images, return_distill=False)
        target = F.interpolate(masks.unsqueeze(1), size=logits.shape[-2:], mode="nearest")[:, 0]
        metrics = metrics_per_image(logits, target, threshold=config["metric_threshold"])
        for key in totals:
            totals[key] += metrics[key].sum().item()
        samples += len(images)
    return {**{key: value / samples for key, value in totals.items()}, "samples": samples}


def main(check_only=False):
    config = resolve_config(CONFIG, Path(__file__).resolve().parent)
    if not 0 <= config["W"] <= 12:
        raise ValueError("W must be in [0, 12].")
    if config["mask_size"] != config["prompt_config"]["mask_size"]:
        raise ValueError("mask_size must match prompt_config['mask_size'].")
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["eval_interval"] < 1:
        raise ValueError("epochs, batch_size and eval_interval must be positive.")
    require_files(config, ["dino_checkpoint", "sam_checkpoint", "prompt_checkpoint"])
    set_seed(config["seed"])
    splits = prepare_data(config)
    validate_prompt_metadata(config["prompt_checkpoint"], config)
    if check_only:
        print("Stage 2 paths, split and prompt metadata validated. No training started.")
        return
    output = Path(config["output_dir"])
    save_json(output / "config.json", config)
    train_loader, test_loaders = make_loaders(splits, config)
    model = PolypSAM(sam_type=config["sam_type"], sam_ckpt=config["sam_checkpoint"],
                    dino_backbone_ckpt=config["dino_checkpoint"], prompt_ckpt=config["prompt_checkpoint"],
                    image_size=config["image_size"], shallow_layers=config["W"],
                    adapter_bottleneck=config["adapter_bottleneck"], fusion_bottleneck=config["fusion_bottleneck"],
                    feature_layers=config["feature_layers"], prompt_config=config["prompt_config"],
                    window_size=config["sam_window_size"]).to(config["device"])
    segmentation_loss = BCEDiceLoss(bce_weight=config["bce_weight"])
    distillation_loss = AlignedDistillationLoss(dim=config["distill_dim"],
                                               use_proj=config["distill_use_projection"],
                                               sample_ratio=config["distill_sample_ratio"],
                                               bottleneck_dim=config["distill_bottleneck"]).to(config["device"])
    if config["W"] == 0 or config["distill_weight"] == 0:
        distillation_loss.requires_grad_(False)
    parameters = [p for module in (model, distillation_loss) for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    counts = {"model": parameter_counts(model), "distillation_loss": parameter_counts(distillation_loss)}
    counts["optimizer_trainable"] = sum(p.numel() for p in parameters)
    counts["frozen_prompt_decoder"] = parameter_counts(model.auto_prompt_gen)["total"]
    counts["both_stages_trained"] = counts["optimizer_trainable"] + counts["frozen_prompt_decoder"]
    save_json(output / "parameters.json", counts)
    print(counts)
    history = []
    best_average = -1.0
    best_average_iou = None
    best_metrics = None
    best_epoch = None
    best_per_dataset = {name: {"dice": -1.0, "iou": -1.0, "epoch": None}
                        for name in test_loaders}
    for epoch in range(1, config["epochs"] + 1):
        losses = train_one_epoch(model, train_loader, optimizer, segmentation_loss, distillation_loss, config)
        record = {"epoch": epoch, **losses}
        if epoch % config["eval_interval"] == 0 or epoch == config["epochs"]:
            metrics = {name: evaluate(model, loader, config) for name, loader in test_loaders.items()}
            average = sum(item["dice"] for item in metrics.values()) / len(metrics)
            average_iou = sum(item["iou"] for item in metrics.values()) / len(metrics)
            record["evaluation"] = metrics
            record["average_dice"] = average
            record["average_iou"] = average_iou
            for name, item in metrics.items():
                if item["dice"] > best_per_dataset[name]["dice"]:
                    best_per_dataset[name] = {"dice": item["dice"], "iou": item["iou"], "epoch": epoch}
            if average > best_average:
                best_average, best_average_iou = average, average_iou
                best_metrics, best_epoch = metrics, epoch
                save_checkpoint(output / "model.pth", model, None, epoch, config,
                                distillation_loss=distillation_loss.state_dict(),
                                selection_metric="average_dice", selection_value=average)
            print(f"Epoch {epoch}/{config['epochs']}: {losses}, average_dice={average:.6f}, "
                  f"average_iou={average_iou:.6f}")
        else:
            print(f"Epoch {epoch}/{config['epochs']}: {losses}")
        history.append(record)
        save_json(output / "history.json", history)
        save_checkpoint(output / "last.pth", model, optimizer, epoch, config,
                        distillation_loss=distillation_loss.state_dict())
    metrics_payload = dict(best_metrics)
    metrics_payload["_selection"] = {"metric": "average_dice", "value": best_average,
                                     "average_iou": best_average_iou, "epoch": best_epoch}
    metrics_payload["_best_per_dataset"] = best_per_dataset
    save_json(output / "test_metrics.json", metrics_payload)
    print(metrics_payload)
    print(f"Stage 2 complete: {output / 'model.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check inputs without training")
    main(check_only=parser.parse_args().check)
