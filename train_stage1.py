"""Stage 1: train the Gaussian prompt decoder on frozen DINOv3 features.

Edit CONFIG below, then run `python train_stage1.py`.
Use `--check` to validate paths and the shared split without loading the backbone.
"""
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from dinov3.loadmodel import load_dinov3_vitb16
from model.loss import heatmap_regression_loss
from selfprompt.selfprompt import DinoPromptDecoder
from utils.training import (make_loaders, parameter_counts, prepare_data, require_files,
                            resolve_config, save_checkpoint, save_json, set_seed)


# All Stage 1 experiment settings. Relative paths start at this file's directory.
CONFIG = {
    "datasets": {
        "Kvasir": "data/Kvasir-SEG", "ClinicDB": "data/CVC-ClinicDB",
        "ColonDB": "data/CVC-ColonDB", "CVC-300": "data/CVC-300",
    },
    "train_ratios": {"Kvasir": 0.9, "ClinicDB": 0.9, "ColonDB": 0.0,
                     "CVC-300": 0.0, "ETIS": 0.0},
    "split_manifest": "outputs/splits/five_datasets.json",
    "dino_checkpoint": "checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "output_dir": "outputs/stage1",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "image_size": 1024,
    "mask_size": 256,
    "mask_threshold": 0.5,
    "suppress_opencv_warnings": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "batch_size": 2,
    "num_workers": 0,  # Portable Windows default; increase for your server.
    "epochs": 50,
    "eval_interval": 5,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "feature_layers": [2, 5, 8, 11],  # Zero-based DINO blocks.
    "prompt_config": {"in_dim": 768, "embed_dim": 256, "patch_size": 16,
                      "mask_size": 256, "num_points": 1, "nms_kernel": 15,
                      "blur_kernel": 11, "blur_sigma": 3.0},
    "gaussian_sigma": 20.0,  # Pixels on the 256 x 256 heatmap.
    "mse_weight": 10.0,
    "bce_weight": 1.0,
    "augmentation": {"horizontal_flip_prob": 0.0, "vertical_flip_prob": 0.0,
                     "rotation90_prob": 0.0},
}


def train_one_epoch(backbone, decoder, loader, optimizer, config):
    backbone.eval()
    decoder.train()
    total_loss, samples = 0.0, 0
    progress = tqdm(loader, desc="Stage 1 train")
    for batch in progress:
        images = batch["image"].to(config["device"])
        masks = batch["mask"].to(config["device"])
        with torch.no_grad():
            features = backbone.get_intermediate_layers(images, n=config["feature_layers"], reshape=True)
        logits, _, _ = decoder(features, input_size=images.shape[-2:])
        loss = heatmap_regression_loss(logits, masks, sigma=config["gaussian_sigma"],
                                      mse_weight=config["mse_weight"], bce_weight=config["bce_weight"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite Stage 1 loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        samples += len(images)
        total_loss += loss.item() * len(images)
        progress.set_postfix(loss=total_loss / samples)
    return total_loss / samples


@torch.no_grad()
def evaluate(backbone, decoder, loader, config):
    backbone.eval()
    decoder.eval()
    hits, count = 0, 0
    for batch in tqdm(loader, desc="Stage 1 test", leave=False):
        images = batch["image"].to(config["device"])
        masks = batch["mask"].to(config["device"])
        features = backbone.get_intermediate_layers(images, n=config["feature_layers"], reshape=True)
        _, points, _ = decoder(features, input_size=images.shape[-2:])
        x = points[:, 0, 0].round().long().clamp(0, masks.shape[-1] - 1)
        y = points[:, 0, 1].round().long().clamp(0, masks.shape[-2] - 1)
        hits += int((masks[torch.arange(len(masks), device=masks.device), y, x] > 0.5).sum())
        count += len(masks)
    return {"hit_rate": hits / count, "hits": hits, "samples": count}


def main(check_only=False):
    config = resolve_config(CONFIG, Path(__file__).resolve().parent)
    if config["mask_size"] != config["prompt_config"]["mask_size"]:
        raise ValueError("mask_size must match prompt_config['mask_size'].")
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["eval_interval"] < 1:
        raise ValueError("epochs, batch_size and eval_interval must be positive.")
    require_files(config, ["dino_checkpoint"])
    set_seed(config["seed"])
    splits = prepare_data(config)
    if check_only:
        print("Stage 1 paths and shared split validated. No training started.")
        return
    output = Path(config["output_dir"])
    save_json(output / "config.json", config)
    train_loader, test_loaders = make_loaders(splits, config)
    backbone = load_dinov3_vitb16(config["dino_checkpoint"]).to(config["device"]).eval()
    backbone.requires_grad_(False)
    decoder = DinoPromptDecoder(**config["prompt_config"]).to(config["device"])
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    counts = parameter_counts(decoder)
    print(f"Stage 1 decoder: {counts['trainable']:,} trainable parameters")
    save_json(output / "parameters.json", {"decoder": counts, "backbone": parameter_counts(backbone)})
    history = []
    best_average = -1.0
    best_metrics = None
    best_epoch = None
    for epoch in range(1, config["epochs"] + 1):
        loss = train_one_epoch(backbone, decoder, train_loader, optimizer, config)
        record = {"epoch": epoch, "loss": loss}
        if epoch % config["eval_interval"] == 0 or epoch == config["epochs"]:
            metrics = {name: evaluate(backbone, decoder, loader, config)
                       for name, loader in test_loaders.items()}
            average = sum(item["hit_rate"] for item in metrics.values()) / len(metrics)
            record["evaluation"] = metrics
            record["average_hit_rate"] = average
            if average >= best_average:
                best_average, best_metrics, best_epoch = average, metrics, epoch
                save_checkpoint(output / "prompt_decoder.pth", decoder, None, epoch, config,
                                selection_metric="average_hit_rate", selection_value=average)
            print(f"Epoch {epoch}/{config['epochs']}: loss={loss:.6f}, "
                  f"average_hit_rate={average:.6f}")
        else:
            print(f"Epoch {epoch}/{config['epochs']}: loss={loss:.6f}")
        history.append(record)
        save_json(output / "history.json", history)
        save_checkpoint(output / "last.pth", decoder, optimizer, epoch, config)
    metrics_payload = dict(best_metrics)
    metrics_payload["_selection"] = {"metric": "average_hit_rate", "value": best_average,
                                     "epoch": best_epoch}
    save_json(output / "test_metrics.json", metrics_payload)
    print(metrics_payload)
    print(f"Stage 1 complete: {output / 'prompt_decoder.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check inputs without training")
    main(check_only=parser.parse_args().check)
