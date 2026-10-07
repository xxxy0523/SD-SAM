"""Reproducible loading, checkpointing and experiment bookkeeping."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.datasets import PolypDataset, list_images_masks, load_or_create_splits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def resolve_config(config, project_root):
    """Resolve relative paths against the repository, never the launch directory."""
    result = dict(config)
    root = Path(project_root)
    for key in ("dino_checkpoint", "sam_checkpoint", "prompt_checkpoint", "output_dir",
                "split_manifest", "evaluation_checkpoint"):
        if key in result and result[key] is not None:
            result[key] = str((root / result[key]).resolve())
    result["datasets"] = {
        name: str((root / path).resolve())
        for name, path in config["datasets"].items() if path is not None
    }
    if "train_ratios" in config:
        result["train_ratios"] = {name: config["train_ratios"][name] for name in result["datasets"]}
    else:
        result["train_counts"] = {name: config["train_counts"][name] for name in result["datasets"]}
    return result


def prepare_data(config):
    if "train_ratios" in config:
        counts = {}
        for name, root in config["datasets"].items():
            ratio = config["train_ratios"][name]
            if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not 0 <= ratio <= 1:
                raise ValueError(f"Invalid train ratio for {name}: {ratio!r}")
            counts[name] = int(len(list_images_masks(root)) * ratio)
    else:
        counts = config["train_counts"]
    splits = load_or_create_splits(config["datasets"], counts,
                                   config["seed"], config["split_manifest"])
    if not sum(map(len, splits["train"].values())):
        raise ValueError("No training samples configured.")
    for name in config["datasets"]:
        print(f"{name}: train={len(splits['train'].get(name, []))}, "
              f"test={len(splits['test'].get(name, []))}")
    return splits


def make_loader(pairs, config, training):
    augmentation = config["augmentation"] if training else {}
    dataset = PolypDataset(pairs, image_size=config["image_size"],
                           mean=config["mean"], std=config["std"],
                           is_train=training, mask_threshold=config["mask_threshold"],
                           suppress_opencv_warnings=config["suppress_opencv_warnings"],
                           **augmentation)
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=training,
                      num_workers=config["num_workers"], pin_memory=config["device"].startswith("cuda"),
                      drop_last=False, worker_init_fn=seed_worker, generator=generator)


def make_loaders(splits, config):
    train_pairs = [pair for pairs in splits["train"].values() for pair in pairs]
    train = make_loader(train_pairs, config, True)
    tests = {name: make_loader(pairs, config, False)
             for name, pairs in splits["test"].items() if pairs}
    return train, tests


def split_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_files(config, keys):
    layers = config["feature_layers"]
    if len(layers) != 4 or layers != sorted(set(layers)) or any(i < 0 or i > 11 for i in layers):
        raise ValueError("feature_layers must contain four unique increasing indices in [0, 11].")
    if config["image_size"] % 16 or config["mask_size"] != config["image_size"] // 4:
        raise ValueError("Use an image_size divisible by 16 and mask_size=image_size//4 (SAM output grid).")
    missing = [f"{key}: {config[key]}" for key in keys if not Path(config[key]).is_file()]
    if missing:
        raise FileNotFoundError("Set these paths in the training file:\n" + "\n".join(missing))
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; change CONFIG['device'] to 'cpu'.")
    config["dino_sha256"] = file_digest(config["dino_checkpoint"])


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_checkpoint(path, model, optimizer, epoch, config, **extra):
    """Atomic local save; a failed write does not replace the previous checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "epoch": epoch, "config": config,
               "split_sha256": split_digest(config["split_manifest"]), **extra}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_prompt_metadata(checkpoint, config):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "config" not in payload:
        raise ValueError("Legacy APG weights contain no split/preprocessing metadata. Retrain Stage 1 "
                         "with train_stage1.py to ensure the two stages share the same split.")
    if payload.get("split_sha256") != split_digest(config["split_manifest"]):
        raise ValueError("Stage 1 and Stage 2 split manifests differ. Use the same split and retrain Stage 1.")
    if payload["config"].get("dino_sha256") != config["dino_sha256"]:
        raise ValueError("Stage 1 and Stage 2 use different DINOv3 checkpoint contents.")
    for key in ("image_size", "mask_size", "mask_threshold", "mean", "std", "feature_layers", "prompt_config"):
        left, right = payload["config"][key], config[key]
        if json.dumps(left, sort_keys=True) != json.dumps(right, sort_keys=True):
            raise ValueError(f"Stage 1/2 {key} differs: {left} != {right}")


def parameter_counts(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
