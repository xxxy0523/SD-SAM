"""Image/mask loading and one reproducible split shared by both training stages."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


Pair = tuple[str, str]
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _files_by_stem(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"Expected a directory: {directory}")
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        stem = path.stem.casefold()
        if stem in files:
            raise ValueError(f"Ambiguous filename stem {path.stem!r}: {files[stem]} and {path}")
        files[stem] = path
    return files


def list_images_masks(root: str | Path) -> list[Pair]:
    """Match images and masks by case-insensitive stem, failing on missing pairs."""
    root = Path(root).expanduser().resolve()
    images = _files_by_stem(root / "images")
    masks = _files_by_stem(root / "masks")
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        raise ValueError(
            f"Unpaired files in {root}: missing masks={missing_masks[:10]}, "
            f"missing images={missing_images[:10]}"
        )
    if not images:
        raise ValueError(f"No image/mask pairs found in {root}")
    return [(str(images[stem]), str(masks[stem])) for stem in sorted(images)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_create_splits(
    dataset_roots: Mapping[str, str | Path | None],
    train_counts: Mapping[str, int],
    seed: int,
    manifest_path: str | Path,
) -> dict[str, dict[str, list[Pair]]]:
    """Create or verify a portable, content-checked train/test split manifest.

    A count of zero makes a dataset test-only. Both stages must use the same
    enabled datasets, counts, seed and manifest. Paths in the JSON are relative
    to each dataset root, so datasets can be relocated without changing splits.
    File hashes detect edited/replaced datasets and byte-identical train/test
    images, including overlap across differently named datasets. This does not
    detect re-encoded copies or establish patient-level independence.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    roots = {name: Path(root).expanduser().resolve() for name, root in dataset_roots.items() if root is not None}
    if not roots:
        raise ValueError("At least one dataset must be enabled")
    missing_counts = roots.keys() - train_counts.keys()
    if missing_counts:
        raise ValueError(f"Set an explicit train count for each enabled dataset: {sorted(missing_counts)}")
    all_pairs = {name: list_images_masks(root) for name, root in sorted(roots.items())}
    counts = {}
    inventory = {}
    for name, pairs in all_pairs.items():
        count = train_counts[name]
        if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= len(pairs):
            raise ValueError(f"Invalid train count {count!r} for {name}: dataset has {len(pairs)} pairs")
        counts[name] = count
        records = []
        for image_path, mask_path in pairs:
            image_path, mask_path = Path(image_path), Path(mask_path)
            records.append({
                "image": image_path.relative_to(roots[name]).as_posix(),
                "mask": mask_path.relative_to(roots[name]).as_posix(),
                "image_sha256": _sha256(image_path),
                "mask_sha256": _sha256(mask_path),
            })
        inventory[name] = records

    manifest_path = Path(manifest_path).expanduser().resolve()
    exists = manifest_path.is_file()
    if exists:
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict) or manifest.get("version") != 1:
            raise ValueError(f"Unsupported split manifest: {manifest_path}")
        if manifest.get("seed") != seed or manifest.get("train_counts") != counts:
            raise ValueError("Split configuration differs from the saved manifest; use the same counts/seed in both stages")
        if manifest.get("inventory") != inventory:
            raise ValueError("Dataset filenames or contents differ from the saved split manifest")
    else:
        membership = {}
        for name, records in inventory.items():
            indices = list(range(len(records)))
            random.Random(seed).shuffle(indices)
            membership[name] = {"train": indices[:counts[name]], "test": indices[counts[name]:]}
        manifest = {"version": 1, "seed": seed, "train_counts": counts, "inventory": inventory, "splits": membership}

    membership = manifest.get("splits", {})
    if not isinstance(membership, dict) or set(membership) != set(roots):
        raise ValueError("Split manifest does not cover exactly the enabled datasets")
    result: dict[str, dict[str, list[Pair]]] = {"train": {}, "test": {}}
    image_membership: dict[str, tuple[str, str]] = {}
    for name, pairs in all_pairs.items():
        partition = membership[name]
        if not isinstance(partition, dict) or set(partition) != {"train", "test"}:
            raise ValueError(f"Invalid train/test partition for {name}")
        train_indices, test_indices = partition["train"], partition["test"]
        if not isinstance(train_indices, list) or not isinstance(test_indices, list):
            raise ValueError(f"Invalid split index lists for {name}")
        indices = train_indices + test_indices
        if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
            raise ValueError(f"Invalid split index type for {name}")
        if len(train_indices) != counts[name] or sorted(indices) != list(range(len(pairs))):
            raise ValueError(f"Overlapping, missing or invalid split indices for {name}")
        for split, selected in partition.items():
            result[split][name] = [pairs[index] for index in selected]
            for index in selected:
                record = inventory[name][index]
                digest = record["image_sha256"]
                prior = image_membership.get(digest)
                identity = f"{name}/{record['image']}"
                if prior is not None and prior[0] != split:
                    raise ValueError(f"Train/test image overlap: {prior[1]} and {identity}")
                image_membership[digest] = (split, identity)

    if not exists:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=manifest_path.parent, suffix=".tmp", delete=False) as stream:
                temporary_path = Path(stream.name)
                json.dump(manifest, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            os.replace(temporary_path, manifest_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
    return result


def _read_image(path: str, mode: str, suppress_opencv_warnings: bool) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert(mode)
    except (UnidentifiedImageError, OSError, ValueError) as pil_error:
        try:
            import cv2

            if suppress_opencv_warnings:
                # OpenCV 4.x: 0 = SILENT. This hides TIFF metadata warnings while
                # decode failures below still raise a Python exception.
                cv2.setLogLevel(0)

            flag = cv2.IMREAD_COLOR if mode == "RGB" else cv2.IMREAD_GRAYSCALE
            decoded = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)
            if decoded is None:
                raise ValueError("OpenCV could not decode this file")
            if mode == "RGB":
                decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            return Image.fromarray(decoded)
        except Exception as fallback_error:
            raise RuntimeError(f"Cannot read {path!r}; PIL: {pil_error}; fallback: {fallback_error}") from fallback_error


class PolypDataset(Dataset):
    """Square resize and explicit normalization shared by the two stages."""

    def __init__(
        self,
        pairs: Sequence[Pair],
        image_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        is_train: bool = False,
        horizontal_flip_prob: float = 0.0,
        vertical_flip_prob: float = 0.0,
        rotation90_prob: float = 0.0,
        mask_threshold: float = 0.5,
        suppress_opencv_warnings: bool = True,
    ):
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
            raise ValueError("mean/std must have three channels and std must be positive")
        if not 0 <= mask_threshold < 1:
            raise ValueError("mask_threshold must be in [0, 1)")
        for probability in (horizontal_flip_prob, vertical_flip_prob, rotation90_prob):
            if not 0 <= probability <= 1:
                raise ValueError("Augmentation probabilities must be between zero and one")
        self.pairs = list(pairs)
        self.image_size = image_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.is_train = is_train
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation90_prob = rotation90_prob
        self.mask_threshold = mask_threshold
        self.suppress_opencv_warnings = suppress_opencv_warnings

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.pairs[index]
        image = _read_image(image_path, "RGB", self.suppress_opencv_warnings)
        mask = _read_image(mask_path, "L", self.suppress_opencv_warnings)
        if image.size != mask.size:
            raise ValueError(f"Image/mask size mismatch for {image_path}: {image.size} versus {mask.size}")
        size = (self.image_size, self.image_size)
        image = image.resize(size, resample=Image.Resampling.BILINEAR)
        mask = mask.resize(size, resample=Image.Resampling.NEAREST)
        if self.is_train:
            if random.random() < self.horizontal_flip_prob:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < self.vertical_flip_prob:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if random.random() < self.rotation90_prob:
                operation = random.choice((Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270))
                image, mask = image.transpose(operation), mask.transpose(operation)
        image_tensor = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
        # Accept 0/1 binary masks; otherwise threshold on the conventional 0..255 scale.
        mask_array = np.array(mask, dtype=np.float32)
        if mask_array.max() > 1:
            mask_array /= 255.0
        mask_tensor = torch.from_numpy((mask_array > self.mask_threshold).astype(np.float32))
        return {"image": (image_tensor - self.mean) / self.std, "mask": mask_tensor, "name": Path(image_path).name}


def collate_fn(batch):
    """Stack tensor fields and preserve image names as a list."""
    return default_collate(batch)
