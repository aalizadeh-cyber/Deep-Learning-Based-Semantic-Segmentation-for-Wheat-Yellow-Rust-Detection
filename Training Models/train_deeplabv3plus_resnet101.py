"""Train DeepLabV3+ (resnet101) for binary wheat yellow-rust segmentation.

The script assumes that image resizing, tiling, split assignment, and any augmentation have already been completed by separate data-preparation scripts.

Input layout: DATA_ROOT/{train,valid,test}/_annotations.coco.json plus the corresponding
image files in each split directory. Validation and test images are evaluated once each.
A fixed learning rate is used, and the final-epoch weights are saved and tested.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp


SEED = 42
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert an already-prepared RGB image to normalized CHW float tensor."""
    image = image.astype(np.float32) / 255.0
    mean = np.asarray(IMAGE_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(IMAGE_STD, dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


class CocoBinarySegmentationDataset(Dataset):
    """COCO instance annotations merged into a single binary rust mask."""

    def __init__(self, annotation_file: Path, image_dir: Path):
        if not annotation_file.is_file():
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_file}")
        if not image_dir.is_dir():
            raise NotADirectoryError(f"Image directory not found: {image_dir}")
        self.coco = COCO(str(annotation_file))
        self.image_dir = image_dir
        self.image_ids = sorted(self.coco.imgs.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[index]
        info = self.coco.loadImgs(image_id)[0]
        image_path = self.image_dir / info["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Image listed in COCO annotations is missing: {image_path}")

        with Image.open(image_path) as image_file:
            image = np.asarray(image_file.convert("RGB"))

        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        annotation_ids = self.coco.getAnnIds(imgIds=[image_id])
        for annotation in self.coco.loadAnns(annotation_ids):
            # Binary segmentation: any annotated rust region is foreground.
            mask = np.maximum(mask, self.coco.annToMask(annotation).astype(np.uint8))

        image_tensor = image_to_tensor(image)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return image_tensor, mask_tensor


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def calculate_binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Compute metrics per image, then macro-average images within the batch."""
    predictions = torch.sigmoid(logits) >= 0.5
    targets = targets >= 0.5
    eps = 1e-8
    per_image: dict[str, list[float]] = {
        key: [] for key in ("iou", "dice", "precision", "recall", "pixel_accuracy", "f1")
    }
    for pred, target in zip(predictions, targets):
        tp = torch.logical_and(pred, target).sum().item()
        fp = torch.logical_and(pred, ~target).sum().item()
        fn = torch.logical_and(~pred, target).sum().item()
        tn = torch.logical_and(~pred, ~target).sum().item()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        dice = (2 * tp) / (2 * tp + fp + fn + eps)
        per_image["iou"].append(tp / (tp + fp + fn + eps))
        per_image["dice"].append(dice)
        per_image["precision"].append(precision)
        per_image["recall"].append(recall)
        per_image["pixel_accuracy"].append((tp + tn) / (tp + tn + fp + fn + eps))
        per_image["f1"].append((2 * precision * recall) / (precision + recall + eps))
    return {key: float(np.mean(values)) for key, values in per_image.items()}


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    samples = 0
    totals = {key: 0.0 for key in ("iou", "dice", "precision", "recall", "pixel_accuracy", "f1")}

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, masks in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).float()

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
            loss = loss_fn(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

            batch_size_actual = int(images.shape[0])
            total_loss += float(loss.item()) * batch_size_actual
            batch_metrics = calculate_binary_metrics(logits.detach(), masks)
            for key, value in batch_metrics.items():
                totals[key] += value * batch_size_actual
            samples += batch_size_actual
            batches += 1

    if batches == 0:
        raise RuntimeError("DataLoader yielded no batches.")
    return {"loss": total_loss / samples, **{k: v / samples for k, v in totals.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing train/, valid/, and test/ subdirectories.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/deeplabv3plus_resnet101"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers cannot be negative.")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits: dict[str, tuple[Path, Path]] = {}
    for split in ("train", "valid", "test"):
        split_dir = args.data_root / split
        splits[split] = (split_dir / "_annotations.coco.json", split_dir)

    train_ds = CocoBinarySegmentationDataset(*splits["train"])
    valid_ds = CocoBinarySegmentationDataset(*splits["valid"])
    test_ds = CocoBinarySegmentationDataset(*splits["test"])

    train_loader = make_loader(train_ds, args.batch_size, True, args.workers)
    valid_loader = make_loader(valid_ds, args.batch_size, False, args.workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.workers)

    print(f"Samples — train: {len(train_ds)}, validation: {len(valid_ds)}, test: {len(test_ds)}")

    # Model DeepLabV3+ with Resnet 101 encoder 
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)
    loss_fn = smp.losses.DiceLoss(mode="binary")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, Any]] = []
    checkpoint_path = args.output_dir / "final_model_state_dict.pth"
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_metrics = run_epoch(model, valid_loader, loss_fn, device)

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f}, IoU={train_metrics['iou']:.4f} | "
            f"val loss={val_metrics['loss']:.4f}, IoU={val_metrics['iou']:.4f}"
        )


    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    # Save and evaluate the weights from the final training epoch.
    torch.save(model.state_dict(), checkpoint_path)
    test_metrics = run_epoch(model, test_loader, loss_fn, device)
    summary = {
        "model": "DeepLabV3+",
        "encoder": "resnet101",
        "final_validation_iou": history[-1]["validation"]["iou"],
        "test_metrics": test_metrics,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "initial_learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": str(device),
        "training_seconds": time.perf_counter() - start,
        "checkpoint": checkpoint_path.name,
    }
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
