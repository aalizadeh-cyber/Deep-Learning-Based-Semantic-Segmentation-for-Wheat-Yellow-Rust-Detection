#!/usr/bin/env python3
"""
augmentation for a preprocessed COCO semantic-segmentation dataset.

Input structure:
  dataset/
    train/_annotations.coco.json + images
    valid/_annotations.coco.json + images
    test/_annotations.coco.json + images


For binary semantic segmentation, all foreground instance masks are merged
into one foreground mask for each augmented copy.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import albumentations as A
from pycocotools import mask as mask_utils


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}


def decode_segmentation(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        return (decoded > 0).astype(np.uint8)

    if isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((height, width), dtype=np.uint8)
        rles = mask_utils.frPyObjects(segmentation, height, width)
        decoded = mask_utils.decode(mask_utils.merge(rles))
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        return (decoded > 0).astype(np.uint8)

    return np.zeros((height, width), dtype=np.uint8)


def augment_split(split_dir: Path, multiplier: int, seed: int) -> dict[str, int]:
    annotation_path = split_dir / "_annotations.coco.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing COCO annotation file: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = list(coco.get("images", []))
    annotations = list(coco.get("annotations", []))
    if not images:
        raise ValueError(f"No images found in {annotation_path}")
    already_augmented = [
        im for im in images
        if Path(str(im.get("file_name", ""))).stem.rsplit("_aug", 1)[-1].isdigit()
        and "_aug" in Path(str(im.get("file_name", ""))).stem
    ]
    if already_augmented:
        raise ValueError(
            f"Found {len(already_augmented)} augmented images in {split_dir}. "
            "This script modifies the dataset in place; do not run it twice on the same output. "
            "Regenerate the preprocessed dataset first, then run augmentation once."
        )

    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        anns_by_image[int(ann["image_id"])].append(ann)

    categories = coco.get("categories", [])
    category_id = int(categories[0]["id"]) if categories else 1

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.05, contrast_limit=0.0, p=0.5
        ),
    ])

    rng = random.Random(seed)
    next_image_id = max(int(im["id"]) for im in images) + 1
    next_ann_id = max((int(a["id"]) for a in annotations), default=0) + 1
    augmented_images = []
    augmented_annotations = []

    for image_record in images:
        image_path = split_dir / image_record["file_name"]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        union_mask = np.zeros((height, width), dtype=np.uint8)
        for ann in anns_by_image.get(int(image_record["id"]), []):
            union_mask = np.maximum(
                union_mask,
                decode_segmentation(ann.get("segmentation"), height, width),
            )

        for variant in range(1, multiplier):
            # Make each variant reproducible without using a fixed transform.
            local_seed = rng.randint(0, 2**31 - 1)
            random.seed(local_seed)
            np.random.seed(local_seed)
            result = transform(image=image_rgb, mask=union_mask)
            out_image = result["image"]
            out_mask = (result["mask"] > 0).astype(np.uint8)

            stem = Path(image_record["file_name"]).stem
            suffix = Path(image_record["file_name"]).suffix or ".jpg"
            out_name = f"{stem}_aug{variant:02d}{suffix}"
            out_path = split_dir / out_name
            out_bgr = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
            if suffix.lower() in {".jpg", ".jpeg"}:
                cv2.imwrite(str(out_path), out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                cv2.imwrite(str(out_path), out_bgr)

            new_image_id = next_image_id
            next_image_id += 1
            augmented_images.append({
                "id": new_image_id,
                "file_name": out_name,
                "width": int(width),
                "height": int(height),
            })

            if np.any(out_mask):
                rle = encode_mask(out_mask)
                rle_for_bbox = {
                    "size": rle["size"],
                    "counts": rle["counts"].encode("ascii")
                    if isinstance(rle["counts"], str) else rle["counts"],
                }
                bbox = mask_utils.toBbox(rle_for_bbox).tolist()
                augmented_annotations.append({
                    "id": next_ann_id,
                    "image_id": new_image_id,
                    "category_id": category_id,
                    "segmentation": rle,
                    "area": int(out_mask.sum()),
                    "bbox": [float(v) for v in bbox],
                    "iscrowd": 0,
                })
                next_ann_id += 1

    coco["images"] = images + augmented_images
    coco["annotations"] = annotations + augmented_annotations
    with annotation_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    return {
        "base_images": len(images),
        "multiplier": multiplier,
        "augmented_copies_created": len(augmented_images),
        "total_images_after_augmentation": len(images) + len(augmented_images),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment preprocessed COCO segmentation splits.")
    parser.add_argument("--dataset-dir", required=True, help="Root directory containing train/, valid/, and test/.")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], choices=["train", "valid", "test"])
    parser.add_argument("--multiplier", type=int, default=3, help="Total samples per base image, including the original (default 3).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.multiplier < 1 or args.multiplier > 3:
        raise SystemExit("--multiplier must be between 1 and 3.")

    root = Path(args.dataset_dir)
    summary = {}
    for offset, split in enumerate(args.splits):
        stats = augment_split(root / split, args.multiplier, args.seed + offset)
        summary[split] = stats
        print(f"\n[{split.upper()}]")
        for key, value in stats.items():
            print(f"{key}: {value}")

    summary_path = root / "augmentation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary written to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
