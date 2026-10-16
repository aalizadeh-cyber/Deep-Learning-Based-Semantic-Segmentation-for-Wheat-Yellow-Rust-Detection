#!/usr/bin/env python3
"""
Preprocess a raw COCO instance-segmentation dataset for wheat yellow-rust
segmentation.

Pipeline:
1) Read original images and COCO annotations.
2) Split ORIGINAL images into train/valid/test before tiling (default counts:
   200 / 44 / 44; change these with command-line arguments if needed).
3) Resize each image to 1024 x 1024 and resize its binary masks with nearest
   neighbour interpolation.
4) Tile each resized image into a 2 x 2 grid of 512 x 512 tiles.
5) Apply the dataset-level "at least 90% annotated images" rule separately
   within each split: keep all annotated tiles and, if needed, retain a
   reproducibly selected subset of empty tiles so >=90% of retained tiles
   contain annotations.
6) Write a COCO JSON file for each split. Cropped masks are stored as COCO RLE.

Important:
- This script is for RAW images + a COCO annotation JSON, not an already
  tiled/augmented Roboflow export.
- The split is made before tiling to prevent tiles from the same source image
  leaking into multiple subsets.
- Augmentation is performed separately by augment_wheat_rust_coco.py after
  preprocessing. This script writes the preprocessed base tiles only.
- The augmentation script can be run on the train, valid, and test splits to
  match the split counts reported in the study. The independent Local/Internet
  evaluation images are not passed through this tiling/augmentation pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as mask_utils
except ImportError as exc:
    raise SystemExit(
        "Missing dependency pycocotools. Install with: pip install pycocotools"
    ) from exc


# ---------------------------- Optional Roboflow download ---------------------

def optionally_download_roboflow(args: argparse.Namespace) -> tuple[Path, Path]:
    """
    Optional convenience download. For this preprocessing pipeline, download
    an ORIGINAL/raw COCO dataset. Do not point this script at a version that
    already has tiling/augmentation applied.
    """
    if not args.download_roboflow:
        if not args.images_dir or not args.annotations:
            raise SystemExit(
                "Provide --images-dir and --annotations, or use --download-roboflow."
            )
        return Path(args.images_dir), Path(args.annotations)

    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "Set ROBOFLOW_API_KEY in your environment before using --download-roboflow."
        )
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise SystemExit("Install the optional downloader with: pip install roboflow") from exc

    if not args.workspace or not args.project or args.version is None:
        raise SystemExit(
            "--workspace, --project, and --version are required with --download-roboflow."
        )

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("coco", location=str(args.download_dir))
    root = Path(dataset.location)
    # Raw COCO exports commonly use one split directory. Require an explicit
    # choice if the downloaded export does not have one unambiguous annotation.
    candidates = list(root.rglob("_annotations.coco.json"))
    if len(candidates) != 1:
        raise SystemExit(
            f"Expected one COCO annotation JSON in {root}, found {len(candidates)}. "
            "Download/prepare a single raw COCO export and pass its paths explicitly."
        )
    ann_path = candidates[0]
    return ann_path.parent, ann_path


# --------------------------------- COCO helpers ------------------------------

def load_coco_data(annotation_path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    images = {int(item["id"]): item for item in data.get("images", [])}
    if not images:
        raise ValueError(f"No images found in COCO JSON: {annotation_path}")
    return data, images


def segmentation_to_mask(
    annotation: dict[str, Any],
    image_height: int,
    image_width: int,
    coco: COCO,
) -> np.ndarray:
    """Decode one annotation to a binary uint8 mask."""
    mask = coco.annToMask(annotation)
    if mask.shape != (image_height, image_width):
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8)).resize(
                (image_width, image_height), resample=Image.Resampling.NEAREST
            )
        )
    return (mask > 0).astype(np.uint8)


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    """Encode a 2-D binary mask as JSON-serializable COCO RLE."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}


def image_file_for_record(images_dir: Path, record: dict[str, Any]) -> Path:
    file_name = Path(record["file_name"])
    candidate = images_dir / file_name
    if candidate.exists():
        return candidate
    # Some COCO exports store paths relative to the dataset root.
    candidate = images_dir / file_name.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find image '{record['file_name']}' under {images_dir}"
    )


def resize_image_and_mask(
    image: Image.Image, mask: np.ndarray, size: int
) -> tuple[Image.Image, np.ndarray]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((size, size), resample=Image.Resampling.BILINEAR)
    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_img = mask_img.resize((size, size), resample=Image.Resampling.NEAREST)
    return image, np.asarray(mask_img, dtype=np.uint8)


def get_split_ids(
    image_ids: list[int],
    train_count: int,
    valid_count: int,
    test_count: int,
    seed: int,
) -> dict[str, list[int]]:
    total_requested = train_count + valid_count + test_count
    if total_requested != len(image_ids):
        raise ValueError(
            f"Requested split counts sum to {total_requested}, but COCO contains "
            f"{len(image_ids)} images. Set --train-count/--valid-count/--test-count "
            "to match the source dataset."
        )
    ids = list(image_ids)
    random.Random(seed).shuffle(ids)
    return {
        "train": ids[:train_count],
        "valid": ids[train_count:train_count + valid_count],
        "test": ids[train_count + valid_count:],
    }


# ------------------------------- Main pipeline ------------------------------

def process_split(
    split_name: str,
    source_ids: list[int],
    images_by_id: dict[int, dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    source_coco: COCO,
    images_dir: Path,
    output_root: Path,
    resize_size: int,
    tile_size: int,
    min_annotated_fraction: float,
    seed: int,
) -> dict[str, int]:
    split_root = output_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)
    tile_records: list[dict[str, Any]] = []
    tile_annotations: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1

    # Generate all tiles first, then enforce the >=90% annotation rule over
    # this split's tile set.
    for source_id in source_ids:
        record = images_by_id[source_id]
        src_path = image_file_for_record(images_dir, record)
        with Image.open(src_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")

        original_h, original_w = image.height, image.width
        source_anns = annotations_by_image.get(source_id, [])
        full_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        for ann in source_anns:
            full_mask = np.maximum(
                full_mask,
                segmentation_to_mask(ann, original_h, original_w, source_coco),
            )

        resized_image, resized_mask = resize_image_and_mask(image, full_mask, resize_size)
        if resize_size != tile_size * 2:
            raise ValueError(
                f"This implementation expects a 2x2 grid: resize_size must equal "
                f"2 * tile_size; got {resize_size} and {tile_size}."
            )

        # Each tile gets its own binary foreground mask. If the source COCO
        # has multiple categories, masks are reconstructed per annotation below
        # so category IDs are retained in the output.
        for row in range(2):
            for col in range(2):
                x0, y0 = col * tile_size, row * tile_size
                x1, y1 = x0 + tile_size, y0 + tile_size
                tile_image = resized_image.crop((x0, y0, x1, y1))
                tile_mask = resized_mask[y0:y1, x0:x1]
                tile_name = f"{split_name}_{source_id}_{row}_{col}.jpg"
                tile_path = split_root / tile_name
                tile_image.save(tile_path, quality=95)

                this_image_id = next_image_id
                next_image_id += 1
                tile_records.append({
                    "id": this_image_id,
                    "file_name": tile_name,
                    "width": tile_size,
                    "height": tile_size,
                    "_has_foreground": bool(np.any(tile_mask)),
                    "_source_image_id": source_id,
                    "_x0": x0,
                    "_y0": y0,
                })

                # Crop each source annotation independently, preserving category.
                for ann in source_anns:
                    ann_mask = segmentation_to_mask(ann, original_h, original_w, source_coco)
                    # Resize each instance mask with nearest-neighbour, then crop tile.
                    ann_mask_resized = np.asarray(
                        Image.fromarray(ann_mask).resize(
                            (resize_size, resize_size),
                            resample=Image.Resampling.NEAREST,
                        ),
                        dtype=np.uint8,
                    )
                    crop = ann_mask_resized[y0:y1, x0:x1]
                    if not np.any(crop):
                        continue
                    bbox = mask_utils.toBbox(np.asfortranarray(crop.astype(np.uint8))).tolist()
                    area = int(crop.sum())
                    tile_annotations.append({
                        "id": next_annotation_id,
                        "image_id": this_image_id,
                        "category_id": int(ann["category_id"]),
                        "segmentation": encode_binary_mask(crop),
                        "area": area,
                        "bbox": [float(v) for v in bbox],
                        "iscrowd": 0,
                    })
                    next_annotation_id += 1

    annotated_tiles = [r for r in tile_records if r["_has_foreground"]]
    empty_tiles = [r for r in tile_records if not r["_has_foreground"]]
    if not annotated_tiles:
        raise ValueError(
            f"No annotated tiles were generated for split '{split_name}'. "
            "Check that the source COCO annotations and category IDs are correct."
        )

    # Require annotated_tiles / retained_tiles >= min_annotated_fraction.
    # Keep all annotated tiles; keep only as many empty tiles as this ratio allows.
    max_empty = math.floor(
        len(annotated_tiles) * (1.0 - min_annotated_fraction) / min_annotated_fraction
    )
    rng = random.Random(seed + {"train": 11, "valid": 23, "test": 37}[split_name])
    rng.shuffle(empty_tiles)
    retained_empty = empty_tiles[:max_empty]
    retained_ids = {r["id"] for r in annotated_tiles + retained_empty}

    # Delete image files for filtered-out tiles.
    for record in tile_records:
        if record["id"] not in retained_ids:
            try:
                (split_root / record["file_name"]).unlink()
            except FileNotFoundError:
                pass

    final_images = []
    id_remap: dict[int, int] = {}
    for new_id, record in enumerate(
        [r for r in tile_records if r["id"] in retained_ids], start=1
    ):
        old_id = record["id"]
        id_remap[old_id] = new_id
        final_images.append({
            "id": new_id,
            "file_name": record["file_name"],
            "width": record["width"],
            "height": record["height"],
        })

    final_annotations = []
    next_ann_id = 1
    for ann in tile_annotations:
        if ann["image_id"] not in retained_ids:
            continue
        item = dict(ann)
        item["id"] = next_ann_id
        item["image_id"] = id_remap[ann["image_id"]]
        final_annotations.append(item)
        next_ann_id += 1

    output_json = {
        "info": {
            "description": "Wheat yellow-rust semantic segmentation tiles",
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [],
        "images": final_images,
        "annotations": final_annotations,
        "categories": [
            {"id": int(c["id"]), "name": str(c["name"]), "supercategory": str(c.get("supercategory", ""))}
            for c in source_coco.dataset.get("categories", [])
        ],
    }
    with (split_root / "_annotations.coco.json").open("w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False)

    annotated_final = sum(1 for r in annotated_tiles + retained_empty if r["_has_foreground"])
    total_final = len(final_images)
    return {
        "source_images": len(source_ids),
        "tiles_before_filter": len(tile_records),
        "annotated_tiles": len(annotated_tiles),
        "empty_tiles_available": len(empty_tiles),
        "empty_tiles_retained": len(retained_empty),
        "tiles_after_filter": total_final,
        "annotation_fraction_percent": round(100.0 * annotated_final / total_final, 2),
        "annotations_written": len(final_annotations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess raw COCO images: split -> resize -> tile -> 90% annotation filter."
    )
    parser.add_argument("--images-dir", help="Directory containing original images.")
    parser.add_argument("--annotations", help="Path to the original COCO JSON.")
    parser.add_argument("--output-dir", default="processed_wheat_rust_dataset",
                        help="Output folder (default: processed_wheat_rust_dataset).")
    parser.add_argument("--resize-size", type=int, default=1024)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--min-annotated-fraction", type=float, default=0.90,
                        help="Minimum fraction of retained tiles with annotations (default 0.90).")
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--valid-count", type=int, default=44)
    parser.add_argument("--test-count", type=int, default=44)
    parser.add_argument("--seed", type=int, default=42)

    # Optional Roboflow download. Keep credentials out of source code.
    parser.add_argument("--download-roboflow", action="store_true")
    parser.add_argument("--workspace", default="YOUR_WORKSPACE")
    parser.add_argument("--project", default="YOUR_PROJECT")
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--download-dir", default="raw_coco_download")

    args = parser.parse_args()
    if not 0 < args.min_annotated_fraction <= 1:
        raise SystemExit("--min-annotated-fraction must be in (0, 1].")
    if args.resize_size != 2 * args.tile_size:
        raise SystemExit("--resize-size must equal 2 * --tile-size for 2x2 tiling.")

    images_dir, annotation_path = optionally_download_roboflow(args)
    if not images_dir.exists():
        raise SystemExit(f"Image directory does not exist: {images_dir}")
    if not annotation_path.exists():
        raise SystemExit(f"COCO annotation file does not exist: {annotation_path}")

    source_data, images_by_id = load_coco_data(annotation_path)
    source_coco = COCO(str(annotation_path))
    image_ids = sorted(images_by_id)
    split_ids = get_split_ids(
        image_ids, args.train_count, args.valid_count, args.test_count, args.seed
    )

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in source_data.get("annotations", []):
        annotations_by_image[int(ann["image_id"])].append(ann)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "input_images": len(image_ids),
        "split_counts_before_preprocessing": {
            name: len(ids) for name, ids in split_ids.items()
        },
        "settings": {
            "resize": [args.resize_size, args.resize_size],
            "tiling": f"2x2 tiles of {args.tile_size}x{args.tile_size}",
            "minimum_annotated_fraction": args.min_annotated_fraction,
            "augmentation_max_multiplier": 3,
            "seed": args.seed,
        },
        "splits": {},
    }

    for split_name in ("train", "valid", "test"):
        stats = process_split(
            split_name=split_name,
            source_ids=split_ids[split_name],
            images_by_id=images_by_id,
            annotations_by_image=annotations_by_image,
            source_coco=source_coco,
            images_dir=images_dir,
            output_root=output_root,
            resize_size=args.resize_size,
            tile_size=args.tile_size,
            min_annotated_fraction=args.min_annotated_fraction,
            seed=args.seed,
        )
        summary["splits"][split_name] = stats
        print(f"\n[{split_name.upper()}]")
        for key, value in stats.items():
            print(f"{key}: {value}")

    with (output_root / "preprocessing_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nFinished. Dataset written to: {output_root.resolve()}")
    print(f"Summary written to: {(output_root / 'preprocessing_summary.json').resolve()}")


if __name__ == "__main__":
    main()
