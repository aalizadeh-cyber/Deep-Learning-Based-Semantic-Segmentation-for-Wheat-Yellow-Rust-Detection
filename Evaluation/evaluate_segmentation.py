#!/usr/bin/env python3
"""Evaluate binary wheat yellow-rust segmentation models on paired images/masks.

Metrics are computed independently for each image and then macro-averaged across
images, matching the per-image evaluation protocol described for this study.
Images are evaluated once. Input images are kept at native resolution; only
reflection padding to a multiple of 16 is applied and removed after inference.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageOps
import torch
import segmentation_models_pytorch as smp

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
MODELS = {
    'unet_mobilenetv2': ('unet', 'mobilenet_v2', 'U-Net (MobileNetV2)'),
    'unet_resnet50': ('unet', 'resnet50', 'U-Net (ResNet50)'),
    'deeplabv3plus_efficientnet_b0': ('deeplabv3plus', 'efficientnet-b0', 'DeepLabV3+ (EfficientNet-B0)'),
    'deeplabv3plus_resnet101': ('deeplabv3plus', 'resnet101', 'DeepLabV3+ (ResNet101)'),
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', required=True, help='Folder containing evaluation images')
    p.add_argument('--masks', required=True, help='Folder containing binary ground-truth masks with matching filename stems')
    p.add_argument('--model', required=True, choices=MODELS, help='Model architecture key')
    p.add_argument('--weights', required=True, help='Path to the final-epoch .pth state_dict')
    p.add_argument('--output-dir', default='evaluation_results', help='Directory for CSV/JSON outputs')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--threshold', type=float, default=0.5, help='Sigmoid probability threshold')
    p.add_argument('--padding-multiple', type=int, default=16)
    return p.parse_args()


def load_model(model_key: str, weights_path: Path, device: torch.device):
    arch, encoder, _ = MODELS[model_key]
    constructor = smp.Unet if arch == 'unet' else smp.DeepLabV3Plus
    model = constructor(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases
        state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def image_to_tensor(image_path: Path, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    with Image.open(image_path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        arr = np.asarray(im, dtype=np.uint8)
    h, w = arr.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        # Pad only bottom/right, reflecting edge pixels. For one-pixel dimensions,
        # NumPy's reflect mode is invalid, so use edge padding in that rare case.
        mode = 'reflect' if h > 1 and w > 1 else 'edge'
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    x = arr.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)
    return x, (h, w)


def read_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('L')
        arr = np.asarray(im)
    return (arr > 127).astype(np.uint8)


def calculate_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool); gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    # Match the epsilon-based convention used in the original evaluation code:
    # if both masks are empty, Dice and IoU are 0 rather than 1.
    dice = (2.0 * tp) / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    f1_from_pr = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'pixel_accuracy': accuracy,
            'dice': dice, 'f1': f1_from_pr, 'iou': iou,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def main():
    args = parse_args()
    image_dir, mask_dir, weights = Path(args.images), Path(args.masks), Path(args.weights)
    if not image_dir.is_dir(): raise FileNotFoundError(f'Image folder not found: {image_dir}')
    if not mask_dir.is_dir(): raise FileNotFoundError(f'Mask folder not found: {mask_dir}')
    if not weights.is_file(): raise FileNotFoundError(f'Weights file not found: {weights}')
    if not 0 < args.threshold < 1: raise ValueError('--threshold must be between 0 and 1')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device))
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA was requested but is unavailable.')
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    masks = {p.stem.casefold(): p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    images = sorted((p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=lambda p: p.name.casefold())
    if not images: raise RuntimeError(f'No supported images found in {image_dir}')
    model = load_model(args.model, weights, device)
    rows: List[Dict] = []; skipped: List[Dict] = []
    print(f'Model: {MODELS[args.model][2]} | Device: {device} | Images found: {len(images)}')

    for image_path in images:
        mask_path = masks.get(image_path.stem.casefold())
        if mask_path is None:
            skipped.append({'image': image_path.name, 'reason': 'matching mask not found'})
            print(f'[SKIP] No matching mask for {image_path.name}')
            continue
        try:
            x, (h, w) = image_to_tensor(image_path, args.padding_multiple)
            gt = read_binary_mask(mask_path)
            if gt.shape != (h, w):
                skipped.append({'image': image_path.name, 'reason': f'mask size {gt.shape[::-1]} differs from image size {(w, h)}'})
                print(f'[SKIP] Size mismatch: {image_path.name} image={(w,h)} mask={gt.shape[::-1]}')
                continue
            x = x.to(device)
            with torch.inference_mode():
                logits = model(x)
                prob = torch.sigmoid(logits)[0, 0, :h, :w].cpu().numpy()
            pred = (prob > args.threshold).astype(np.uint8)
            metrics = calculate_metrics(pred, gt)
            rows.append({'image': image_path.name, 'mask': mask_path.name, 'width': w, 'height': h, **metrics})
            print(f"{image_path.name}: Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        except Exception as exc:
            skipped.append({'image': image_path.name, 'reason': f'{type(exc).__name__}: {exc}'})
            print(f'[ERROR] {image_path.name}: {exc}')

    if not rows: raise RuntimeError('No images were evaluated successfully. Check filename pairing, mask sizes, and paths.')
    per_image_path = out_dir / f'{args.model}_per_image.csv'
    with per_image_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    metric_names = ['precision', 'recall', 'pixel_accuracy', 'dice', 'f1', 'iou']
    summary = {'model_key': args.model, 'model_name': MODELS[args.model][2], 'weights': str(weights),
               'device': str(device), 'threshold': args.threshold, 'evaluated_images': len(rows),
               'skipped_images': len(skipped), 'aggregation': 'arithmetic mean of per-image metrics (macro average)',
               'metrics': {name: float(np.mean([r[name] for r in rows])) for name in metric_names},
               'metric_std': {name: float(np.std([r[name] for r in rows], ddof=1)) if len(rows) > 1 else 0.0 for name in metric_names},
               'skipped': skipped}
    summary_path = out_dir / f'{args.model}_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nMACRO-AVERAGED METRICS')
    for k, v in summary['metrics'].items(): print(f'{k:16s}: {v:.6f}')
    print(f'Per-image CSV: {per_image_path}\nSummary JSON: {summary_path}')
    if skipped: print(f'Warning: {len(skipped)} image(s) were skipped; inspect the JSON report before reporting results.')

if __name__ == '__main__': main()
