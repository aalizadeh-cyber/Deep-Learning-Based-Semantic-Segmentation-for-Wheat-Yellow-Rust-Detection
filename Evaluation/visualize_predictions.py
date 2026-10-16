#!/usr/bin/env python3
"""Save qualitative segmentation panels (image, ground truth, prediction, overlay)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import torch
from evaluate_segmentation import MODELS, image_to_tensor, load_model, read_binary_mask

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', required=True); p.add_argument('--masks', required=True)
    p.add_argument('--model', required=True, choices=MODELS); p.add_argument('--weights', required=True)
    p.add_argument('--output-dir', default='qualitative_results'); p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    p.add_argument('--threshold', type=float, default=0.5); p.add_argument('--limit', type=int, default=20)
    args = p.parse_args()
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device))
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    image_dir, mask_dir, out = Path(args.images), Path(args.masks), Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    masks = {x.stem.casefold(): x for x in mask_dir.iterdir() if x.suffix.lower() in EXTS}
    images = sorted([x for x in image_dir.iterdir() if x.suffix.lower() in EXTS], key=lambda x:x.name.casefold())[:args.limit]
    model = load_model(args.model, Path(args.weights), device)
    for path in images:
        mask_path = masks.get(path.stem.casefold())
        if mask_path is None: print(f'[SKIP] no mask: {path.name}'); continue
        with Image.open(path) as im: original = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
        h,w = original.shape[:2]; gt = read_binary_mask(mask_path)
        if gt.shape != (h,w): print(f'[SKIP] size mismatch: {path.name}'); continue
        x,_ = image_to_tensor(path,16)
        with torch.inference_mode():
            prob = torch.sigmoid(model(x.to(device)))[0,0,:h,:w].cpu().numpy()
        pred = prob > args.threshold
        overlay = original.copy().astype(np.float32)
        overlay[pred] = 0.45*overlay[pred] + 0.55*np.array([255, 220, 0], dtype=np.float32)
        # Compose a single four-panel PNG at native spatial dimensions.
        panels = [original, np.repeat(gt[...,None]*255,3,axis=2), np.repeat(pred[...,None]*255,3,axis=2), np.clip(overlay,0,255).astype(np.uint8)]
        labels = ['Original image','Ground-truth mask','Predicted mask','Overlay (yellow = predicted rust)']
        panel_w = min(w, 1000); scale = panel_w / w; panel_h = max(1,int(h*scale))
        canvas = Image.new('RGB',(panel_w*2,(panel_h+34)*2),'white')
        draw = ImageDraw.Draw(canvas)
        for i,(arr,label) in enumerate(zip(panels,labels)):
            im = Image.fromarray(arr).resize((panel_w,panel_h),Image.Resampling.NEAREST if i in (1,2) else Image.Resampling.LANCZOS)
            xx=(i%2)*panel_w; yy=(i//2)*(panel_h+34)
            canvas.paste(im,(xx,yy+34)); draw.text((xx+8,yy+10),label,fill='black')
        target = out / f'{path.stem}_{args.model}_qualitative.png'; canvas.save(target)
        print(f'Saved {target}')

if __name__ == '__main__': main()
