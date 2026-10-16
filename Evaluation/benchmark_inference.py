#!/usr/bin/env python3
"""Benchmark single-image inference latency for one segmentation model."""
from __future__ import annotations
import argparse, csv, time, platform
from pathlib import Path
import torch
from evaluate_segmentation import MODELS, image_to_tensor, load_model, IMAGE_EXTS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',required=True); p.add_argument('--model',required=True,choices=MODELS); p.add_argument('--weights',required=True)
    p.add_argument('--output-csv',default='inference_timing.csv'); p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--warmup',type=int,default=5); p.add_argument('--repeats',type=int,default=30); p.add_argument('--max-images',type=int,default=20)
    a=p.parse_args(); device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    image_dir=Path(a.images); images=sorted([x for x in image_dir.iterdir() if x.suffix.lower() in IMAGE_EXTS],key=lambda x:x.name.casefold())[:a.max_images]
    if not images: raise RuntimeError(f'No images in {image_dir}')
    model=load_model(a.model,Path(a.weights),device); rows=[]
    for path in images:
        x,(original_h, original_w)=image_to_tensor(path,16); x=x.to(device)
        with torch.inference_mode():
            for _ in range(a.warmup):
                _=model(x)
            if device.type=='cuda': torch.cuda.synchronize()
            times=[]
            for _ in range(a.repeats):
                if device.type=='cuda': torch.cuda.synchronize()
                t0=time.perf_counter(); _=model(x)
                if device.type=='cuda': torch.cuda.synchronize()
                times.append((time.perf_counter()-t0)*1000)
        rows.append({'image':path.name,'model_key':a.model,'device':a.device,'original_height':int(original_h),'original_width':int(original_w),'padded_height':int(x.shape[-2]),'padded_width':int(x.shape[-1]),'repeats':a.repeats,'mean_ms':sum(times)/len(times),'median_ms':sorted(times)[len(times)//2],'std_ms':float(torch.tensor(times).std(unbiased=False))})
        print(f"{path.name}: mean={rows[-1]['mean_ms']:.3f} ms, median={rows[-1]['median_ms']:.3f} ms")
    out=Path(a.output_csv); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    print(f'Environment: Python {platform.python_version()}, PyTorch {torch.__version__}, device={device}')
    print(f'Saved {out}. Compare timing only when hardware, warm-up, repeat count, and image sizes are held constant.')
if __name__=='__main__': main()
