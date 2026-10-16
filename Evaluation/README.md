# Evaluation scripts

These scripts evaluate trained binary segmentation models for wheat yellow-rust masks. They are designed to be kept separate from training and preprocessing.

## Scripts

- `evaluate_segmentation.py`: per-image precision, recall, pixel accuracy, Dice/F1, and IoU; writes per-image CSV and macro-average JSON. A matching ground-truth mask is required for each image.
- `visualize_predictions.py`: creates qualitative four-panel figures for selected image/mask pairs.
- `benchmark_inference.py`: measures single-image model-forward latency on CPU or CUDA, including warm-up and synchronization for CUDA timing.

## Install

```bash
pip install torch torchvision segmentation-models-pytorch pillow numpy
```

## Data layout

```text
evaluation_data/
├── images/
│   ├── sample_001.jpg
│   └── ...
└── masks/
    ├── sample_001.png
    └── ...
```

Image and mask filename stems must match. Ground-truth masks must be single-channel or convertible to grayscale, with rust pixels encoded above 127 and background at or below 127. Masks must have exactly the same native dimensions as the corresponding images; mismatches are skipped and reported rather than silently resized.

## Evaluate one model

```bash
python evaluate_segmentation.py \
  --images /path/to/evaluation_data/images \
  --masks /path/to/evaluation_data/masks \
  --model deeplabv3plus_resnet101 \
  --weights /path/to/final_model.pth \
  --output-dir results/external_resnet101 \
  --device auto
```

Available model keys:

- `unet_mobilenetv2`
- `unet_resnet50`
- `deeplabv3plus_efficientnet_b0`
- `deeplabv3plus_resnet101`

Repeat the command for each model and change `--model`, `--weights`, and `--output-dir` accordingly. The script uses the final-epoch state dictionary expected from the training scripts, RGB input, ImageNet normalization, a 0.5 sigmoid threshold, native-resolution evaluation, and reflection padding to a multiple of 16 that is removed before metric calculation.

## Qualitative figures

```bash
python visualize_predictions.py \
  --images /path/to/evaluation_data/images \
  --masks /path/to/evaluation_data/masks \
  --model deeplabv3plus_resnet101 \
  --weights /path/to/final_model.pth \
  --output-dir results/qualitative \
  --limit 20
```

## Inference latency

Run separately for each model and device. CPU and GPU times should not be compared unless the hardware, image sizes, warm-up, and repeat count are documented and controlled.

```bash
python benchmark_inference.py \
  --images /path/to/evaluation_data/images \
  --model deeplabv3plus_resnet101 \
  --weights /path/to/final_model.pth \
  --device cpu --output-csv results/time_cpu_resnet101.csv
```

Use `--device cuda` for GPU timing on a CUDA-enabled machine.

## Metric protocol and interpretation

Metrics are calculated for each image independently and then arithmetic-averaged across successfully evaluated images (macro average), rather than pooling pixels across the whole dataset. For precision and recall, a zero denominator returns 0. For Dice and IoU, the epsilon-based implementation returns 0 when both the ground-truth and prediction masks are empty, consistent with the original evaluation functions. The model threshold is 0.5 by default and can be changed using `--threshold`.

For results corresponding to a particular experiment, use that experiment's saved checkpoint, image list, ground-truth masks, 0.5 threshold, and preprocessing protocol. The scripts report skipped or mismatched image-mask pairs so that the input list can be checked transparently. The saved per-image TP, FP, FN, and TN values can also be summed if a pooled pixel-level confusion matrix is required; do not substitute image-level healthy/infected classification for pixel-level segmentation metrics.
