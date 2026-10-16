# Deep Learning-Based Semantic Segmentation for Wheat Yellow Rust Detection

This repository organizes the preprocessing, augmentation, model training, and evaluation scripts associated with the study comparing U-Net and DeepLabV3+ architectures for wheat yellow-rust segmentation.

## Repository structure

- `Preprocessing/`: source-image splitting, EXIF orientation correction, resizing, 2×2 tiling, COCO mask generation, and separate offline augmentation.
- `Training Models/`: four model-training scripts for U-Net (MobileNetV2 and ResNet50) and DeepLabV3+ (EfficientNet-B0 and ResNet101).
- `Evaluation/`: pixel-level segmentation metrics, qualitative prediction panels, and CPU/GPU inference latency benchmarking.

## Data and credentials

The dataset and trained model weights are not included. Provide the appropriate dataset and checkpoint paths when running scripts. Never commit API keys, credentials, private data, or files for which redistribution is not authorized. The optional Roboflow download in the preprocessing script reads `ROBOFLOW_API_KEY` from an environment variable.

## Important protocol notes

- Preprocessing and offline augmentation are separate from the training scripts.
- The four training scripts use a fixed learning rate, train for 50 epochs by default, and save the final-epoch state dictionary.
- Evaluation reports per-image metrics and their arithmetic mean. The evaluation threshold defaults to 0.5, using a strict `probability > threshold` decision.
- For a comparison corresponding to a reported experiment, use its saved weights, exact image/mask set, and the same input protocol described in the manuscript.

See each subfolder's script help (`python script.py --help`) for arguments and examples.
