# Cat Breed Classifier (CNN)

A PyTorch image classifier that identifies a cat's breed from a photo — one of 6 breeds: American Shorthair, Bengal, Maine Coon, Ragdoll, Scottish Fold, or Sphynx.

## Dataset

`cat-breed/` — 1,500 images split into `TRAIN` (200 per breed) and `TEST` (50 per breed), organized as one folder per class (`ImageFolder`-compatible).

## Approach

The notebook builds and compares two models:

1. **A CNN trained from scratch** — two conv+pool blocks feeding into a small dense head, trained on 64x64 images with basic augmentation (flip, shear, zoom). A validation split (20% of TRAIN) is used to track loss/accuracy during training, keeping the TEST set untouched until final evaluation.
   - **Result: 44% test accuracy**

2. **Transfer learning** — a frozen, ImageNet-pretrained ResNet18 used as a feature extractor, with a small trainable classifier head (2-layer MLP with dropout) on top of its 512-dim features. Given the small dataset (~1,000 training images), leaning on features already learned from ImageNet's ~1.2M images generalizes far better than training convolutional filters from zero.
   - **Result: 92.3% test accuracy / macro F1**

## Reproducibility

Seeded per [PyTorch's randomness guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html) (`random`, NumPy, and PyTorch RNGs all seeded; cuDNN determinism enabled), and forced to run on CPU rather than GPU/MPS, since PyTorch doesn't guarantee bit-for-bit deterministic GPU ops the way it does for CPU ops.

## Running it

```bash
pip install torch torchvision numpy matplotlib scikit-learn certifi
jupyter notebook code.ipynb
```

Run all cells top to bottom — later cells (training, evaluation, transfer learning) depend on state set up in earlier ones. The transfer-learning section downloads pretrained ResNet18 weights on first run (cached afterward).

## Results at a glance

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| CNN (from scratch) | 44.0% | 0.44 |
| ResNet18 (frozen) + MLP head | 92.3% | 0.92 |
