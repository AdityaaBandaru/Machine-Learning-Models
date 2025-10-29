"""Visualization helpers for Digit Insight Studio."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from ..data import MNIST_MEAN, MNIST_STD


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * MNIST_STD + MNIST_MEAN


def to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0, 1)
    array = (tensor.squeeze().numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def overlay_heatmap(image: torch.Tensor, heatmap: torch.Tensor, output_path: Path | None = None) -> Image.Image:
    base = denormalize(image).clamp(0, 1)
    base_img = base.squeeze().cpu().numpy()
    heat = heatmap.squeeze().cpu().numpy()

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.imshow(base_img, cmap="gray")
    ax.imshow(heat, cmap="jet", alpha=0.5)
    ax.axis("off")
    fig.tight_layout(pad=0)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def save_grid(samples: torch.Tensor, path: Path, nrow: int = 8) -> None:
    from torchvision.utils import make_grid

    grid = make_grid(samples, nrow=nrow, normalize=True, value_range=(0, 1))
    array = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


__all__ = ["overlay_heatmap", "to_pil", "save_grid", "denormalize"]
