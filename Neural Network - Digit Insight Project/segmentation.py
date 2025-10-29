"""Digit segmentation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class SegmentConfig:
    threshold: int = 200
    min_pixels: int = 40
    padding_ratio: float = 0.25
    target_size: int = 28
    min_density: float = 0.15
    min_box_side: int = 18


def _binarize(image: np.ndarray, threshold: int) -> np.ndarray:
    return (image < threshold).astype(np.uint8)


def extract_digit_arrays(image: np.ndarray, config: SegmentConfig | None = None) -> List[np.ndarray]:
    """Extract individual digit crops as 28x28 grayscale arrays sorted left-to-right."""
    cfg = config or SegmentConfig()
    if image.ndim == 3:
        image = image[..., 0]
    image = image.astype(np.uint8)

    binary = _binarize(image, cfg.threshold)
    binary = ndimage.binary_closing(binary, structure=np.ones((3, 3)))
    binary = ndimage.binary_fill_holes(binary).astype(np.uint8)
    labeled, num_features = ndimage.label(binary)
    if num_features == 0:
        return []

    objects = ndimage.find_objects(labeled)
    digits: List[Tuple[float, np.ndarray]] = []
    for idx, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        y_slice, x_slice = slc
        region_mask = (labeled[slc] == idx)
        height = y_slice.stop - y_slice.start
        width = x_slice.stop - x_slice.start
        if max(height, width) < cfg.min_box_side:
            continue
        pixel_count = int(region_mask.sum())
        bbox_area = height * width
        density = pixel_count / max(bbox_area, 1)
        if pixel_count < cfg.min_pixels or density < cfg.min_density:
            continue
        crop = image[y_slice, x_slice]
        # apply mask to keep only foreground, set background to white
        crop = np.where(region_mask, crop, 255)

        h, w = crop.shape
        pad = int(max(h, w) * cfg.padding_ratio)
        crop = np.pad(crop, ((pad, pad), (pad, pad)), mode="constant", constant_values=255)

        h, w = crop.shape
        if h > w:
            diff = h - w
            left = diff // 2
            right = diff - left
            crop = np.pad(crop, ((0, 0), (left, right)), mode="constant", constant_values=255)
        elif w > h:
            diff = w - h
            top = diff // 2
            bottom = diff - top
            crop = np.pad(crop, ((top, bottom), (0, 0)), mode="constant", constant_values=255)

        pil = Image.fromarray(crop).resize((cfg.target_size, cfg.target_size), Image.BILINEAR)
        digits.append((float((x_slice.start + x_slice.stop) / 2), np.array(pil, dtype=np.uint8)))

    digits.sort(key=lambda item: item[0])
    return [digit for _, digit in digits]


__all__ = ["SegmentConfig", "extract_digit_arrays"]
