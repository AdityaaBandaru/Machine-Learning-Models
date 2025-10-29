"""Grad-CAM helper for the digit classifier."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.classifier import DigitClassifier


def compute_gradcam(model: DigitClassifier, image: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
    """Return a Grad-CAM heatmap for a single image tensor (1x1x28x28)."""
    model.eval()
    image = image.clone().requires_grad_(True)
    logits, features = model.forward_with_features(image)
    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    score = logits[:, target_class]
    gradients = torch.autograd.grad(score, features, retain_graph=False, create_graph=False)[0]
    activations = features.detach()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    return cam.squeeze().cpu()


__all__ = ["compute_gradcam"]
