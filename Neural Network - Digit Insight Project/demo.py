"""Interactive Digit Insight Studio demo using Gradio."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import MNIST_MEAN, MNIST_STD
from src.models.classifier import classifier_from_dict, DigitClassifier
from src.models.vae import vae_from_dict, DigitVAE
from src.utils.gradcam import compute_gradcam
from src.utils.visualization import overlay_heatmap
from src.utils.segmentation import extract_digit_arrays


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: Path, build_fn):
    state = torch.load(checkpoint, map_location="cpu")
    config = state.get("config", {})
    model = build_fn(config.get("model", {}))
    payload = state["model"] if "model" in state else state
    model.load_state_dict(payload)
    return model, config


def _extract_array(image: np.ndarray | Image.Image | dict) -> np.ndarray:
    if isinstance(image, dict):
        for key in ("image", "composite", "mask"):
            if key in image and image[key] is not None:
                return _extract_array(image[key])
        raise ValueError("Unsupported sketchpad payload: missing image content")
    if isinstance(image, Image.Image):
        return np.array(image)
    return np.array(image)


def _to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    pil_img = Image.fromarray(array).convert("L")
    image_np = np.array(pil_img)
    tensor = torch.from_numpy(image_np).float() / 255.0
    tensor = 1.0 - tensor
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = (tensor - MNIST_MEAN) / MNIST_STD
    return tensor.to(device)


def preprocess_digits(image: np.ndarray | Image.Image | dict, device: torch.device) -> tuple[List[torch.Tensor], List[np.ndarray]]:
    array = _extract_array(image)
    if array.ndim == 3 and array.shape[2] >= 3:
        array = array[..., 0]
    array = np.clip(array, 0, 255).astype(np.uint8)
    digit_arrays = extract_digit_arrays(array)
    tensors = [_to_tensor(digit, device) for digit in digit_arrays]
    return tensors, digit_arrays


def generate_variations(vae: DigitVAE, image: torch.Tensor, num_samples: int = 6) -> List[Image.Image]:
    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(image)
    samples = []
    for _ in range(num_samples):
        z = vae.reparameterize(mu, logvar)
        with torch.no_grad():
            recon = vae.decode(z).cpu()
        array = (recon.squeeze().numpy() * 255).astype(np.uint8)
        samples.append(Image.fromarray(array, mode="L"))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Digit Insight Studio demo.")
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    args = parser.parse_args()

    device = resolve_device()
    classifier, cls_config = load_model(args.classifier, classifier_from_dict)
    vae, vae_config = load_model(args.vae, vae_from_dict)
    classifier.to(device)
    vae.to(device)
    classifier.eval()
    vae.eval()

    def inference(image: np.ndarray):
        digit_tensors, digit_arrays = preprocess_digits(image, device)
        if not digit_tensors:
            return "No digits detected. Please draw a number.", [], []

        batch = torch.cat(digit_tensors, dim=0)
        with torch.no_grad():
            logits = classifier(batch)
            probabilities = F.softmax(logits, dim=1).cpu()
        preds = probabilities.argmax(dim=1).tolist()

        predicted_number = "".join(str(p) for p in preds)
        details = [
            f"{idx + 1}: {pred} (confidence {probabilities[idx, pred]:.2f})"
            for idx, pred in enumerate(preds)
        ]
        summary = "Predicted number: " + predicted_number + "\n" + "\n".join(details)

        overlay_entries: List[tuple[Image.Image, str]] = []
        variation_entries: List[tuple[Image.Image, str]] = []

        for idx, (tensor_digit, pred, digit_array) in enumerate(zip(digit_tensors, preds, digit_arrays), start=1):
            with torch.enable_grad():
                heatmap = compute_gradcam(classifier, tensor_digit.clone(), target_class=pred)
            overlay_img = overlay_heatmap(tensor_digit.cpu(), heatmap)
            overlay_entries.append((overlay_img, f"Digit {idx}: {pred}"))

            variations = generate_variations(vae, tensor_digit, num_samples=3)
            for j, var_img in enumerate(variations, start=1):
                variation_entries.append((var_img, f"Digit {idx} sample {j}"))

        return summary, overlay_entries, variation_entries

    iface = gr.Interface(
        fn=inference,
        inputs=gr.Sketchpad(image_mode="L", width=280, height=280),
        outputs=[
            gr.Textbox(label="Predicted number"),
            gr.Gallery(label="Grad-CAM overlays", columns=3, height="auto"),
            gr.Gallery(label="Creative variations", columns=3, height="auto"),
        ],
        title="Digit Insight Studio",
        description="Draw any number sequence to inspect predictions, saliency, and creative variations.",
    )
    iface.launch()


if __name__ == "__main__":
    main()
