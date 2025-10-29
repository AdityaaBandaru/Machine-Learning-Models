"""Evaluate classifier accuracy and log predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml

from src.data import dataconfig_from_dict, load_mnist
from src.models.classifier import classifier_from_dict
from src.utils.visualization import save_grid, denormalize


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the digit classifier.")
    parser.add_argument("--config", type=Path, default=Path("config_classifier.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = resolve_device(config.get("training", {}).get("device", "auto"))
    _, _, test_loader = load_mnist(dataconfig_from_dict(config.get("data", {})))
    model = classifier_from_dict(config.get("model", {}))
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    losses = 0.0
    sample_batch = None

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            losses += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            if sample_batch is None:
                sample_batch = (images[:16].cpu(), preds[:16].cpu())

    accuracy = correct / total
    avg_loss = losses / total
    print(f"Test loss: {avg_loss:.4f} | accuracy: {accuracy:.4f}")

    metrics_path = Path(config.get("logging", {}).get("save_metrics", "results/classifier_metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump({"test_loss": avg_loss, "test_accuracy": accuracy}, handle, indent=2)

    if sample_batch is not None:
        images, preds = sample_batch
        captioned = torch.stack([denormalize(img) for img in images])
        save_grid(captioned, Path("results/test_predictions.png"))
        print("Saved sample predictions to results/test_predictions.png")


if __name__ == "__main__":
    main()
