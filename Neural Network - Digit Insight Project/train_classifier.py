"""Train the digit classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import yaml

from src.data import DataConfig, dataconfig_from_dict, load_mnist
from src.models.classifier import classifier_from_dict


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return running_loss / total, correct / total


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Digit Insight classifier.")
    parser.add_argument("--config", type=Path, default=Path("config_classifier.yaml"))
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_cfg = dataconfig_from_dict(config.get("data", {}))
    device = resolve_device(config.get("training", {}).get("device", "auto"))
    seed = config.get("training", {}).get("seed", 1337)
    set_seed(seed)

    train_loader, val_loader, test_loader = load_mnist(data_cfg)
    model = classifier_from_dict(config.get("model", {})).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("training", {}).get("lr", 1e-3),
        weight_decay=config.get("training", {}).get("weight_decay", 1e-4),
    )

    epochs = config.get("training", {}).get("epochs", 5)
    best_val = 0.0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f}"
        )
        if val_acc > best_val:
            best_val = val_acc
            output_path = Path(config.get("training", {}).get("model_output", "checkpoints/classifier.pth"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": config}, output_path)

    metrics_path = Path(config.get("logging", {}).get("save_metrics", "results/classifier_metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    # Final evaluation on test set
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")

    history.append({"split": "test", "loss": test_loss, "acc": test_acc})
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


if __name__ == "__main__":
    main()
