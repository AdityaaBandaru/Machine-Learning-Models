"""Train the digit VAE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm
import yaml

from src.data import dataconfig_from_dict, load_mnist
from src.models.vae import vae_from_dict
from src.utils.visualization import save_grid


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the VAE component.")
    parser.add_argument("--config", type=Path, default=Path("config_vae.yaml"))
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_cfg = dataconfig_from_dict({
        "batch_size": config.get("training", {}).get("batch_size", 128),
        "num_workers": config.get("training", {}).get("num_workers", 2),
        "augment": False,
        "validation_split": 0.1,
    })
    device = resolve_device(config.get("training", {}).get("device", "auto"))
    seed = config.get("training", {}).get("seed", 2024)
    set_seed(seed)

    train_loader, val_loader, _ = load_mnist(data_cfg)
    model = vae_from_dict(config.get("model", {})).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get("training", {}).get("lr", 1e-3),
        weight_decay=config.get("training", {}).get("weight_decay", 0.0),
    )
    beta = config.get("training", {}).get("beta", 1.0)
    epochs = config.get("training", {}).get("epochs", 10)

    best_val = float("inf")
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        total = 0
        for images, _ in tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False):
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(images)
            loss = model.loss_fn(recon, images, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            total += images.size(0)
        train_loss /= total

        model.eval()
        val_loss = 0.0
        total_val = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                recon, mu, logvar = model(images)
                loss = model.loss_fn(recon, images, mu, logvar, beta=beta)
                val_loss += loss.item() * images.size(0)
                total_val += images.size(0)
        val_loss /= total_val

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            output_path = Path(config.get("training", {}).get("model_output", "checkpoints/vae.pth"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": config}, output_path)

            with torch.no_grad():
                z = torch.randn(64, config.get("model", {}).get("latent_dim", 16), device=device)
                samples = model.decode(z).cpu()
                sample_dir = Path(config.get("logging", {}).get("sample_dir", "results/vae_samples"))
                sample_dir.mkdir(parents=True, exist_ok=True)
                save_grid(samples, sample_dir / "samples.png")

    metrics_path = Path(config.get("logging", {}).get("save_metrics", "results/vae_metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


if __name__ == "__main__":
    main()
