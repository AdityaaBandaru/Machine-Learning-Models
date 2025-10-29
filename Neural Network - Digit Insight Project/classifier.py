"""CNN classifier for MNIST digits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ClassifierConfig:
    hidden_channels: Sequence[int] = (32, 64, 128)
    dropout: float = 0.3


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DigitClassifier(nn.Module):
    """Compact CNN with three conv stages and global pooling."""

    def __init__(self, config: ClassifierConfig) -> None:
        super().__init__()
        channels = [1, *config.hidden_channels]
        blocks = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            blocks.append(ConvBlock(in_ch, out_ch))
            blocks.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
            blocks.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        pooled = self.gap(feats)
        logits = self.classifier(pooled)
        return logits

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x)
        pooled = self.gap(feats)
        logits = self.classifier(pooled)
        return logits, feats


def classifier_from_dict(cfg: dict) -> DigitClassifier:
    config = ClassifierConfig(
        hidden_channels=tuple(cfg.get("hidden_channels", [32, 64, 128])),
        dropout=cfg.get("dropout", 0.3),
    )
    return DigitClassifier(config)


__all__ = ["DigitClassifier", "classifier_from_dict", "ClassifierConfig"]
