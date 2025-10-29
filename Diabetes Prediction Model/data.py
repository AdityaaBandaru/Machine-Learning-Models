"""Data loading and splitting utilities for the diabetes regression project."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplits:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


FEATURE_NAMES = [
    "age",
    "sex",
    "bmi",
    "bp",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
]
TARGET_NAME = "disease_progression"


def load_dataset(as_frame: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = load_diabetes(as_frame=as_frame)
    X = dataset.data.copy()
    y = dataset.target.rename(TARGET_NAME)
    X.columns = FEATURE_NAMES
    return X, y


def train_valid_test_split(
    test_size: float = 0.15,
    valid_size: float = 0.15,
    seed: int = 1337,
) -> DatasetSplits:
    X, y = load_dataset(as_frame=True)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )

    valid_ratio = valid_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=valid_ratio,
        random_state=seed,
    )

    return DatasetSplits(
        X_train=X_train.reset_index(drop=True),
        X_valid=X_valid.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


__all__ = [
    "DatasetSplits",
    "FEATURE_NAMES",
    "TARGET_NAME",
    "load_dataset",
    "train_valid_test_split",
]
