"""Feature engineering pipeline for diabetes regression."""
from __future__ import annotations

from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, QuantileTransformer, StandardScaler

from data import FEATURE_NAMES


INTERACTION_FEATURES: List[str] = ["bmi", "bp", "s5"]


def build_feature_pipeline() -> ColumnTransformer:
    """Create a preprocessing pipeline with scaling and targeted interactions."""
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    interaction_pipeline = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(include_bias=False, degree=2)),
            ("quantile", QuantileTransformer(output_distribution="normal", random_state=1337)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, FEATURE_NAMES),
            ("interactions", interaction_pipeline, INTERACTION_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


__all__ = ["build_feature_pipeline", "INTERACTION_FEATURES"]
