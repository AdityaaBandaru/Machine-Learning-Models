"""Model selection and evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from data import DatasetSplits
from features import build_feature_pipeline


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    cv_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]


SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def _regressor_candidates(random_state: int) -> Dict[str, object]:
    return {
        "elastic_net": ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            n_alphas=200,
            cv=5,
            n_jobs=None,
            random_state=random_state,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=600,
            max_depth=8,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            learning_rate=0.05,
            n_estimators=400,
            max_depth=3,
            subsample=0.9,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_depth=6,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=random_state,
        ),
    }


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }


def evaluate_candidates(splits: DatasetSplits, cv: int, random_state: int) -> Dict[str, ModelResult]:
    results: Dict[str, ModelResult] = {}
    preprocessor = build_feature_pipeline()

    for name, estimator in _regressor_candidates(random_state).items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )

        cv_result = cross_validate(
            pipeline,
            splits.X_train,
            splits.y_train,
            scoring=SCORING,
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )
        cv_metrics = {
            metric: float(np.mean(scores)) * (-1 if metric in {"mae", "rmse"} else 1)
            for metric, scores in {
                "mae": cv_result["test_mae"],
                "rmse": cv_result["test_rmse"],
                "r2": cv_result["test_r2"],
            }.items()
        }

        pipeline.fit(splits.X_train, splits.y_train)
        y_valid_pred = pipeline.predict(splits.X_valid)
        validation_metrics = _metrics_dict(splits.y_valid, y_valid_pred)

        results[name] = ModelResult(
            name=name,
            pipeline=pipeline,
            cv_metrics=cv_metrics,
            validation_metrics=validation_metrics,
        )

    return results


def select_best_model(results: Dict[str, ModelResult]) -> ModelResult:
    return min(results.values(), key=lambda res: res.validation_metrics["rmse"])


def train_final_model(best: ModelResult, splits: DatasetSplits) -> Pipeline:
    X_final = pd.concat([splits.X_train, splits.X_valid], axis=0)
    y_final = pd.concat([splits.y_train, splits.y_valid], axis=0)
    best.pipeline.fit(X_final, y_final)
    return best.pipeline


def evaluate_on_test(model: Pipeline, splits: DatasetSplits) -> Dict[str, float]:
    y_pred = model.predict(splits.X_test)
    return _metrics_dict(splits.y_test, y_pred)


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


__all__ = [
    "evaluate_candidates",
    "select_best_model",
    "train_final_model",
    "evaluate_on_test",
    "save_model",
    "ModelResult",
]
