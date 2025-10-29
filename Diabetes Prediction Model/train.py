"""CLI for training and evaluating the diabetes regression pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data import DatasetSplits, train_valid_test_split
from models import (
    evaluate_candidates,
    evaluate_on_test,
    save_model,
    select_best_model,
    train_final_model,
)

ARTIFACT_DIR = Path("artifacts")


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title("Residuals vs Predicted Values")
    plt.xlabel("Predicted progression")
    plt.ylabel("Residual (True - Pred)")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_feature_importance(model, path: Path) -> None:
    preprocess = model.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out()
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = np.array(estimator.coef_)
        importances = np.abs(coef)
    else:
        return

    order = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[order], y=[feature_names[i] for i in order], palette="viridis")
    plt.title("Top feature contributions")
    plt.xlabel("Importance (abs value)")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate diabetes regression models.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cv", type=int, default=8, help="Cross-validation folds for model comparison.")
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--valid_size", type=float, default=0.15)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits: DatasetSplits = train_valid_test_split(
        test_size=args.test_size,
        valid_size=args.valid_size,
        seed=args.seed,
    )

    candidate_results = evaluate_candidates(splits, cv=args.cv, random_state=args.seed)
    best_result = select_best_model(candidate_results)
    final_model = train_final_model(best_result, splits)
    test_metrics = evaluate_on_test(final_model, splits)

    y_test_pred = final_model.predict(splits.X_test)

    artifacts_dir = args.artifacts
    artifacts_dir.mkdir(exist_ok=True, parents=True)

    # Save metrics summary
    metrics_payload: Dict[str, object] = {
        "candidates": {
            name: {
                "cv": result.cv_metrics,
                "validation": result.validation_metrics,
            }
            for name, result in candidate_results.items()
        },
        "best_model": best_result.name,
        "test_metrics": test_metrics,
        "test_predictions": {
            "mean": float(y_test_pred.mean()),
            "std": float(y_test_pred.std()),
        },
    }

    metrics_path = artifacts_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    # Save model
    model_path = artifacts_dir / "best_model.joblib"
    save_model(final_model, model_path)

    # Plots
    _plot_residuals(splits.y_test.to_numpy(), y_test_pred, artifacts_dir / "residuals.png")
    _plot_feature_importance(final_model, artifacts_dir / "feature_importance.png")

    print(f"Best model: {best_result.name}")
    print("Validation metrics:", best_result.validation_metrics)
    print("Test metrics:", test_metrics)
    print(f"Artifacts saved to {artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
