"""Batch prediction helper for diabetes regression model."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data import FEATURE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run predictions with a trained diabetes regression model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained joblib pipeline.")
    parser.add_argument("--input", type=Path, required=True, help="CSV with feature columns.")
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Where to write predictions CSV (use '-' for stdout).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    df = pd.read_csv(args.input)

    missing = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing:
        raise ValueError(f"Input file missing columns: {missing}")

    predictions = model.predict(df[FEATURE_NAMES])
    output_df = df.copy()
    output_df["predicted_progression"] = predictions

    if args.output == "-":
        output_df.to_csv(sys.stdout, index=False)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
