#!/usr/bin/env python3
"""Sentiment analysis CLI with optional dataset evaluation."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: vaderSentiment. Install it with 'pip install vaderSentiment'"
    ) from exc


@dataclass
class SentimentResult:
    text: str
    score: float
    label: str


class SentimentAnalyzer:
    """Wraps VADER to expose a simple API constrained to [-1, 1]."""

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> SentimentResult:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Input sentence must not be empty.")

        raw_score = self._analyzer.polarity_scores(cleaned)["compound"]
        bounded_score = max(-1.0, min(1.0, raw_score))
        label = self._score_to_label(bounded_score)
        return SentimentResult(text=cleaned, score=bounded_score, label=label)

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score <= -0.5:
            return "strongly negative"
        if score < -0.05:
            return "negative"
        if score <= 0.05:
            return "neutral"
        if score < 0.5:
            return "positive"
        return "strongly positive"


LABEL_ALIASES = {
    "strongly negative": "strongly negative",
    "very negative": "strongly negative",
    "negative": "negative",
    "neg": "negative",
    "-1": "negative",
    "neutral": "neutral",
    "0": "neutral",
    "positive": "positive",
    "pos": "positive",
    "1": "positive",
    "strongly positive": "strongly positive",
    "very positive": "strongly positive",
}

CANONICAL_MAP = {
    "strongly negative": "negative",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "strongly positive": "positive",
}


def _canonical_label(label: str) -> str:
    return CANONICAL_MAP[label]


def evaluate_dataset(path: Path, analyzer: SentimentAnalyzer) -> Tuple[int, int]:
    """Compare predicted labels against ground truth values from a CSV file."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = {"text", "label"} - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise SystemExit(f"Dataset is missing required column(s): {missing}")

        total = 0
        correct = 0
        for row in reader:
            text = (row.get("text") or "").strip()
            raw_label = (row.get("label") or "").strip().lower()
            if not text:
                continue

            expected_label = LABEL_ALIASES.get(raw_label)
            if expected_label is None:
                raise SystemExit(
                    f"Unrecognized label '{row.get('label')}' on row {total + 2}."
                    " Expected one of: "
                    + ", ".join(sorted(set(LABEL_ALIASES.values())))
                )

            predicted = analyzer.analyze(text).label
            if _canonical_label(predicted) == _canonical_label(expected_label):
                correct += 1
            total += 1

    if total == 0:
        raise SystemExit("Dataset did not contain any valid rows to evaluate.")

    return correct, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run interactive sentiment analysis or evaluate against a CSV dataset."
        )
    )
    parser.add_argument(
        "--evaluate",
        type=Path,
        metavar="CSV",
        help=(
            "Path to a CSV file with columns 'text' and 'label'."
            " Labels may be strongly positive/negative, positive/negative, or neutral."
        ),
    )
    return parser.parse_args()


def interactive_session(analyzer: SentimentAnalyzer) -> None:
    try:
        sentence = input("Enter a sentence to analyze: ")
        result = analyzer.analyze(sentence)
    except (EOFError, KeyboardInterrupt):
        print("\nNo input provided. Exiting.")
        return
    except ValueError as err:
        print(err)
        return

    print(f"Sentiment score: {result.score:.3f} ({result.label})")


def main() -> None:
    args = parse_args()
    analyzer = SentimentAnalyzer()

    if args.evaluate:
        csv_path = args.evaluate
        if not csv_path.exists():
            raise SystemExit(f"Cannot find dataset at '{csv_path}'.")
        correct, total = evaluate_dataset(csv_path, analyzer)
        accuracy = correct / total
        print(
            f"Evaluation results for {csv_path} -> accuracy: {accuracy:.3%}"
            f" ({correct} / {total} samples correct)"
        )
        return

    interactive_session(analyzer)


if __name__ == "__main__":
    main()
