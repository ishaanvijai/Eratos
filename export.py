"""Standalone exporter: results.jsonl -> CSV sorted by fit_score descending."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_COLUMNS = [
    "name",
    "email",
    "undergrad_university",
    "masters_university",
    "field_of_study",
    "still_in_school",
    "years_experience",
    "fit_score",
    "talent_flag",
    "score_rationale",
    "summary",
    "source_file",
]


def export(results_file: Path, output_file: Path) -> int:
    if not results_file.exists():
        sys.exit(f"Results file not found: {results_file}")

    rows = []
    with results_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                import json
                rows.append(json.loads(line))

    if not rows:
        sys.exit("No results to export.")

    df = pd.DataFrame(rows)

    # Ensure all expected columns exist (fill missing with None)
    for col in _COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[_COLUMNS].sort_values("fit_score", ascending=False).reset_index(drop=True)
    df.to_csv(output_file, index=False)
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export results.jsonl to CSV.")
    parser.add_argument(
        "--results", type=Path, default=Path("results.jsonl"),
        help="Path to results.jsonl (default: results.jsonl)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results.csv"),
        help="Output CSV path (default: results.csv)"
    )
    args = parser.parse_args()

    n = export(args.results, args.output)
    print(f"Exported {n} rows -> {args.output}")


if __name__ == "__main__":
    main()
