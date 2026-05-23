"""Offline tests: export produces a CSV sorted by fit_score descending."""
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from export import export, _COLUMNS


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _base_row(**kwargs) -> dict:
    base = {
        "name": "Test Candidate",
        "email": None,
        "undergrad_university": None,
        "masters_university": None,
        "field_of_study": None,
        "still_in_school": False,
        "years_experience": 3.0,
        "summary": "A summary.",
        "score_rationale": "A rationale.",
        "fit_score": 5,
        "source_file": "test.pdf",
    }
    base.update(kwargs)
    return base


def test_export_sorted_descending(tmp_path):
    results = tmp_path / "results.jsonl"
    output = tmp_path / "results.csv"
    rows = [
        _base_row(name="Low", fit_score=3, source_file="low.pdf"),
        _base_row(name="High", fit_score=9, source_file="high.pdf"),
        _base_row(name="Mid", fit_score=6, source_file="mid.pdf"),
    ]
    _write_jsonl(results, rows)

    n = export(results, output)

    assert n == 3
    df = pd.read_csv(output)
    assert list(df["fit_score"]) == [9, 6, 3]
    assert list(df["name"]) == ["High", "Mid", "Low"]


def test_export_has_expected_columns(tmp_path):
    results = tmp_path / "results.jsonl"
    output = tmp_path / "results.csv"
    _write_jsonl(results, [_base_row()])

    export(results, output)

    df = pd.read_csv(output)
    assert list(df.columns) == _COLUMNS


def test_export_single_row(tmp_path):
    results = tmp_path / "results.jsonl"
    output = tmp_path / "results.csv"
    _write_jsonl(results, [_base_row(fit_score=8, name="Solo")])

    n = export(results, output)

    assert n == 1
    df = pd.read_csv(output)
    assert df.iloc[0]["name"] == "Solo"
    assert df.iloc[0]["fit_score"] == 8


def test_export_missing_results_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        export(tmp_path / "nonexistent.jsonl", tmp_path / "out.csv")


def test_export_empty_file_exits(tmp_path):
    results = tmp_path / "results.jsonl"
    results.write_text("")
    with pytest.raises(SystemExit):
        export(results, tmp_path / "out.csv")
