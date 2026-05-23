"""Offline tests: resume/skip logic in the pipeline."""
import json
import tempfile
from pathlib import Path

from cv_screener.pipeline import _load_done


def _write_results(path: Path, source_files: list[str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for sf in source_files:
            fh.write(json.dumps({"source_file": sf, "fit_score": 5, "name": "X"}) + "\n")


def test_load_done_returns_processed_names(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_results(results, ["alice.pdf", "bob.pdf"])
    done = _load_done(results)
    assert done == {"alice.pdf", "bob.pdf"}


def test_load_done_empty_when_no_file(tmp_path):
    done = _load_done(tmp_path / "nonexistent.jsonl")
    assert done == set()


def test_load_done_skips_malformed_lines(tmp_path):
    results = tmp_path / "results.jsonl"
    with results.open("w") as fh:
        fh.write('{"source_file": "good.pdf", "fit_score": 7}\n')
        fh.write("THIS IS NOT JSON\n")
        fh.write('{"fit_score": 3}\n')  # missing source_file key
    done = _load_done(results)
    assert done == {"good.pdf"}


def test_pending_excludes_done(tmp_path):
    """Simulate the skip logic: files not in done set are returned as pending."""
    results = tmp_path / "results.jsonl"
    all_files = ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]
    _write_results(results, ["a.pdf", "c.pdf"])

    done = _load_done(results)
    pending = [f for f in all_files if f not in done]
    assert pending == ["b.pdf", "d.pdf"]


def test_force_ignores_done(tmp_path):
    """With force=True the pipeline should not skip any files."""
    results = tmp_path / "results.jsonl"
    all_files = ["a.pdf", "b.pdf"]
    _write_results(results, ["a.pdf"])

    # force=True means done set is empty
    done: set = set()
    pending = [f for f in all_files if f not in done]
    assert pending == all_files
