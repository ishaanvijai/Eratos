"""Gated smoke test: opt-in, makes real API calls against ~10 PDFs.

Run with:
    pytest -m smoke --smoke-dir /path/to/cv/sample/

Never runs by default — requires the explicit -m smoke marker.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from cv_screener.models import CVResult
from cv_screener.providers import GoogleProvider


def pytest_addoption(parser):
    parser.addoption(
        "--smoke-dir",
        default=None,
        help="Directory of real CV PDFs for the smoke test",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: opt-in smoke test that makes real API calls")


@pytest.fixture
def smoke_dir(request):
    path = request.config.getoption("--smoke-dir")
    if path is None:
        pytest.skip("Provide --smoke-dir /path/to/pdfs to run the smoke test")
    return Path(path)


@pytest.fixture
def google_provider():
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not set")
    return GoogleProvider(api_key=api_key)


@pytest.mark.smoke
def test_smoke_10_cvs(smoke_dir, google_provider, tmp_path):
    system_prompt = (
        "You are evaluating engineering candidate CVs. "
        "Score each candidate on fit for a senior software engineer role."
    )

    pdfs = sorted(smoke_dir.glob("*.pdf"))[:10]
    if not pdfs:
        pytest.skip(f"No PDFs found in {smoke_dir}")

    results = []
    errors = []

    async def run():
        for pdf in pdfs:
            try:
                result = await google_provider.score_cv(
                    pdf.read_bytes(), system_prompt, pdf.name
                )
                results.append(result)
            except Exception as exc:
                errors.append((pdf.name, str(exc)))

    asyncio.run(run())

    assert not errors, f"Errors on: {errors}"
    assert len(results) == len(pdfs)

    for r in results:
        assert isinstance(r, CVResult)
        assert 1 <= r.fit_score <= 10
        assert r.name
        assert r.source_file
