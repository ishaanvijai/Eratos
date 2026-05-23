"""Entry point: route non-PDFs to _unprocessed/, then run the pipeline."""
from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
from pathlib import Path

from .config import CV_DIR, PROVIDER, MODEL_NAME, UNPROCESSED_DIR
from . import pipeline
from .providers import get_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _quarantine_non_pdfs(cv_dir: Path, unprocessed_dir: Path) -> None:
    non_pdfs = [f for f in cv_dir.iterdir() if f.is_file() and f.suffix.lower() != ".pdf"]
    if not non_pdfs:
        return
    unprocessed_dir.mkdir(parents=True, exist_ok=True)
    for f in non_pdfs:
        dest = unprocessed_dir / f.name
        shutil.move(str(f), dest)
        logger.info("Quarantined non-PDF: %s -> %s", f.name, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen CVs with an LLM provider.")
    parser.add_argument("--cv-dir", type=Path, default=CV_DIR)
    parser.add_argument("--provider", default=PROVIDER)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--force", action="store_true", help="Re-process all CVs")
    args = parser.parse_args()

    cv_dir: Path = args.cv_dir
    if not cv_dir.is_dir():
        sys.exit(f"CV directory not found: {cv_dir}")

    unprocessed = UNPROCESSED_DIR if UNPROCESSED_DIR.parent == cv_dir else cv_dir / "_unprocessed"
    _quarantine_non_pdfs(cv_dir, unprocessed)

    provider = get_provider(args.provider, model=args.model)
    asyncio.run(pipeline.run(provider, cv_dir, force=args.force))


if __name__ == "__main__":
    main()
