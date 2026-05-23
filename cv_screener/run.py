"""Entry point: route non-PDFs to _unprocessed/, then run the pipeline."""
from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
from pathlib import Path

from .config import CV_DIR, PROVIDER, MODEL_NAME, UNPROCESSED_DIR, SUPPORTED_EXTS
from . import pipeline
from .providers import get_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _quarantine_unsupported(cv_dir: Path, unprocessed_dir: Path) -> None:
    unsupported = [
        f for f in cv_dir.iterdir()
        if f.is_file() and f.suffix.lower() not in SUPPORTED_EXTS
    ]
    if not unsupported:
        return
    unprocessed_dir.mkdir(parents=True, exist_ok=True)
    for f in unsupported:
        dest = unprocessed_dir / f.name
        shutil.move(str(f), dest)
        logger.info("Quarantined unsupported format: %s -> %s", f.name, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen CVs with an LLM provider.")
    parser.add_argument("--cv-dir", type=Path, default=CV_DIR)
    parser.add_argument("--provider", default=PROVIDER)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--force", action="store_true", help="Re-process all CVs")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N CVs (use this for a small test run)",
    )
    args = parser.parse_args()

    cv_dir: Path = args.cv_dir
    if not cv_dir.is_dir():
        sys.exit(f"CV directory not found: {cv_dir}")

    unprocessed = UNPROCESSED_DIR if UNPROCESSED_DIR.parent == cv_dir else cv_dir / "_unprocessed"
    _quarantine_unsupported(cv_dir, unprocessed)

    provider = get_provider(args.provider, model=args.model)
    asyncio.run(pipeline.run(provider, cv_dir, force=args.force, limit=args.limit))


if __name__ == "__main__":
    main()
