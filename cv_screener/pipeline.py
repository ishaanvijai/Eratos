"""Core pipeline: load prompts, run concurrent CV scoring, persist results."""
from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Optional, Set

from .config import (
    CONCURRENCY,
    ERRORS_FILE,
    RESULTS_FILE,
    SYSTEM_PROMPT_FILE,
    USER_PROMPT_FILE,
)
from .models import CVResult
from .providers import Provider

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_BASE_BACKOFF = 1.0  # seconds


def _load_system_prompt() -> str:
    system = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    prompt = USER_PROMPT_FILE.read_text(encoding="utf-8").strip()
    return f"{system}\n\n{prompt}"


def _load_done(results_file: Path) -> Set[str]:
    """Return the set of source_file values already in results.jsonl."""
    done: Set[str] = set()
    if not results_file.exists():
        return done
    with results_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(row["source_file"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _append_result(result: CVResult, results_file: Path) -> None:
    with results_file.open("a", encoding="utf-8") as fh:
        fh.write(result.model_dump_json() + "\n")


def _append_error(source_file: str, error: str, errors_file: Path) -> None:
    with errors_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"source_file": source_file, "error": error}) + "\n")


async def _score_with_retry(
    provider: Provider,
    pdf_bytes: bytes,
    system_prompt: str,
    source_file: str,
) -> Optional[CVResult]:
    delay = _BASE_BACKOFF
    for attempt in range(_MAX_RETRIES):
        try:
            return await provider.score_cv(pdf_bytes, system_prompt, source_file)
        except Exception as exc:
            msg = str(exc)
            is_rate_limit = "429" in msg or "quota" in msg.lower() or "rate" in msg.lower()
            if is_rate_limit and attempt < _MAX_RETRIES - 1:
                jitter = random.uniform(0, delay * 0.2)
                wait = delay + jitter
                logger.warning("Rate limit on %s — retrying in %.1fs (attempt %d)", source_file, wait, attempt + 1)
                await asyncio.sleep(wait)
                delay *= 2
            else:
                logger.error("Failed %s after %d attempts: %s", source_file, attempt + 1, exc)
                return None
    return None


async def _process_one(
    sem: asyncio.Semaphore,
    provider: Provider,
    pdf_path: Path,
    system_prompt: str,
    results_file: Path,
    errors_file: Path,
) -> None:
    source_file = pdf_path.name
    async with sem:
        pdf_bytes = pdf_path.read_bytes()
        result = await _score_with_retry(provider, pdf_bytes, system_prompt, source_file)

    if result is not None:
        _append_result(result, results_file)
        logger.info("OK  %s  score=%d", source_file, result.fit_score)
    else:
        _append_error(source_file, "max retries exceeded", errors_file)
        logger.warning("ERR %s", source_file)


async def run(
    provider: Provider,
    pdf_dir: Path,
    *,
    force: bool = False,
    results_file: Optional[Path] = None,
    errors_file: Optional[Path] = None,
    concurrency: Optional[int] = None,
) -> None:
    results_file = results_file or RESULTS_FILE
    errors_file = errors_file or ERRORS_FILE
    limit = concurrency or CONCURRENCY

    system_prompt = _load_system_prompt()
    done = set() if force else _load_done(results_file)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    pending = [p for p in pdfs if p.name not in done]

    logger.info(
        "CVs: %d total, %d already done, %d to process",
        len(pdfs),
        len(done),
        len(pending),
    )

    sem = asyncio.Semaphore(limit)
    tasks = [
        _process_one(sem, provider, p, system_prompt, results_file, errors_file)
        for p in pending
    ]
    await asyncio.gather(*tasks)
    logger.info("Pipeline complete.")
