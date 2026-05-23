from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Provider selection
PROVIDER: str = os.getenv("CV_PROVIDER", "google")
MODEL_NAME: str = os.getenv("CV_MODEL", "gemini-2.5-flash")

# Concurrency
CONCURRENCY: int = int(os.getenv("CV_CONCURRENCY", "8"))

# Supported CV file formats and their MIME types (Gemini ingests PDFs and images natively).
# Single source of truth — discovery, quarantine, and upload all read from here.
MIME_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
SUPPORTED_EXTS: frozenset[str] = frozenset(MIME_TYPES)

# API keys
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Paths
BASE_DIR = Path(__file__).parent.parent
CV_DIR: Path = Path(os.getenv("CV_DIR", BASE_DIR / "cvs"))
RESULTS_FILE: Path = Path(os.getenv("CV_RESULTS", BASE_DIR / "results.jsonl"))
ERRORS_FILE: Path = Path(os.getenv("CV_ERRORS", BASE_DIR / "errors.jsonl"))
UNPROCESSED_DIR: Path = Path(os.getenv("CV_UNPROCESSED", BASE_DIR / "cvs" / "_unprocessed"))
SYSTEM_PROMPT_FILE: Path = Path(os.getenv("CV_SYSTEM", BASE_DIR / "system.txt"))
USER_PROMPT_FILE: Path = Path(os.getenv("CV_PROMPT", BASE_DIR / "prompt.txt"))
