"""Provider abstraction layer.

Each provider hides its own PDF-upload mechanism and structured-output handling.
The pipeline only calls score_cv(pdf_bytes, system_prompt) -> CVResult.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from .models import CVResult


class Provider(ABC):
    @abstractmethod
    async def score_cv(self, pdf_bytes: bytes, system_prompt: str, source_file: str) -> CVResult:
        """Score a CV PDF and return a structured CVResult."""


class GoogleProvider(Provider):
    """Gemini adapter using the Files API for PDF upload and response_schema for structured output."""

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str = "") -> None:
        import google.generativeai as genai

        if not api_key:
            from . import config
            api_key = config.GOOGLE_API_KEY

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model
        self._model: Any = None

    def _get_model(self, system_prompt: str) -> Any:
        # Re-create the model only when the system prompt changes (prompt caching).
        # Gemini caches the system_instruction automatically across calls to the same model instance.
        if self._model is None:
            self._model = self._genai.GenerativeModel(
                model_name=self._model_name,
                system_instruction=system_prompt,
                generation_config=self._genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=_cv_result_schema(),
                ),
            )
        return self._model

    async def score_cv(self, pdf_bytes: bytes, system_prompt: str, source_file: str) -> CVResult:
        import asyncio
        import tempfile
        import os

        model = self._get_model(system_prompt)

        # Upload PDF via Files API
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            uploaded = await asyncio.to_thread(
                self._genai.upload_file, tmp_path, mime_type="application/pdf"
            )
        finally:
            os.unlink(tmp_path)

        try:
            response = await asyncio.to_thread(
                model.generate_content,
                [uploaded, "Evaluate this CV according to the system instructions."],
            )
        finally:
            await asyncio.to_thread(self._genai.delete_file, uploaded.name)

        data = json.loads(response.text)
        data["source_file"] = source_file
        return CVResult(**data)


class AnthropicProvider(Provider):
    async def score_cv(self, pdf_bytes: bytes, system_prompt: str, source_file: str) -> CVResult:
        raise NotImplementedError("AnthropicProvider is not implemented in v1")


class OpenAIProvider(Provider):
    async def score_cv(self, pdf_bytes: bytes, system_prompt: str, source_file: str) -> CVResult:
        raise NotImplementedError("OpenAIProvider is not implemented in v1")


def get_provider(name: str = "google", model: str = "") -> Provider:
    name = name.lower()
    if name == "google":
        from . import config
        return GoogleProvider(model=model or config.MODEL_NAME)
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unknown provider: {name!r}. Choose from: google, anthropic, openai")


def _cv_result_schema() -> dict:
    """JSON schema for Gemini response_schema (subset of OpenAPI 3.0)."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string", "nullable": True},
            "undergrad_university": {"type": "string", "nullable": True},
            "masters_university": {"type": "string", "nullable": True},
            "field_of_study": {"type": "string", "nullable": True},
            "still_in_school": {"type": "boolean"},
            "years_experience": {"type": "number", "nullable": True},
            "summary": {"type": "string"},
            "score_rationale": {"type": "string"},
            "fit_score": {"type": "integer"},
        },
        "required": [
            "name",
            "still_in_school",
            "summary",
            "score_rationale",
            "fit_score",
        ],
    }
