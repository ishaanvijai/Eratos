from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class CVResult(BaseModel):
    name: str
    email: Optional[str] = None
    undergrad_university: Optional[str] = None
    masters_university: Optional[str] = None
    field_of_study: Optional[str] = None
    still_in_school: bool
    years_experience: Optional[float] = None
    summary: str
    # score_rationale before fit_score so the model reasons first
    score_rationale: str
    fit_score: int = Field(..., ge=1, le=10)
    # Orthogonal to fit_score: standout talent worth tracking regardless of fit
    # for THIS role. null = not flagged; otherwise a short phrase on why.
    talent_flag: Optional[str] = None
    source_file: str

    @field_validator("fit_score")
    @classmethod
    def validate_fit_score(cls, v: int) -> int:
        if not 1 <= v <= 10:
            raise ValueError(f"fit_score must be 1-10, got {v}")
        return v
