"""Offline tests: schema validation rejects bad model output."""
import pytest
from pydantic import ValidationError

from cv_screener.models import CVResult


def _valid_data(**overrides) -> dict:
    base = {
        "name": "Jane Doe",
        "email": "jane@example.com",
        "undergrad_university": "MIT",
        "masters_university": None,
        "field_of_study": "Computer Science",
        "still_in_school": False,
        "years_experience": 4.0,
        "summary": "Strong backend engineer with 4 years of experience.",
        "score_rationale": "Meets all must-haves; limited ML exposure.",
        "fit_score": 7,
        "source_file": "jane_doe.pdf",
    }
    base.update(overrides)
    return base


def test_valid_result_parses():
    result = CVResult(**_valid_data())
    assert result.fit_score == 7
    assert result.masters_university is None


def test_fit_score_too_high_rejected():
    with pytest.raises(ValidationError):
        CVResult(**_valid_data(fit_score=15))


def test_fit_score_too_low_rejected():
    with pytest.raises(ValidationError):
        CVResult(**_valid_data(fit_score=0))


def test_missing_name_rejected():
    data = _valid_data()
    del data["name"]
    with pytest.raises(ValidationError):
        CVResult(**data)


def test_missing_still_in_school_rejected():
    data = _valid_data()
    del data["still_in_school"]
    with pytest.raises(ValidationError):
        CVResult(**data)


def test_null_optional_fields_accepted():
    result = CVResult(**_valid_data(
        email=None,
        undergrad_university=None,
        masters_university=None,
        field_of_study=None,
        years_experience=None,
    ))
    assert result.email is None
    assert result.years_experience is None


def test_fit_score_boundary_1():
    result = CVResult(**_valid_data(fit_score=1))
    assert result.fit_score == 1


def test_fit_score_boundary_10():
    result = CVResult(**_valid_data(fit_score=10))
    assert result.fit_score == 10
