"""Deterministic offline baseline-state builder for phase 1.5."""

from __future__ import annotations

import json
import re
from pathlib import Path

from forecasting_system.config import FUNDAMENTALS_DIR
from forecasting_system.exceptions import SchemaValidationError
from forecasting_system.types import FundamentalBaseline


def build_baseline_state(source: str | Path) -> FundamentalBaseline:
    """Build a baseline state from a local JSON file or plain-text summary."""
    source_path = Path(source)
    raw_text = source_path.read_text(encoding="utf-8").strip()

    if source_path.suffix.lower() == ".json":
        baseline = _build_from_json(raw_text)
    else:
        baseline = _build_from_text(raw_text)

    _validate_baseline_output(baseline)
    return baseline


def load_sungrow_baseline(
    source: str | Path | None = None,
) -> FundamentalBaseline:
    """Load the default offline Sungrow baseline."""
    source_path = FUNDAMENTALS_DIR / "sungrow_baseline.json" if source is None else Path(source)
    return build_baseline_state(source_path)


def _build_from_json(raw_text: str) -> FundamentalBaseline:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError("Invalid JSON input for baseline state builder.") from exc

    if not isinstance(data, dict):
        raise SchemaValidationError("Baseline JSON input must be an object.")

    baseline: FundamentalBaseline = {
        "profit_margin": _coerce_float(data.get("profit_margin"), "profit_margin"),
        "asset_turnover": _coerce_float(data.get("asset_turnover"), "asset_turnover"),
        "equity_multiplier": _coerce_float(data.get("equity_multiplier", 1.0), "equity_multiplier"),
        "source_summary": _coerce_summary(data.get("source_summary")),
    }
    if "confidence" in data:
        baseline["confidence"] = _coerce_float(data.get("confidence"), "confidence")
    return baseline


def _build_from_text(raw_text: str) -> FundamentalBaseline:
    profit_margin = _extract_numeric_value(raw_text, "profit_margin")
    asset_turnover = _extract_numeric_value(raw_text, "asset_turnover")
    confidence_match = _search_value(raw_text, "confidence")

    baseline: FundamentalBaseline = {
        "profit_margin": profit_margin,
        "asset_turnover": asset_turnover,
        "equity_multiplier": _extract_optional_numeric_value(raw_text, "equity_multiplier", default=1.0),
        "source_summary": raw_text,
    }
    if confidence_match is not None:
        baseline["confidence"] = float(confidence_match)
    return baseline


def _extract_numeric_value(raw_text: str, field_name: str) -> float:
    matched_value = _search_value(raw_text, field_name)
    if matched_value is None:
        raise SchemaValidationError(f"Missing required field {field_name!r} in text baseline input.")
    return float(matched_value)


def _extract_optional_numeric_value(raw_text: str, field_name: str, default: float) -> float:
    matched_value = _search_value(raw_text, field_name)
    return default if matched_value is None else float(matched_value)


def _search_value(raw_text: str, field_name: str) -> str | None:
    pattern = rf"{re.escape(field_name)}\s*:\s*(-?\d+(?:\.\d+)?)"
    match = re.search(pattern, raw_text, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.group(1)


def _coerce_float(value, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise SchemaValidationError(f"Field {field_name!r} must be numeric.")
    return float(value)


def _coerce_summary(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError("Field 'source_summary' must be a non-empty string.")
    return value.strip()


def _validate_baseline_output(baseline: FundamentalBaseline) -> None:
    required_keys = {"profit_margin", "asset_turnover", "equity_multiplier", "source_summary"}
    missing = required_keys.difference(baseline)
    if missing:
        raise SchemaValidationError(f"Baseline output is missing required keys: {sorted(missing)}")
    if baseline["source_summary"] == "":
        raise SchemaValidationError("source_summary must not be empty.")
