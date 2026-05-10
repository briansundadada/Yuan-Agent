"""API-based Fundamental Analyst understanding layer.

This module is additive. It does not replace the deterministic offline
baseline builder in ``forecasting_system.tools.fundamentals`` and does not
execute financial predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forecasting_system.exceptions import SchemaValidationError
from forecasting_system.tools.deepseek_client import chat_completion, resolve_api_key
from forecasting_system.tools.events import load_event_library


MAX_FUNDAMENTAL_RETRIES = 3
REQUIRED_TOP_LEVEL_KEYS = {
    "company_name",
    "fundamental_summary",
    "baseline_state",
    "core_metrics",
    "candidate_scenarios",
    "qualitative_assessment",
}
REQUIRED_QUALITATIVE_KEYS = {
    "export_exposure",
    "cost_structure",
    "competitive_position",
    "growth_driver",
}


def analyze_fundamentals(
    raw_text: str | None = None,
    file_path: str | Path | None = None,
    api_key: str | None = None,
    max_retries: int = MAX_FUNDAMENTAL_RETRIES,
) -> dict[str, Any]:
    """Analyze local company materials into structured fundamental understanding."""
    material_text = _load_material_text(raw_text=raw_text, file_path=file_path)
    active_api_key = resolve_api_key(api_key, purpose="fundamental analysis")
    event_library = load_event_library()
    prompt = _build_fundamental_prompt(material_text, event_library)
    last_error: Exception | None = None

    for attempt in range(max_retries):
        response_payload = _post_chat_completion(prompt, active_api_key)
        try:
            understanding = _parse_fundamental_response(response_payload)
            understanding = _normalize_fundamental_understanding(understanding)
            validate_fundamental_understanding(understanding)
            return understanding
        except (json.JSONDecodeError, KeyError, TypeError, SchemaValidationError) as exc:
            last_error = exc
            prompt = _build_retry_prompt(material_text, event_library, attempt + 1)

    raise SchemaValidationError(
        f"DeepSeek fundamental analysis failed after {max_retries} attempts: {last_error}"
    )


def analyze_fundamentals_from_file(
    file_path: str | Path,
    api_key: str | None = None,
    max_retries: int = MAX_FUNDAMENTAL_RETRIES,
) -> dict[str, Any]:
    """Analyze one local report/research text file."""
    return analyze_fundamentals(file_path=file_path, api_key=api_key, max_retries=max_retries)


def validate_fundamental_understanding(understanding: dict[str, Any]) -> dict[str, Any]:
    """Validate the API output shape for Fundamental Analyst understanding."""
    if not isinstance(understanding, dict):
        raise SchemaValidationError("Fundamental understanding must be a JSON object.")

    missing = REQUIRED_TOP_LEVEL_KEYS.difference(understanding)
    if missing:
        raise SchemaValidationError(f"Fundamental understanding is missing keys: {sorted(missing)}")
    extra = set(understanding).difference(REQUIRED_TOP_LEVEL_KEYS)
    if extra:
        raise SchemaValidationError(f"Fundamental understanding contains unexpected keys: {sorted(extra)}")

    _validate_non_empty_string(understanding["company_name"], "company_name")
    _validate_non_empty_string(understanding["fundamental_summary"], "fundamental_summary")
    _validate_metric_pair(understanding["baseline_state"], "baseline_state")
    _validate_metric_pair(understanding["core_metrics"], "core_metrics")
    _validate_candidate_scenarios(understanding["candidate_scenarios"])
    _validate_qualitative_assessment(understanding["qualitative_assessment"])
    return understanding


def _load_material_text(raw_text: str | None, file_path: str | Path | None) -> str:
    materials = []
    if file_path is not None:
        materials.append(Path(file_path).read_text(encoding="utf-8").strip())
    if raw_text is not None:
        materials.append(raw_text.strip())

    material_text = "\n\n".join(text for text in materials if text)
    if not material_text:
        raise SchemaValidationError("analyze_fundamentals() requires raw_text or file_path input.")
    return material_text


def _build_fundamental_prompt(material_text: str, event_library: dict) -> str:
    taxonomy_json = json.dumps(event_library, ensure_ascii=False, indent=2)
    return (
        "Analyze the company materials and return JSON only. Do not include markdown or free text.\n"
        "This is the Fundamental Analyst Agent, not the Student prediction executor.\n"
        "Compute and report factual historical indicators only; do not predict future ROA or ROE. Do NOT predict ROA directly. Do NOT modify rules.\n"
        "Use cumulative/year-to-date ROE for interim reports and distinguish flow metrics from balance-sheet stock metrics.\n"
        "Keep equity_multiplier in the fundamental analysis, and calculate reported_roa as profit_margin * asset_turnover.\n"
        "Use profit_margin, asset_turnover, and equity_multiplier only as baseline financial factors.\n"
        "candidate_scenarios are observation-only exploratory outputs for later review.\n"
        "Prefer scenario and event_type labels from the controlled taxonomy when a mapping is clear. "
        "If exact mapping is uncertain, still return structured candidate_scenarios and explain the uncertainty in reason.\n"
        f"Controlled event taxonomy:\n{taxonomy_json}\n"
        "Return exactly this JSON object shape and no extra keys:\n"
        "{\n"
        '  "company_name": "string",\n'
        '  "fundamental_summary": "string",\n'
        '  "baseline_state": {\n'
        '    "profit_margin": 0.0,\n'
        '    "asset_turnover": 0.0,\n'
        '    "equity_multiplier": 0.0\n'
        "  },\n"
        '  "core_metrics": {\n'
        '    "profit_margin": 0.0,\n'
        '    "asset_turnover": 0.0,\n'
        '    "equity_multiplier": 0.0,\n'
        '    "reported_roa": 0.0,\n'
        '    "reported_roe": 0.0\n'
        "  },\n"
        '  "candidate_scenarios": [\n'
        '    {"scenario": "string", "event_type": "string", "reason": "string"}\n'
        "  ],\n"
        '  "qualitative_assessment": {\n'
        '    "export_exposure": "string",\n'
        '    "cost_structure": "string",\n'
        '    "competitive_position": "string",\n'
        '    "growth_driver": "string"\n'
        "  }\n"
        "}\n"
        "Company materials:\n"
        f"{material_text}"
    )


def _build_retry_prompt(material_text: str, event_library: dict, attempt_number: int) -> str:
    taxonomy_json = json.dumps(event_library, ensure_ascii=False, indent=2)
    return (
        f"Retry attempt {attempt_number}. Return valid JSON only with the exact required shape. "
        "Do not predict or include forward ROA or ROE. baseline_state must include numeric "
        "profit_margin, asset_turnover, and equity_multiplier. core_metrics must include numeric "
        "profit_margin, asset_turnover, equity_multiplier, reported_roa, and reported_roe. candidate_scenarios must be a list of objects with "
        "scenario, event_type, and reason. qualitative_assessment must include export_exposure, "
        "cost_structure, competitive_position, and growth_driver.\n"
        f"Controlled event taxonomy:\n{taxonomy_json}\n"
        "Company materials:\n"
        f"{material_text}"
    )


def _post_chat_completion(prompt: str, api_key: str) -> dict:
    return chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "You are the Fundamental Analyst Agent. "
                    "Return one valid JSON object only. Do not predict forward ROA or ROE."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        api_key=api_key,
        response_format={"type": "json_object"},
        timeout=30,
        purpose="DeepSeek fundamental analysis request",
    )


def _parse_fundamental_response(response_payload: dict) -> dict[str, Any]:
    content = response_payload["choices"][0]["message"]["content"]
    understanding = json.loads(content)
    if not isinstance(understanding, dict):
        raise SchemaValidationError("DeepSeek fundamental response must decode to a JSON object.")
    return understanding


def _normalize_fundamental_understanding(understanding: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(understanding)
    baseline_state = dict(normalized.get("baseline_state", {}))
    core_metrics = dict(normalized.get("core_metrics", {}))

    baseline_state.setdefault("equity_multiplier", core_metrics.get("equity_multiplier", 1.0))
    core_metrics.setdefault("equity_multiplier", baseline_state.get("equity_multiplier", 1.0))

    if "reported_roa" not in core_metrics:
        profit_margin = core_metrics.get("profit_margin")
        asset_turnover = core_metrics.get("asset_turnover")
        if isinstance(profit_margin, (int, float)) and isinstance(asset_turnover, (int, float)):
            core_metrics["reported_roa"] = float(profit_margin) * float(asset_turnover)
    if "reported_roe" not in core_metrics:
        reported_roa = core_metrics.get("reported_roa")
        equity_multiplier = core_metrics.get("equity_multiplier")
        if isinstance(reported_roa, (int, float)) and isinstance(equity_multiplier, (int, float)):
            core_metrics["reported_roe"] = float(reported_roa) * float(equity_multiplier)

    normalized["baseline_state"] = baseline_state
    normalized["core_metrics"] = core_metrics
    return normalized


def _validate_non_empty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"Field {field_name!r} must be a non-empty string.")


def _validate_metric_pair(value: Any, field_name: str) -> None:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"Field {field_name!r} must be an object.")
    required = {"profit_margin", "asset_turnover", "equity_multiplier"}
    missing = required.difference(value)
    if missing:
        raise SchemaValidationError(f"Field {field_name!r} is missing keys: {sorted(missing)}")
    allowed = set(required)
    if field_name == "core_metrics":
        allowed.add("reported_roe")
        allowed.add("reported_roa")
    extra = set(value).difference(allowed)
    if extra:
        raise SchemaValidationError(f"Field {field_name!r} contains unexpected keys: {sorted(extra)}")
    for metric_name in required:
        if not isinstance(value[metric_name], (int, float)):
            raise SchemaValidationError(f"Field {field_name}.{metric_name} must be numeric.")
    if field_name == "core_metrics":
        if "reported_roe" not in value or not isinstance(value["reported_roe"], (int, float)):
            raise SchemaValidationError("Field core_metrics.reported_roe must be numeric.")
        if "reported_roa" not in value or not isinstance(value["reported_roa"], (int, float)):
            raise SchemaValidationError("Field core_metrics.reported_roa must be numeric.")


def _validate_candidate_scenarios(value: Any) -> None:
    if not isinstance(value, list):
        raise SchemaValidationError("candidate_scenarios must be a list.")
    for item in value:
        if not isinstance(item, dict):
            raise SchemaValidationError("Each candidate_scenarios item must be an object.")
        required = {"scenario", "event_type", "reason"}
        missing = required.difference(item)
        if missing:
            raise SchemaValidationError(f"candidate_scenarios item is missing keys: {sorted(missing)}")
        extra = set(item).difference(required)
        if extra:
            raise SchemaValidationError(f"candidate_scenarios item contains unexpected keys: {sorted(extra)}")
        for key in required:
            _validate_non_empty_string(item[key], f"candidate_scenarios.{key}")


def _validate_qualitative_assessment(value: Any) -> None:
    if not isinstance(value, dict):
        raise SchemaValidationError("qualitative_assessment must be an object.")
    missing = REQUIRED_QUALITATIVE_KEYS.difference(value)
    if missing:
        raise SchemaValidationError(f"qualitative_assessment is missing keys: {sorted(missing)}")
    extra = set(value).difference(REQUIRED_QUALITATIVE_KEYS)
    if extra:
        raise SchemaValidationError(f"qualitative_assessment contains unexpected keys: {sorted(extra)}")
    for key in REQUIRED_QUALITATIVE_KEYS:
        _validate_non_empty_string(value[key], f"qualitative_assessment.{key}")
