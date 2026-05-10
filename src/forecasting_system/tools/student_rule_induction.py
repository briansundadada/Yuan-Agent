"""DeepSeek-assisted student induction of Sungrow-specific events and rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forecasting_system.config import (
    EVENT_LIBRARY_PATH,
    RULES_PATH,
)
from forecasting_system.exceptions import SchemaValidationError
from forecasting_system.tools.deepseek_client import chat_json, resolve_api_key
from forecasting_system.tools.events import validate_event_library
from forecasting_system.tools.report_analysis import (
    extract_pdf_text,
    get_report_path_for_month,
    load_or_analyze_report_pdf,
)
from forecasting_system.tools.rules import validate_rules


def generate_student_event_library(
    output_dir: str | Path,
    report_cache_path: str | Path | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Generate only a student-authored Sungrow event library from the 2023 annual report."""
    report_path = get_report_path_for_month(0)
    if report_path is None:
        raise SchemaValidationError("Missing 2023 annual report path for student event generation.")

    example_event_library = json.loads(Path(EVENT_LIBRARY_PATH).read_text(encoding="utf-8-sig"))
    report_analysis = load_or_analyze_report_pdf(
        report_path,
        "2023FY",
        cache_path=report_cache_path,
        api_key=api_key,
    )
    report_text = extract_pdf_text(report_path)

    active_api_key = resolve_api_key(api_key, purpose="student prior induction")

    generated = _generate_event_library_with_deepseek(
        report_text=report_text,
        report_analysis=report_analysis,
        example_event_library=example_event_library,
        api_key=active_api_key,
    )
    validated = _validate_generated_event_library_payload(generated)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "student_event_library.json").write_text(
        json.dumps(validated["event_library"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (target_dir / "student_event_library_notes.md").write_text(
        _build_event_library_notes_markdown(validated["design_notes"]),
        encoding="utf-8",
    )
    (target_dir / "student_event_library_payload.json").write_text(
        json.dumps(validated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return validated


def generate_student_sungrow_priors(
    output_dir: str | Path,
    report_cache_path: str | Path | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Generate a student-authored Sungrow event library and rules from the 2023 annual report."""
    report_path = get_report_path_for_month(0)
    if report_path is None:
        raise SchemaValidationError("Missing 2023 annual report path for student prior generation.")

    example_event_library = json.loads(Path(EVENT_LIBRARY_PATH).read_text(encoding="utf-8-sig"))
    example_rules = json.loads(Path(RULES_PATH).read_text(encoding="utf-8"))
    report_analysis = load_or_analyze_report_pdf(
        report_path,
        "2023FY",
        cache_path=report_cache_path,
        api_key=api_key,
    )
    report_text = extract_pdf_text(report_path)

    active_api_key = resolve_api_key(api_key, purpose="student prior induction")

    generated = _generate_with_deepseek(
        report_text=report_text,
        report_analysis=report_analysis,
        example_event_library=example_event_library,
        example_rules=example_rules,
        api_key=active_api_key,
    )
    validated = _validate_generated_payload(generated)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "student_event_library.json").write_text(
        json.dumps(validated["event_library"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (target_dir / "student_rules.json").write_text(
        json.dumps(validated["rules"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (target_dir / "student_rule_design_notes.md").write_text(
        _build_notes_markdown(validated["design_notes"]),
        encoding="utf-8",
    )
    (target_dir / "student_rule_design_payload.json").write_text(
        json.dumps(validated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return validated


def _generate_event_library_with_deepseek(
    report_text: str,
    report_analysis: dict[str, Any],
    example_event_library: dict[str, Any],
    api_key: str,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return chat_json(
                system_prompt=(
                    "You are a student analyst building a company-specific event taxonomy. "
                    "Return one valid JSON object only."
                ),
                user_prompt=_build_event_library_prompt(
                    report_text=report_text,
                    report_analysis=report_analysis,
                    example_event_library=example_event_library,
                    attempt=attempt,
                ),
                api_key=api_key,
                timeout=180,
                purpose="DeepSeek student event library request",
            )
        except (SchemaValidationError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            continue
    raise SchemaValidationError(f"DeepSeek student event library request failed after retries: {last_error}")


def _generate_with_deepseek(
    report_text: str,
    report_analysis: dict[str, Any],
    example_event_library: dict[str, Any],
    example_rules: list[dict[str, Any]],
    api_key: str,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return chat_json(
                system_prompt=(
                    "You are a student analyst building a company-specific event taxonomy and rule set. "
                    "Return one valid JSON object only."
                ),
                user_prompt=_build_prompt(
                    report_text=report_text,
                    report_analysis=report_analysis,
                    example_event_library=example_event_library,
                    example_rules=example_rules,
                    attempt=attempt,
                ),
                api_key=api_key,
                timeout=180,
                purpose="DeepSeek student rule induction request",
            )
        except (SchemaValidationError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            continue
    raise SchemaValidationError(f"DeepSeek student rule induction request failed after retries: {last_error}")


def _build_event_library_prompt(
    report_text: str,
    report_analysis: dict[str, Any],
    example_event_library: dict[str, Any],
    attempt: int = 0,
) -> str:
    raw_text_section = "" if attempt > 0 else f"\n\n2023 annual report extracted text (short excerpt):\n{report_text[:5000]}"
    return (
        "Read Sungrow's 2023 annual report materials and design a student-authored event library only.\n"
        "Use the example event library only as formatting reference.\n"
        "Do not copy it mechanically. Make the taxonomy more Sungrow-specific when justified by the annual report.\n"
        "The forecasting target is cumulative ROA in 2024, so prioritize scenarios that can affect margin or turnover.\n"
        "Return exactly one JSON object with keys event_library and design_notes.\n"
        "event_library must be an object of scenario -> {event_types: [..]}.\n"
        "design_notes must include keys summary, report_signals_used, differences_from_examples.\n"
        "Keep the event library to 5-8 scenarios and each scenario to 2-5 event_types.\n\n"
        f"Report analysis summary:\n{json.dumps(report_analysis, ensure_ascii=False, indent=2)}\n\n"
        f"Example event library:\n{json.dumps(example_event_library, ensure_ascii=False, indent=2)}\n"
        f"{raw_text_section}"
    )


def _build_prompt(
    report_text: str,
    report_analysis: dict[str, Any],
    example_event_library: dict[str, Any],
    example_rules: list[dict[str, Any]],
    attempt: int = 0,
) -> str:
    compact_rules = example_rules[:10]
    raw_text_section = "" if attempt > 0 else f"\n\n2023 annual report extracted text (short excerpt):\n{report_text[:6000]}"
    return (
        "Read Sungrow's 2023 annual report materials and design a student-authored event library and rule set.\n"
        "Use the example event library and example rules only as formatting and schema references.\n"
        "Do not copy them mechanically. Make the taxonomy more Sungrow-specific when justified by the annual report.\n"
        "The forecasting target is cumulative ROA in 2024.\n"
        "Rules must stay compatible with the existing engine.\n"
        "Allowed target_component values: profit_margin, asset_turnover.\n"
        "Allowed function_name: linear_adjustment.\n"
        "Each rule must include trigger, params.base_impact, explanation, and may include lag and duration.\n"
        "Event library format must be an object of scenario -> {event_types: [..]}.\n"
        "Return exactly one JSON object with keys event_library, rules, design_notes.\n"
        "design_notes must include keys summary, report_signals_used, differences_from_examples.\n"
        "Keep the event library to 5-8 scenarios and each scenario to 2-5 event_types.\n"
        "Keep the rules concise but usable: ideally one increase rule and one decrease rule for each event_type.\n"
        "Use reported_roa as the authoritative ROA field when mentioned in the report analysis.\n\n"
        f"Report analysis summary:\n{json.dumps(report_analysis, ensure_ascii=False, indent=2)}\n\n"
        f"Example event library:\n{json.dumps(example_event_library, ensure_ascii=False, indent=2)}\n\n"
        f"Example rules (subset for schema reference):\n{json.dumps(compact_rules, ensure_ascii=False, indent=2)}\n\n"
        f"{raw_text_section}"
    )


def _validate_generated_event_library_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required = {"event_library", "design_notes"}
    missing = required.difference(payload)
    if missing:
        raise SchemaValidationError(f"Student event library output missing keys: {sorted(missing)}")

    event_library = validate_event_library(payload["event_library"])
    design_notes = payload["design_notes"]
    if not isinstance(design_notes, dict):
        raise SchemaValidationError("design_notes must be an object.")
    required_notes = {"summary", "report_signals_used", "differences_from_examples"}
    missing_notes = required_notes.difference(design_notes)
    if missing_notes:
        raise SchemaValidationError(f"design_notes missing keys: {sorted(missing_notes)}")
    for key in required_notes:
        if not isinstance(design_notes[key], str) or not design_notes[key].strip():
            raise SchemaValidationError(f"design_notes.{key} must be a non-empty string.")

    return {
        "event_library": event_library,
        "design_notes": design_notes,
    }


def _validate_generated_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required = {"event_library", "rules", "design_notes"}
    missing = required.difference(payload)
    if missing:
        raise SchemaValidationError(f"Student rule induction output missing keys: {sorted(missing)}")

    event_library = validate_event_library(payload["event_library"])
    rules = validate_rules(payload["rules"])
    design_notes = payload["design_notes"]
    if not isinstance(design_notes, dict):
        raise SchemaValidationError("design_notes must be an object.")
    required_notes = {"summary", "report_signals_used", "differences_from_examples"}
    missing_notes = required_notes.difference(design_notes)
    if missing_notes:
        raise SchemaValidationError(f"design_notes missing keys: {sorted(missing_notes)}")
    for key in required_notes:
        if not isinstance(design_notes[key], str) or not design_notes[key].strip():
            raise SchemaValidationError(f"design_notes.{key} must be a non-empty string.")

    return {
        "event_library": event_library,
        "rules": rules,
        "design_notes": design_notes,
    }


def _build_notes_markdown(design_notes: dict[str, str]) -> str:
    return (
        "# Student Rule Design Notes\n\n"
        "## Summary\n"
        f"{design_notes['summary'].strip()}\n\n"
        "## Report Signals Used\n"
        f"{design_notes['report_signals_used'].strip()}\n\n"
        "## Differences From Examples\n"
        f"{design_notes['differences_from_examples'].strip()}\n"
    )


def _build_event_library_notes_markdown(design_notes: dict[str, str]) -> str:
    return (
        "# Student Event Library Notes\n\n"
        "## Summary\n"
        f"{design_notes['summary'].strip()}\n\n"
        "## Report Signals Used\n"
        f"{design_notes['report_signals_used'].strip()}\n\n"
        "## Differences From Examples\n"
        f"{design_notes['differences_from_examples'].strip()}\n"
    )
