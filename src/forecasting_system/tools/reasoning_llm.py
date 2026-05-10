"""DeepSeek-assisted reasoning analysis for rule update suggestions."""

from __future__ import annotations

import json
import time
from typing import Any

from forecasting_system.exceptions import SchemaValidationError
from forecasting_system.tools.deepseek_client import chat_completion, resolve_api_key
from forecasting_system.types import Rule, StudentRecord, TeacherFeedback


MAX_REASONING_RETRIES = 5
ALLOWED_ACTIONS = {"update_base_impact", "update_lag", "update_duration", "no_change"}


def analyze_rule_updates(
    student_record: StudentRecord,
    teacher_feedback: TeacherFeedback,
    matched_rules: list[Rule],
    api_key: str | None = None,
    max_retries: int = MAX_REASONING_RETRIES,
) -> dict[str, Any]:
    """Ask DeepSeek for structured, JSON-only rule update suggestions."""
    active_api_key = resolve_api_key(api_key, purpose="rule update reasoning")
    prompt = _build_reasoning_prompt(student_record, teacher_feedback, matched_rules)
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response_payload = _post_chat_completion(prompt, active_api_key)
            suggestions = _parse_reasoning_response(response_payload)
            validate_reasoning_suggestions(suggestions)
            return suggestions
        except (json.JSONDecodeError, KeyError, TypeError, SchemaValidationError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
            prompt = _build_retry_prompt(student_record, teacher_feedback, matched_rules, attempt + 1)

    raise SchemaValidationError(f"DeepSeek reasoning failed after {max_retries} attempts: {last_error}")


def validate_reasoning_suggestions(suggestions: dict[str, Any]) -> dict[str, Any]:
    """Validate the strict reasoning output schema before Python applies updates."""
    if not isinstance(suggestions, dict):
        raise SchemaValidationError("Reasoning suggestions must be a JSON object.")
    required = {"analysis_summary", "rule_update_suggestions"}
    missing = required.difference(suggestions)
    if missing:
        raise SchemaValidationError(f"Reasoning suggestions are missing keys: {sorted(missing)}")
    extra = set(suggestions).difference(required)
    if extra:
        raise SchemaValidationError(f"Reasoning suggestions contain unexpected keys: {sorted(extra)}")
    if not isinstance(suggestions["analysis_summary"], str) or not suggestions["analysis_summary"].strip():
        raise SchemaValidationError("analysis_summary must be a non-empty string.")
    if not isinstance(suggestions["rule_update_suggestions"], list):
        raise SchemaValidationError("rule_update_suggestions must be a list.")

    for item in suggestions["rule_update_suggestions"]:
        _validate_one_suggestion(item)
    return suggestions


def _validate_one_suggestion(item: Any) -> None:
    if not isinstance(item, dict):
        raise SchemaValidationError("Each rule_update_suggestions item must be an object.")
    required = {"rule_id", "action", "new_base_impact", "new_lag", "new_duration", "reason"}
    missing = required.difference(item)
    if missing:
        raise SchemaValidationError(f"Rule update suggestion is missing keys: {sorted(missing)}")
    extra = set(item).difference(required)
    if extra:
        raise SchemaValidationError(f"Rule update suggestion contains unexpected keys: {sorted(extra)}")
    if not isinstance(item["rule_id"], str) or not item["rule_id"].strip():
        raise SchemaValidationError("rule_id must be a non-empty string.")
    if item["action"] not in ALLOWED_ACTIONS:
        raise SchemaValidationError(f"Unsupported rule update action: {item['action']!r}")
    if item["new_base_impact"] is not None and not isinstance(item["new_base_impact"], (int, float)):
        raise SchemaValidationError("new_base_impact must be numeric or null.")
    if not isinstance(item["reason"], str) or not item["reason"].strip():
        raise SchemaValidationError("reason must be a non-empty string.")
    if item["new_lag"] is not None and (not isinstance(item["new_lag"], int) or item["new_lag"] < 0):
        raise SchemaValidationError("new_lag must be a non-negative integer or null.")
    if item["new_duration"] is not None and (not isinstance(item["new_duration"], int) or item["new_duration"] < 0):
        raise SchemaValidationError("new_duration must be a non-negative integer or null.")


def _build_reasoning_prompt(
    student_record: StudentRecord,
    teacher_feedback: TeacherFeedback,
    matched_rules: list[Rule],
) -> str:
    context = {
        "student_record": _strip_em_context(student_record),
        "teacher_feedback": _strip_em_context(teacher_feedback),
        "current_matched_rules": _strip_em_context(matched_rules),
    }
    return (
        "You are the Reasoning Agent for an auditable financial forecasting system.\n"
        "Return JSON only. Do not include markdown or free text.\n"
        "Analyze why the Student prediction was wrong or reasonable using only the provided StudentRecord, "
        "TeacherFeedback, and current matched rules.\n"
        "The target metric is ROA, not ROE.\n"
        "Do not create rules. Do not delete rules. Do not change function_name or target_component.\n"
        "Only suggest updates for existing matched rule_ids.\n"
        "Allowed update fields are params.base_impact, lag, and duration.\n"
        "base_impact must be numeric. lag/duration must be non-negative integers.\n"
        "Strict parameter discipline for params.base_impact:\n"
        "- Never change base_impact to 0.\n"
        "- Never change the sign of base_impact. Positive rules must stay positive; negative rules must stay negative.\n"
        "- If TeacherFeedback.supervision_snapshot.actual_roa is null, the evidence is monthly export supervision: "
        "a single update may change base_impact by at most 10% versus the current value.\n"
        "- If TeacherFeedback.supervision_snapshot.actual_roa is not null, the evidence is disclosed quarterly, "
        "semi-annual, or annual ROA and is the most authoritative supervision variable: a single update may change "
        "base_impact by at most 30% versus the current value.\n"
        "- For monthly export supervision and a positive current base_impact, new_base_impact must stay within "
        "[current*0.90, current*1.10]. For report ROA supervision, it must stay within [current*0.70, current*1.30].\n"
        "- For monthly export supervision and a negative current base_impact, new_base_impact must stay within "
        "[current*1.10, current*0.90]. For report ROA supervision, it must stay within [current*1.30, current*0.70].\n"
        "- Prefer smaller changes when the evidence is only one monthly Teacher signal; use the wider 30% range only "
        "when disclosed actual ROA shows a large gap from the prediction.\n"
        "Reasoning discipline:\n"
        "- Before changing any parameter, decide whether the error is mainly an impact magnitude problem, a lag timing problem, or a duration persistence problem.\n"
        "- If news direction is economically correct but appears too early or too late, update lag or duration instead of base_impact.\n"
        "- If the issue is only magnitude, update base_impact within the applicable 10% or 30% bound and preserve its sign.\n"
        "- When actual_roa is present, treat it as more authoritative than monthly export direction. If prediction and actual ROA differ materially, "
        "inspect the rules used during that quarter and adjust their impacts more decisively, while preserving sign and avoiding zero.\n"
        "- Read any fundamental_analyst_report, latest_fundamental_analyst_report, student_fundamental_report, "
        "latest_student_fundamental_report, or quarter_report_analysis included in the StudentRecord context. "
        "Use those fundamentals to decide whether the mismatch is from impact magnitude, timing lag, or duration persistence.\n"
        "- In each reason, explicitly state why impact, lag, or duration is the right field to change.\n"
        "Do not add new research variables or redesign the methodology.\n"
        "If no rule change is needed, return an empty rule_update_suggestions list.\n"
        "Required output schema:\n"
        "{\n"
        '  "analysis_summary": "string",\n'
        '  "rule_update_suggestions": [\n'
        "    {\n"
        '      "rule_id": "string",\n'
        '      "action": "update_base_impact" | "update_lag" | "update_duration" | "no_change",\n'
        '      "new_base_impact": number or null,\n'
        '      "new_lag": integer or null,\n'
        '      "new_duration": integer or null,\n'
        '      "reason": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Input context:\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}"
    )


def _strip_em_context(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_em_context(item)
            for key, item in value.items()
            if not _is_em_key(str(key))
        }
    if isinstance(value, list):
        return [_strip_em_context(item) for item in value]
    if isinstance(value, str):
        return (
            value.replace("equity_multiplier", "[removed_leverage_metric]")
            .replace("Equity Multiplier", "[removed leverage metric]")
            .replace("equity multiplier", "[removed leverage metric]")
            .replace("权益乘数", "[removed leverage metric]")
            .replace(" EM ", " [removed leverage metric] ")
        )
    return value


def _is_em_key(key: str) -> bool:
    lowered = key.lower()
    return "equity_multiplier" in lowered or lowered in {"em", "equity_multiplier"}


def _build_retry_prompt(
    student_record: StudentRecord,
    teacher_feedback: TeacherFeedback,
    matched_rules: list[Rule],
    attempt_number: int,
) -> str:
    return (
        f"Retry attempt {attempt_number}. Return strict JSON only with keys analysis_summary and "
        "rule_update_suggestions. Each suggestion must include rule_id, action, new_base_impact, "
        "new_lag, new_duration, and reason. Actions are update_base_impact, "
        "update_lag, update_duration, or no_change. "
        "Do not create/delete rules or update unmatched rule_ids.\n"
        + _build_reasoning_prompt(student_record, teacher_feedback, matched_rules)
    )


def _post_chat_completion(prompt: str, api_key: str) -> dict:
    return chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "You are a rule-update reasoning service. "
                    "Return one JSON object only and never write files."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        api_key=api_key,
        response_format={"type": "json_object"},
        timeout=120,
        purpose="DeepSeek reasoning request",
    )


def _parse_reasoning_response(response_payload: dict) -> dict[str, Any]:
    content = response_payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise SchemaValidationError("DeepSeek reasoning response must decode to a JSON object.")
    return parsed
