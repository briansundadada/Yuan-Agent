"""Global DeepSeek-assisted reasoning for quarterly rule calibration."""

from __future__ import annotations

import json
import time
from typing import Any

from forecasting_system.exceptions import SchemaValidationError
from forecasting_system.tools.deepseek_client import chat_completion, resolve_api_key


GLOBAL_REASONING_MODEL = "deepseek-v4-pro"
MAX_GLOBAL_REASONING_RETRIES = 4
ALLOWED_UPDATE_FIELDS = {
    "target_component",
    "params.base_impact",
    "lag",
    "duration",
    "explanation",
    "business_chain",
    "fundamental_basis",
    "operating_metric_links",
    "component_impacts",
}
ALLOWED_CYCLE_STATES = {
    "early_recovery",
    "demand_expansion",
    "capacity_expansion",
    "margin_pressure",
    "high_growth_operating_leverage",
    "working_capital_stress",
    "mature_normalization",
    "downturn_shock",
}


def analyze_global_rule_updates(
    *,
    pass_id: int,
    total_passes: int,
    training_context: dict[str, Any],
    active_rules: list[dict[str, Any]],
    api_key: str | None = None,
    max_retries: int = MAX_GLOBAL_REASONING_RETRIES,
) -> dict[str, Any]:
    """Ask DeepSeek v4 pro for one global rule calibration pass."""
    active_api_key = resolve_api_key(api_key, purpose="global quarterly rule reasoning")
    prompt = _build_global_reasoning_prompt(
        pass_id=pass_id,
        total_passes=total_passes,
        training_context=training_context,
        active_rules=active_rules,
    )
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response_payload = _post_chat_completion(prompt, active_api_key)
            suggestions = _parse_response(response_payload)
            return validate_global_reasoning_suggestions(suggestions)
        except (json.JSONDecodeError, KeyError, TypeError, SchemaValidationError) as exc:
            last_error = exc
            time.sleep(2.0 * (attempt + 1))
            prompt = (
                f"Retry attempt {attempt + 1}. Return strict JSON only. "
                "Do not modify trigger.scenario, trigger.event_type, trigger.direction, rule_id, or function_name. "
                "Do not add or delete rules.\n"
                + _build_global_reasoning_prompt(
                    pass_id=pass_id,
                    total_passes=total_passes,
                    training_context=training_context,
                    active_rules=active_rules,
                )
            )

    raise SchemaValidationError(f"DeepSeek global reasoning failed after retries: {last_error}")


def validate_global_reasoning_suggestions(suggestions: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(suggestions, dict):
        raise SchemaValidationError("Global reasoning response must be a JSON object.")
    required = {
        "analysis_summary",
        "global_error_diagnosis",
        "rule_update_suggestions",
        "period_cycle_states",
        "event_strength_overrides",
        "cycle_weight_suggestions",
        "experimental_design_recommendations",
    }
    missing = required.difference(suggestions)
    if missing:
        raise SchemaValidationError(f"Global reasoning response missing keys: {sorted(missing)}")
    extra = set(suggestions).difference(required)
    if extra:
        raise SchemaValidationError(f"Global reasoning response contains unexpected keys: {sorted(extra)}")
    for key in ("analysis_summary", "global_error_diagnosis"):
        if not isinstance(suggestions[key], str) or not suggestions[key].strip():
            raise SchemaValidationError(f"{key} must be a non-empty string.")
    if not isinstance(suggestions["rule_update_suggestions"], list):
        raise SchemaValidationError("rule_update_suggestions must be a list.")
    for key in ("period_cycle_states", "event_strength_overrides", "cycle_weight_suggestions"):
        if not isinstance(suggestions[key], list):
            raise SchemaValidationError(f"{key} must be a list.")
    if not isinstance(suggestions["experimental_design_recommendations"], list):
        raise SchemaValidationError("experimental_design_recommendations must be a list.")

    for item in suggestions["rule_update_suggestions"]:
        _validate_update_item(item)
    for item in suggestions["period_cycle_states"]:
        _validate_cycle_state_item(item)
    for item in suggestions["event_strength_overrides"]:
        _validate_strength_override_item(item)
    for item in suggestions["cycle_weight_suggestions"]:
        _validate_cycle_weight_item(item)
    for item in suggestions["experimental_design_recommendations"]:
        if not isinstance(item, str) or not item.strip():
            raise SchemaValidationError("Each experimental design recommendation must be a non-empty string.")
    return suggestions


def _validate_update_item(item: Any) -> None:
    if not isinstance(item, dict):
        raise SchemaValidationError("Each rule update suggestion must be an object.")
    required = {"rule_id", "field", "new_value", "reason"}
    missing = required.difference(item)
    if missing:
        raise SchemaValidationError(f"Rule update suggestion missing keys: {sorted(missing)}")
    extra = set(item).difference(required)
    if extra:
        raise SchemaValidationError(f"Rule update suggestion contains unexpected keys: {sorted(extra)}")
    if not isinstance(item["rule_id"], str) or not item["rule_id"].strip():
        raise SchemaValidationError("rule_id must be a non-empty string.")
    if item["field"] not in ALLOWED_UPDATE_FIELDS:
        raise SchemaValidationError(f"Unsupported global reasoning field: {item['field']!r}")
    if not isinstance(item["reason"], str) or not item["reason"].strip():
        raise SchemaValidationError("reason must be a non-empty string.")


def _validate_cycle_state_item(item: Any) -> None:
    if not isinstance(item, dict):
        raise SchemaValidationError("Each period_cycle_states item must be an object.")
    required = {"period", "primary_cycle_state", "secondary_cycle_state", "evidence"}
    missing = required.difference(item)
    if missing:
        raise SchemaValidationError(f"Cycle state item missing keys: {sorted(missing)}")
    if item["primary_cycle_state"] not in ALLOWED_CYCLE_STATES:
        raise SchemaValidationError(f"Unsupported primary cycle state: {item['primary_cycle_state']!r}")
    if item["secondary_cycle_state"] is not None and item["secondary_cycle_state"] not in ALLOWED_CYCLE_STATES:
        raise SchemaValidationError(f"Unsupported secondary cycle state: {item['secondary_cycle_state']!r}")


def _validate_strength_override_item(item: Any) -> None:
    if not isinstance(item, dict):
        raise SchemaValidationError("Each event_strength_overrides item must be an object.")
    required = {"period", "event_key", "canonical_news", "new_strength", "reason"}
    missing = required.difference(item)
    if missing:
        raise SchemaValidationError(f"Strength override item missing keys: {sorted(missing)}")
    if not isinstance(item["new_strength"], (int, float)) or float(item["new_strength"]) <= 0:
        raise SchemaValidationError("new_strength must be a positive number.")
    if float(item["new_strength"]) > 6:
        raise SchemaValidationError("new_strength may not exceed 6.")


def _validate_cycle_weight_item(item: Any) -> None:
    if not isinstance(item, dict):
        raise SchemaValidationError("Each cycle_weight_suggestions item must be an object.")
    required = {"period", "scope", "key", "weight", "reason"}
    missing = required.difference(item)
    if missing:
        raise SchemaValidationError(f"Cycle weight item missing keys: {sorted(missing)}")
    if item["scope"] not in {"rule", "rule_component", "scenario", "default"}:
        raise SchemaValidationError(f"Unsupported cycle weight scope: {item['scope']!r}")
    if not isinstance(item["weight"], (int, float)) or not 0.3 <= float(item["weight"]) <= 2.0:
        raise SchemaValidationError("cycle weight must be between 0.3 and 2.0.")


def _build_global_reasoning_prompt(
    *,
    pass_id: int,
    total_passes: int,
    training_context: dict[str, Any],
    active_rules: list[dict[str, Any]],
) -> str:
    compact_rules = [
        {
            "rule_id": rule.get("rule_id"),
            "trigger": rule.get("trigger"),
            "target_component": rule.get("target_component"),
            "function_name": rule.get("function_name"),
            "params": rule.get("params"),
            "component_impacts": rule.get("component_impacts"),
            "lag": rule.get("lag"),
            "duration": rule.get("duration"),
            "explanation": rule.get("explanation"),
            "business_chain": rule.get("business_chain"),
            "fundamental_basis": rule.get("fundamental_basis"),
            "operating_metric_links": rule.get("operating_metric_links"),
        }
        for rule in active_rules
    ]
    context = {
        "pass_id": pass_id,
        "total_passes": total_passes,
        "training_context": _strip_em_context(training_context),
        "active_rules": _strip_em_context(compact_rules),
    }
    return (
        "You are the Global Reasoning Agent for an auditable quarterly ROA forecasting experiment.\n"
        "Return JSON only. Do not include markdown or free text.\n"
        "You have a full training-set view over 2021-2023 quarterly reported ROA, operating components, "
        "Student predictions, Teacher labels, component-level Teacher feedback, MSE metrics, rule hit attribution, "
        "canonical/raw event names, and Fundamental Analyst report summaries.\n"
        "Act like a top-tier equity analyst and model auditor: explain concrete error patterns, connect them "
        "to business fundamentals, and modify existing rules only when the evidence supports the change.\n"
        "\nHard constraints:\n"
        "- Do not add rules. Do not delete rules.\n"
        "- Do not change rule_id or function_name.\n"
        "- Do not change trigger.scenario, trigger.event_type, or trigger.direction.\n"
        "- Allowed fields are target_component, params.base_impact, component_impacts, lag, duration, explanation, "
        "business_chain, fundamental_basis, and operating_metric_links.\n"
        "- params.base_impact must never become 0 and must never change sign.\n"
        "- Keep base_impact within the existing report-supervision bound: at most 30% up or down from "
        "the current value in this pass. For positive values use [current*0.70, current*1.30]; "
        "for negative values use [current*1.30, current*0.70].\n"
        "- lag and duration must be non-negative integers.\n"
        "- target_component must be one of profit_margin or asset_turnover.\n"
        "- operating_metric_links must be a list of concise strings.\n"
        "- component_impacts must be a list of {target_component, base_impact}; it may split one rule across "
        "profit_margin and asset_turnover. Keep component-specific magnitudes economically scaled.\n"
        "- period_cycle_states primary/secondary states must use only: early_recovery, demand_expansion, "
        "capacity_expansion, margin_pressure, high_growth_operating_leverage, working_capital_stress, "
        "mature_normalization, downturn_shock.\n"
        "- event_strength_overrides may set strength above 1 only for clearly major or surprising events. "
        "Use 1-2 for major, 2-4 for very major, 4-6 only for extreme shocks with explicit evidence.\n"
        "- cycle_weight_suggestions enter Student computation as base_impact * relevance * strength * cycle_weight. "
        "Use 0.5-1.5 normally; 0.3-2.0 only for strong cycle evidence.\n"
        "\nReasoning discipline:\n"
        "- Use all quarters together. Do not overfit one quarter when the same rule behaves well elsewhere.\n"
        "- Read the updated Student/Teacher outputs carefully: quarter_error_table includes baseline_state, "
        "student_predicted_state, actual_state, component_feedback, evaluation_metrics, and primary_error_driver.\n"
        "- Treat teacher_component_mse and teacher_primary_error_driver_counts as central diagnostics. "
        "Do not improve ROA by making profit_margin or asset_turnover structurally worse.\n"
        "- For each rule edit, explain which supervised variable it is intended to improve: ROA, profit_margin, "
        "asset_turnover, or the interaction between them.\n"
        "- Every rule edit must name its error_driver and first diagnose whether the current target_component is correct. "
        "If it is correct, state whether impact, lag, or duration is the problem.\n"
        "- Pay attention to canonical_news, raw_news, duplicate_news, and event_key in rule_attribution_table. "
        "Use original event names to distinguish large commercial events from generic sentiment or repeated news.\n"
        "- Before changing base_impact, decide whether the issue is magnitude, timing lag, duration persistence, "
        "wrong target_component, or incomplete business_chain.\n"
        "- Reasons must cite specific periods and the direction of the error, for example 2021Q3/2022Q3 "
        "underprediction or 2023H1 overprediction.\n"
        "- Use Fundamental Analyst reports and derived states to distinguish profit_margin and asset_turnover channels.\n"
        "- Prefer a small number of high-confidence rule edits. Leave uncertain rules unchanged.\n"
        "- On the final pass, include experimental_design_recommendations for improving the method. "
        "On non-final passes, return [] unless a design issue is already obvious.\n"
        "\nRequired output schema:\n"
        "{\n"
        '  "analysis_summary": "string",\n'
        '  "global_error_diagnosis": "string",\n'
        '  "rule_update_suggestions": [\n'
        "    {\n"
        '      "rule_id": "string",\n'
        '      "field": "target_component" | "params.base_impact" | "component_impacts" | "lag" | "duration" | '
        '"explanation" | "business_chain" | "fundamental_basis" | "operating_metric_links",\n'
        '      "new_value": "string, number, integer, or string list depending on field",\n'
        '      "reason": "specific analyst-grade attribution with periods, error_driver, target-component diagnosis, and why this field"\n'
        "    }\n"
        "  ],\n"
        '  "period_cycle_states": [\n'
        '    {"period": "string", "primary_cycle_state": "string", "secondary_cycle_state": "string or null", "evidence": "string"}\n'
        "  ],\n"
        '  "event_strength_overrides": [\n'
        '    {"period": "string", "event_key": "string", "canonical_news": "string", "new_strength": number, "reason": "string"}\n'
        "  ],\n"
        '  "cycle_weight_suggestions": [\n'
        '    {"period": "string", "scope": "rule|rule_component|scenario|default", "key": "string", "weight": number, "reason": "string"}\n'
        "  ],\n"
        '  "experimental_design_recommendations": ["string"]\n'
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


def _post_chat_completion(prompt: str, api_key: str) -> dict[str, Any]:
    return chat_completion(
        [
            {
                "role": "system",
                "content": "You are a strict JSON rule-calibration service. Return one JSON object only.",
            },
            {"role": "user", "content": prompt},
        ],
        api_key=api_key,
        model=GLOBAL_REASONING_MODEL,
        response_format={"type": "json_object"},
        timeout=180,
        purpose="DeepSeek global quarterly reasoning request",
    )


def _parse_response(response_payload: dict[str, Any]) -> dict[str, Any]:
    content = response_payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise SchemaValidationError("DeepSeek global reasoning response must decode to a JSON object.")
    return parsed
