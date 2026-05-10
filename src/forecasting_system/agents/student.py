"""Student Agent implementation for phase 1."""

from __future__ import annotations

from forecasting_system.config import TARGET_METRIC
from forecasting_system.exceptions import SchemaValidationError
from forecasting_system.tools.financial_model import (
    apply_factor_update,
    compute_roa,
    linear_adjustment,
)
from forecasting_system.tools.rules import match_rules_for_event
from forecasting_system.types import BaselineState, Event, Rule, StudentRecord


class Student:
    """Phase-1 explicit prediction orchestration with tool-only computations."""

    def run(self, events: list[Event], baseline_state: BaselineState, rules: list[Rule]) -> StudentRecord:
        _validate_baseline_state(baseline_state)

        current_profit_margin = float(baseline_state["profit_margin"])
        current_asset_turnover = float(baseline_state["asset_turnover"])
        delta_profit_margin = 0.0
        delta_asset_turnover = 0.0

        matched_rules_record = []
        function_calls = []

        for event in events:
            if event.get("noise", False):
                continue
            event_rules = match_rules_for_event(event, rules)
            active_rule_ids = event.get("active_rule_ids")
            if active_rule_ids is not None:
                allowed_rule_ids = set(active_rule_ids)
                event_rules = [rule for rule in event_rules if rule["rule_id"] in allowed_rule_ids]

            for rule in event_rules:
                if rule["function_name"] != "linear_adjustment":
                    raise SchemaValidationError(
                        f"Unsupported function_name for phase 1 student execution: {rule['function_name']}"
                    )

                component_impacts = _component_impacts_for_rule(rule)
                for component_impact in component_impacts:
                    base_impact = float(component_impact["base_impact"])
                    target_component = component_impact["target_component"]
                    cycle_weight = _cycle_weight_for_event(event, rule, target_component)
                    effective_strength = float(event["strength"]) * cycle_weight
                    delta = linear_adjustment(base_impact, float(event["relevance"]), effective_strength)
                    function_calls.append(
                        {
                            "function_name": "linear_adjustment",
                            "inputs": {
                                "base_impact": base_impact,
                                "relevance": float(event["relevance"]),
                                "strength": float(event["strength"]),
                                "cycle_weight": cycle_weight,
                                "effective_strength": effective_strength,
                            },
                            "output": delta,
                        }
                    )

                    if target_component == "profit_margin":
                        current_profit_margin = apply_factor_update(current_profit_margin, delta)
                        delta_profit_margin += delta
                        updated_value = current_profit_margin
                    elif target_component == "asset_turnover":
                        current_asset_turnover = apply_factor_update(current_asset_turnover, delta)
                        delta_asset_turnover += delta
                        updated_value = current_asset_turnover
                    else:
                        raise SchemaValidationError(
                            f"Unsupported target_component for phase 1 student execution: {target_component}"
                        )

                    function_calls.append(
                        {
                            "function_name": "apply_factor_update",
                            "inputs": {
                                "target_component": target_component,
                                "current_value": updated_value - delta,
                                "delta": delta,
                            },
                            "output": updated_value,
                        }
                    )
                    matched_rules_record.append(
                        {
                            "rule_id": rule["rule_id"],
                            "target_component": target_component,
                            "function_name": rule["function_name"],
                            "inputs": {
                                "event": event,
                                "base_impact": base_impact,
                                "relevance": float(event["relevance"]),
                                "strength": float(event["strength"]),
                                "cycle_weight": cycle_weight,
                                "effective_strength": effective_strength,
                            },
                            "delta": delta,
                        }
                    )

        roa = compute_roa(current_profit_margin, current_asset_turnover)
        function_calls.append(
            {
                "function_name": "compute_roa",
                "inputs": {
                    "profit_margin": current_profit_margin,
                    "asset_turnover": current_asset_turnover,
                },
                "output": roa,
            }
        )

        record: StudentRecord = {
            "metric": TARGET_METRIC,
            "input_events": events,
            "baseline_state": {
                "profit_margin": float(baseline_state["profit_margin"]),
                "asset_turnover": float(baseline_state["asset_turnover"]),
                "equity_multiplier": float(baseline_state["equity_multiplier"]),
            },
            "matched_rules": matched_rules_record,
            "function_calls": function_calls,
            "intermediate_values": {
                "baseline_profit_margin": float(baseline_state["profit_margin"]),
                "baseline_asset_turnover": float(baseline_state["asset_turnover"]),
                "baseline_equity_multiplier": float(baseline_state["equity_multiplier"]),
                "delta_profit_margin": delta_profit_margin,
                "delta_asset_turnover": delta_asset_turnover,
                "updated_profit_margin": current_profit_margin,
                "updated_asset_turnover": current_asset_turnover,
                "updated_equity_multiplier": float(baseline_state["equity_multiplier"]),
            },
            "final_prediction": {
                "roa": roa,
            },
        }
        _validate_student_record_shape(record)
        return record


def _validate_baseline_state(baseline_state: BaselineState) -> None:
    required_keys = {"profit_margin", "asset_turnover", "equity_multiplier"}
    missing = required_keys.difference(baseline_state)
    if missing:
        raise SchemaValidationError(f"baseline_state is missing required keys: {sorted(missing)}")


def _component_impacts_for_rule(rule: Rule) -> list[dict]:
    component_impacts = rule.get("component_impacts")
    if component_impacts:
        return [
            {
                "target_component": item["target_component"],
                "base_impact": float(item["base_impact"]),
            }
            for item in component_impacts
        ]
    return [
        {
            "target_component": rule["target_component"],
            "base_impact": float(rule["params"]["base_impact"]),
        }
    ]


def _cycle_weight_for_event(event: Event, rule: Rule, target_component: str) -> float:
    weights = event.get("cycle_weights") or {}
    candidates = [
        f"{rule['rule_id']}::{target_component}",
        f"{rule['rule_id']}:{target_component}",
        rule["rule_id"],
        f"{event.get('scenario')}::{target_component}",
        f"{event.get('scenario')}:{target_component}",
        str(event.get("scenario")),
        "default",
    ]
    for key in candidates:
        value = weights.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 1.0


def _validate_student_record_shape(record: StudentRecord) -> None:
    required_keys = {
        "metric",
        "input_events",
        "baseline_state",
        "matched_rules",
        "function_calls",
        "intermediate_values",
        "final_prediction",
    }
    missing = required_keys.difference(record)
    if missing:
        raise SchemaValidationError(f"student record is missing required keys: {sorted(missing)}")
    if record["metric"] != "roa":
        raise SchemaValidationError("student record metric must be 'roa' in this pipeline.")
    baseline_keys = {"profit_margin", "asset_turnover", "equity_multiplier"}
    if baseline_keys.difference(record["baseline_state"]):
        raise SchemaValidationError("baseline_state must include profit_margin, asset_turnover, and equity_multiplier.")
    if "roa" not in record["final_prediction"]:
        raise SchemaValidationError("final_prediction must include roa.")
