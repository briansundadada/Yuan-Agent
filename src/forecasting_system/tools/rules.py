"""Rule loading and matching tools."""

from __future__ import annotations

import json
from pathlib import Path

from forecasting_system.config import RULES_PATH
from forecasting_system.exceptions import SchemaValidationError
from forecasting_system.types import Event, Rule


def load_rules(path: str | Path = RULES_PATH) -> list[Rule]:
    """Load structured rules from the local rules file."""
    file_path = Path(path)
    data = json.loads(file_path.read_text(encoding="utf-8-sig"))
    return validate_rules(data)


def validate_rules(data: object) -> list[Rule]:
    """Validate one in-memory rules payload."""
    if not isinstance(data, list):
        raise SchemaValidationError("rules.json must contain a list of rules.")
    for rule in data:
        _validate_phase1_rule(rule)
    return data


def match_rules_for_event(event: Event, rules: list[Rule] | None = None) -> list[Rule]:
    """Match triggered rules for one event."""
    if rules is None:
        rules = load_rules()
    return [
        rule
        for rule in rules
        if rule["trigger"]["scenario"] == event["scenario"]
        and rule["trigger"]["event_type"] == event["event_type"]
        and rule["trigger"]["direction"] == event["direction"]
    ]


def update_rule_base_impact(rule: Rule, multiplier: float) -> float:
    """Return an updated base_impact using a deterministic multiplier."""
    current_base_impact = float(rule["params"]["base_impact"])
    return current_base_impact * float(multiplier)


def _validate_phase1_rule(rule: dict) -> None:
    required_keys = {
        "rule_id",
        "trigger",
        "target_component",
        "function_name",
        "params",
        "explanation",
    }
    missing = required_keys.difference(rule)
    if missing:
        raise SchemaValidationError(f"Rule is missing required keys: {sorted(missing)}")
    if rule["target_component"] not in {"profit_margin", "asset_turnover"}:
        raise SchemaValidationError(
            "Rules may only target profit_margin or asset_turnover."
        )
    if rule["function_name"] != "linear_adjustment":
        raise SchemaValidationError("Phase 1 rules may only use linear_adjustment.")
    if "base_impact" not in rule["params"]:
        raise SchemaValidationError("Rule params must include base_impact.")
    if "component_impacts" in rule:
        if not isinstance(rule["component_impacts"], list) or not rule["component_impacts"]:
            raise SchemaValidationError("component_impacts must be a non-empty list when present.")
        for item in rule["component_impacts"]:
            if not isinstance(item, dict):
                raise SchemaValidationError("Each component_impacts item must be an object.")
            if item.get("target_component") not in {"profit_margin", "asset_turnover"}:
                raise SchemaValidationError("component_impacts target_component is invalid.")
            if not isinstance(item.get("base_impact"), (int, float)):
                raise SchemaValidationError("component_impacts base_impact must be numeric.")
    if "lag" in rule and (not isinstance(rule["lag"], int) or rule["lag"] < 0):
        raise SchemaValidationError("Rule lag must be a non-negative integer when present.")
    if "duration" in rule and (not isinstance(rule["duration"], int) or rule["duration"] < 0):
        raise SchemaValidationError("Rule duration must be a non-negative integer when present.")
