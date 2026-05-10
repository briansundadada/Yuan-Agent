"""Reasoning Agent implementation with LLM-assisted rule suggestions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from forecasting_system.config import RULES_PATH
from forecasting_system.exceptions import SchemaValidationError
from forecasting_system.tools.reasoning_llm import analyze_rule_updates
from forecasting_system.tools.rules import load_rules
from forecasting_system.types import Rule, RuleUpdateRecord, StudentRecord, TeacherFeedback


ReasoningAnalysisFn = Callable[[StudentRecord, TeacherFeedback, list[Rule]], dict]


class Reasoning:
    """Apply validated LLM rule-update suggestions to the local rules library."""

    def __init__(
        self,
        rules_path: str | Path = RULES_PATH,
        analysis_fn: ReasoningAnalysisFn = analyze_rule_updates,
    ) -> None:
        self.rules_path = Path(rules_path)
        self.analysis_fn = analysis_fn

    def run(
        self,
        student_record: StudentRecord,
        teacher_feedback: TeacherFeedback,
        rules: list[Rule],
        source_month: int | None = None,
    ) -> RuleUpdateRecord:
        _validate_inputs(student_record, teacher_feedback)

        file_rules = load_rules(self.rules_path)
        matched_rule_ids = {matched_rule["rule_id"] for matched_rule in student_record["matched_rules"]}
        if not matched_rule_ids:
            return {
                "updated_rules": file_rules,
                "rule_update_records": [],
                "explanation": "No rule changes were applied because no rules were used by the Student.",
            }

        matched_file_rules = [rule for rule in file_rules if rule["rule_id"] in matched_rule_ids]
        missing_rule_ids = matched_rule_ids.difference({rule["rule_id"] for rule in matched_file_rules})
        if missing_rule_ids:
            raise SchemaValidationError(
                f"Matched rule_ids are not present in the local rules library: {sorted(missing_rule_ids)}"
            )

        if teacher_feedback["evaluation_label"] == "reasonable":
            return {
                "updated_rules": file_rules,
                "rule_update_records": [],
                "explanation": "No rule changes were applied because Teacher feedback was reasonable.",
            }

        suggestions = self.analysis_fn(student_record, teacher_feedback, matched_file_rules)
        updated_rules, rule_update_records = _apply_validated_suggestions(
            file_rules=file_rules,
            matched_rule_ids=matched_rule_ids,
            suggestions=suggestions,
            source_month=source_month,
            teacher_feedback=teacher_feedback,
            rule_version=_next_rule_version(self.rules_path),
        )

        if rule_update_records:
            _write_rules(self.rules_path, updated_rules)
            _write_rule_version_record(
                self.rules_path,
                {
                    "version": rule_update_records[0]["rule_version"],
                    "source_month": source_month,
                    "teacher_label": teacher_feedback["evaluation_label"],
                    "error_type": teacher_feedback["error_type"],
                    "matched_rule_ids": sorted(matched_rule_ids),
                    "updates": rule_update_records,
                    "explanation": suggestions["analysis_summary"],
                },
            )

        return {
            "updated_rules": updated_rules,
            "rule_update_records": rule_update_records,
            "explanation": suggestions["analysis_summary"],
        }


def _apply_validated_suggestions(
    file_rules: list[Rule],
    matched_rule_ids: set[str],
    suggestions: dict,
    source_month: int | None = None,
    teacher_feedback: TeacherFeedback | None = None,
    rule_version: int | None = None,
) -> tuple[list[Rule], list[dict]]:
    _validate_suggestions_against_rules(suggestions, file_rules, matched_rule_ids)

    rules_by_id = {rule["rule_id"]: rule for rule in file_rules}
    rule_update_records = []

    for suggestion in suggestions["rule_update_suggestions"]:
        action = suggestion["action"]
        if action == "no_change":
            continue

        rule = rules_by_id[suggestion["rule_id"]]
        if action == "update_base_impact":
            previous_value = float(rule["params"]["base_impact"])
            new_value = float(suggestion["new_base_impact"])
            if previous_value == new_value:
                continue
            rule["params"]["base_impact"] = new_value
            rule_update_records.append(
                {
                    "rule_id": rule["rule_id"],
                    "field": "params.base_impact",
                    "previous_value": previous_value,
                    "new_value": new_value,
                    "source_month": source_month,
                    "rule_version": rule_version,
                    "teacher_label": None if teacher_feedback is None else teacher_feedback["evaluation_label"],
                    "error_type": None if teacher_feedback is None else teacher_feedback["error_type"],
                    "reason": suggestion["reason"],
                }
            )
        elif action == "update_lag":
            previous_value = int(rule.get("lag", 0))
            new_value = int(suggestion["new_lag"])
            if previous_value == new_value:
                continue
            rule["lag"] = new_value
            rule_update_records.append(
                {
                    "rule_id": rule["rule_id"],
                    "field": "lag",
                    "previous_value": previous_value,
                    "new_value": new_value,
                    "source_month": source_month,
                    "rule_version": rule_version,
                    "teacher_label": None if teacher_feedback is None else teacher_feedback["evaluation_label"],
                    "error_type": None if teacher_feedback is None else teacher_feedback["error_type"],
                    "reason": suggestion["reason"],
                }
            )
        elif action == "update_duration":
            previous_value = int(rule.get("duration", 0))
            new_value = int(suggestion["new_duration"])
            if previous_value == new_value:
                continue
            rule["duration"] = new_value
            rule_update_records.append(
                {
                    "rule_id": rule["rule_id"],
                    "field": "duration",
                    "previous_value": previous_value,
                    "new_value": new_value,
                    "source_month": source_month,
                    "rule_version": rule_version,
                    "teacher_label": None if teacher_feedback is None else teacher_feedback["evaluation_label"],
                    "error_type": None if teacher_feedback is None else teacher_feedback["error_type"],
                    "reason": suggestion["reason"],
                }
            )

    return file_rules, rule_update_records


def _validate_suggestions_against_rules(
    suggestions: dict,
    file_rules: list[Rule],
    matched_rule_ids: set[str],
) -> None:
    required_keys = {"analysis_summary", "rule_update_suggestions"}
    missing = required_keys.difference(suggestions)
    if missing:
        raise SchemaValidationError(f"Reasoning suggestions are missing keys: {sorted(missing)}")
    if not isinstance(suggestions["analysis_summary"], str) or not suggestions["analysis_summary"].strip():
        raise SchemaValidationError("analysis_summary must be a non-empty string.")
    if not isinstance(suggestions["rule_update_suggestions"], list):
        raise SchemaValidationError("rule_update_suggestions must be a list.")

    existing_rule_ids = {rule["rule_id"] for rule in file_rules}
    for suggestion in suggestions["rule_update_suggestions"]:
        _validate_one_suggestion(suggestion, existing_rule_ids, matched_rule_ids)


def _validate_one_suggestion(
    suggestion: dict,
    existing_rule_ids: set[str],
    matched_rule_ids: set[str],
) -> None:
    required = {"rule_id", "action", "new_base_impact", "new_lag", "new_duration", "reason"}
    if not isinstance(suggestion, dict):
        raise SchemaValidationError("Each rule update suggestion must be an object.")
    missing = required.difference(suggestion)
    if missing:
        raise SchemaValidationError(f"Rule update suggestion is missing keys: {sorted(missing)}")
    extra = set(suggestion).difference(required)
    if extra:
        raise SchemaValidationError(f"Rule update suggestion contains unexpected keys: {sorted(extra)}")

    rule_id = suggestion["rule_id"]
    action = suggestion["action"]
    if rule_id not in existing_rule_ids:
        raise SchemaValidationError(f"Suggested rule_id does not exist in rules library: {rule_id!r}")
    if rule_id not in matched_rule_ids:
        raise SchemaValidationError(f"Suggested rule_id was not used by Student: {rule_id!r}")
    if action not in {"update_base_impact", "update_lag", "update_duration", "no_change"}:
        raise SchemaValidationError(f"Unsupported rule update action: {action!r}")
    if not isinstance(suggestion["reason"], str) or not suggestion["reason"].strip():
        raise SchemaValidationError("Rule update suggestion reason must be a non-empty string.")

    if action == "update_base_impact":
        if not isinstance(suggestion["new_base_impact"], (int, float)):
            raise SchemaValidationError("update_base_impact requires numeric new_base_impact.")
        if suggestion["new_lag"] is not None or suggestion["new_duration"] is not None:
            raise SchemaValidationError("update_base_impact must keep unrelated fields as null.")
    elif action == "update_lag":
        if not isinstance(suggestion["new_lag"], int) or suggestion["new_lag"] < 0:
            raise SchemaValidationError("update_lag requires non-negative integer new_lag.")
        if suggestion["new_base_impact"] is not None or suggestion["new_duration"] is not None:
            raise SchemaValidationError("update_lag must keep unrelated fields as null.")
    elif action == "update_duration":
        if not isinstance(suggestion["new_duration"], int) or suggestion["new_duration"] < 0:
            raise SchemaValidationError("update_duration requires non-negative integer new_duration.")
        if suggestion["new_base_impact"] is not None or suggestion["new_lag"] is not None:
            raise SchemaValidationError("update_duration must keep unrelated fields as null.")
    elif action == "no_change":
        if (
            suggestion["new_base_impact"] is not None
            or suggestion["new_lag"] is not None
            or suggestion["new_duration"] is not None
        ):
            raise SchemaValidationError("no_change suggestions must keep all update fields as null.")


def _write_rules(path: Path, rules: list[Rule]) -> None:
    path.write_text(json.dumps(rules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _next_rule_version(rules_path: Path) -> int:
    version_path = rules_path.with_name("active_rules_versions.jsonl")
    if not version_path.exists():
        return 1
    with version_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip()) + 1


def _write_rule_version_record(rules_path: Path, record: dict) -> None:
    version_path = rules_path.with_name("active_rules_versions.jsonl")
    with version_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _validate_inputs(student_record: StudentRecord, teacher_feedback: TeacherFeedback) -> None:
    if student_record.get("metric") != "roa":
        raise SchemaValidationError("Reasoning only supports ROA records in this pipeline.")
    required_feedback_keys = {
        "metric",
        "evaluation_label",
        "error_type",
        "explanation",
        "supervision_snapshot",
        "comparison_summary",
    }
    missing = required_feedback_keys.difference(teacher_feedback)
    if missing:
        raise SchemaValidationError(f"teacher_feedback is missing required keys: {sorted(missing)}")
    if teacher_feedback["metric"] != "roa":
        raise SchemaValidationError("Reasoning only supports ROA teacher feedback in this pipeline.")
