import json

from forecasting_system.config import RULES_PATH
from forecasting_system.tools.logging_tools import (
    log_rule_update,
    log_student_record,
    log_teacher_feedback,
)
from forecasting_system.tools.rules import load_rules, match_rules_for_event


def test_load_rules_returns_phase1_rules():
    rules = load_rules(RULES_PATH)
    assert len(rules) >= 1
    assert {rule["target_component"] for rule in rules}.issubset({"profit_margin", "asset_turnover"})
    assert {rule["function_name"] for rule in rules} == {"linear_adjustment"}


def test_match_rules_for_event_filters_exact_trigger():
    rules = load_rules(RULES_PATH)
    matches = match_rules_for_event(
        {
            "scenario": "raw_materials",
            "event_type": "upstream_price",
            "direction": "increase",
            "strength": 0.8,
            "relevance": 0.9,
            "lag": 0,
            "duration": 2,
            "noise": False,
        },
        rules,
    )
    assert [rule["rule_id"] for rule in matches] == ["raw_materials_upstream_price_increase"]


def test_logging_tools_append_jsonl(tmp_path):
    student_log = tmp_path / "student_records.jsonl"
    teacher_log = tmp_path / "teacher_feedback.jsonl"
    rule_log = tmp_path / "rule_updates.jsonl"

    log_student_record({"metric": "roa"}, student_log)
    log_teacher_feedback({"evaluation_label": "ok"}, teacher_log)
    log_rule_update({"updates": []}, rule_log)

    assert json.loads(student_log.read_text(encoding="utf-8").strip()) == {"metric": "roa"}
    assert json.loads(teacher_log.read_text(encoding="utf-8").strip()) == {"evaluation_label": "ok"}
    assert json.loads(rule_log.read_text(encoding="utf-8").strip()) == {"updates": []}
