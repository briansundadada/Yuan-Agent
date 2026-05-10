"""Teacher Agent implementation for phase 1."""

from __future__ import annotations

from forecasting_system.exceptions import SchemaValidationError
from forecasting_system.tools.evaluation import evaluate_prediction
from forecasting_system.types import StudentRecord, TeacherFeedback


class Teacher:
    """Phase-1 explicit interface for deterministic evaluation."""

    def run(self, student_record: StudentRecord, supervision_data) -> TeacherFeedback:
        _validate_student_record_input(student_record)
        feedback = evaluate_prediction(student_record, supervision_data)
        _validate_teacher_feedback(feedback)
        return feedback


def _validate_student_record_input(student_record: StudentRecord) -> None:
    required_keys = {"metric", "baseline_state", "final_prediction"}
    missing = required_keys.difference(student_record)
    if missing:
        raise SchemaValidationError(f"student_record is missing required keys: {sorted(missing)}")
    if student_record["metric"] != "roa":
        raise SchemaValidationError("Teacher only supports ROA student records in this pipeline.")
    baseline_keys = {"profit_margin", "asset_turnover", "equity_multiplier"}
    if baseline_keys.difference(student_record["baseline_state"]):
        raise SchemaValidationError(
            "student_record baseline_state must include profit_margin, asset_turnover, and equity_multiplier."
        )
    if "roa" not in student_record["final_prediction"]:
        raise SchemaValidationError("student_record final_prediction must include roa.")


def _validate_teacher_feedback(feedback: TeacherFeedback) -> None:
    required_keys = {
        "metric",
        "evaluation_label",
        "error_type",
        "explanation",
        "supervision_snapshot",
        "comparison_summary",
    }
    missing = required_keys.difference(feedback)
    if missing:
        raise SchemaValidationError(f"teacher feedback is missing required keys: {sorted(missing)}")
