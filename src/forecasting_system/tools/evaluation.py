"""Evaluation tools."""

from __future__ import annotations

from typing import Any

from forecasting_system.config import TARGET_METRIC
from forecasting_system.exceptions import PlaceholderNotImplementedError, SchemaValidationError
from forecasting_system.tools.financial_model import compute_roa
from forecasting_system.types import Direction, StudentRecord, TeacherFeedback


DEFAULT_EVALUATION_TOLERANCE = 0.03


def load_exports():
    """Load local supervision data such as export signals."""
    raise PlaceholderNotImplementedError("load_exports() is scaffolded but not implemented in phase 1.")


def evaluate_prediction(student_record: StudentRecord, supervision_data: Any) -> TeacherFeedback:
    """Return deterministic teacher feedback for monthly ROA forecasting."""
    baseline_state = student_record["baseline_state"]
    baseline_roa = _extract_baseline_metric_value(supervision_data)
    if baseline_roa is None:
        baseline_roa = compute_roa(
            float(baseline_state["profit_margin"]),
            float(baseline_state["asset_turnover"]),
        )
    predicted_roa = float(student_record["final_prediction"]["roa"])
    predicted_direction = _derive_direction(predicted_roa - baseline_roa)

    supervision_direction, actual_roa, export_change_rate, snapshot_status = _extract_supervision_signal(
        supervision_data,
        baseline_roa,
    )

    comparison_summary = {
        "baseline_roa": baseline_roa,
        "predicted_roa": predicted_roa,
        "actual_roa": actual_roa,
        "predicted_direction": predicted_direction,
        "supervision_direction": supervision_direction,
        "predicted_delta": predicted_roa - baseline_roa,
        "supervision_delta": None if actual_roa is None else actual_roa - baseline_roa,
        "predicted_change_rate": _safe_change_rate(predicted_roa, baseline_roa),
        "actual_change_rate": None if actual_roa is None else _safe_change_rate(actual_roa, baseline_roa),
        "export_change_rate": export_change_rate,
    }
    component_feedback = _evaluate_dupont_components(student_record, supervision_data)
    evaluation_metrics = _build_evaluation_metrics(comparison_summary, component_feedback)

    if snapshot_status == "missing":
        return {
            "metric": TARGET_METRIC,
            "evaluation_label": "no_supervision",
            "error_type": "invalid_supervision",
            "explanation": "No usable supervision data was provided for monthly ROA evaluation.",
            "supervision_snapshot": {
                "status": "missing",
                "raw_input": supervision_data,
            },
            "comparison_summary": comparison_summary,
            "component_feedback": component_feedback,
            "evaluation_metrics": evaluation_metrics,
        }

    if snapshot_status == "invalid":
        return {
            "metric": TARGET_METRIC,
            "evaluation_label": "invalid_supervision",
            "error_type": "invalid_supervision",
            "explanation": "Supervision data did not contain a valid ROA or export-change signal.",
            "supervision_snapshot": {
                "status": "invalid",
                "raw_input": supervision_data,
            },
            "comparison_summary": comparison_summary,
            "component_feedback": component_feedback,
            "evaluation_metrics": evaluation_metrics,
        }

    evaluation_label, error_type, explanation = _compare_prediction(
        baseline_roa=baseline_roa,
        predicted_roa=predicted_roa,
        actual_roa=actual_roa,
        predicted_direction=predicted_direction,
        supervision_direction=supervision_direction,
        export_change_rate=export_change_rate,
        tolerance=(
            float(supervision_data.get("tolerance", DEFAULT_EVALUATION_TOLERANCE))
            if isinstance(supervision_data, dict)
            else DEFAULT_EVALUATION_TOLERANCE
        ),
    )
    return {
        "metric": TARGET_METRIC,
        "evaluation_label": evaluation_label,
        "error_type": error_type,
        "explanation": explanation,
        "supervision_snapshot": {
            "status": "ok",
            "raw_input": supervision_data,
            "actual_roa": actual_roa,
            "direction": supervision_direction,
            "export_change_rate": export_change_rate,
        },
        "comparison_summary": comparison_summary,
        "component_feedback": component_feedback,
        "evaluation_metrics": evaluation_metrics,
    }


def _evaluate_dupont_components(
    student_record: StudentRecord,
    supervision_data: Any,
) -> dict[str, Any]:
    if not isinstance(supervision_data, dict):
        return {
            "status": "missing",
            "components": {},
            "primary_error_driver": None,
            "component_error_ranking": [],
        }

    actual_state = supervision_data.get("actual_state") or supervision_data.get("derived_state")
    if not isinstance(actual_state, dict):
        context = student_record.get("quarterly_student_teacher_context") or {}
        actual_state = context.get("derived_state")
    if not isinstance(actual_state, dict):
        return {
            "status": "missing",
            "components": {},
            "primary_error_driver": None,
            "component_error_ranking": [],
        }

    tolerance = float(supervision_data.get("component_tolerance", supervision_data.get("tolerance", DEFAULT_EVALUATION_TOLERANCE)))
    baseline_state = student_record["baseline_state"]
    predicted_state = _extract_predicted_dupont_state(student_record)
    components: dict[str, dict[str, Any]] = {}

    for component in ("profit_margin", "asset_turnover"):
        baseline = float(baseline_state[component])
        predicted = float(predicted_state[component])
        actual = float(actual_state[component])
        predicted_delta = predicted - baseline
        actual_delta = actual - baseline
        level_error = predicted - actual
        delta_error = predicted_delta - actual_delta
        label = _label_from_error(level_error, tolerance)
        components[component] = {
            "baseline": baseline,
            "predicted": predicted,
            "actual": actual,
            "predicted_delta": predicted_delta,
            "actual_delta": actual_delta,
            "level_error": level_error,
            "delta_error": delta_error,
            "squared_error": level_error**2,
            "level_squared_error": level_error**2,
            "delta_squared_error": delta_error**2,
            "evaluation_label": label,
            "explanation": _component_explanation(component, label, level_error, delta_error),
        }

    ranking = sorted(
        (
            {
                "component": component,
                "level_squared_error": values["level_squared_error"],
                "delta_squared_error": values["delta_squared_error"],
                "squared_error": values["squared_error"],
                "level_error": values["level_error"],
                "delta_error": values["delta_error"],
                "evaluation_label": values["evaluation_label"],
            }
            for component, values in components.items()
        ),
        key=lambda item: (item["level_squared_error"], item["delta_squared_error"]),
        reverse=True,
    )
    return {
        "status": "ok",
        "components": components,
        "primary_error_driver": ranking[0]["component"] if ranking else None,
        "component_error_ranking": ranking,
    }


def _extract_predicted_dupont_state(student_record: StudentRecord) -> dict[str, float]:
    intermediate = student_record.get("intermediate_values", {})
    required = {
        "updated_profit_margin": "profit_margin",
        "updated_asset_turnover": "asset_turnover",
    }
    if all(key in intermediate for key in required):
        return {
            component: float(intermediate[key])
            for key, component in required.items()
        }
    baseline = student_record["baseline_state"]
    return {
        "profit_margin": float(baseline["profit_margin"]),
        "asset_turnover": float(baseline["asset_turnover"]),
    }


def _build_evaluation_metrics(
    comparison_summary: dict[str, Any],
    component_feedback: dict[str, Any],
) -> dict[str, Any]:
    predicted_roa = float(comparison_summary["predicted_roa"])
    actual_roa = comparison_summary.get("actual_roa")
    predicted_delta = float(comparison_summary["predicted_delta"])
    actual_delta = comparison_summary.get("supervision_delta")
    metrics = {
        "roa": {
            "level_error": None if actual_roa is None else predicted_roa - float(actual_roa),
            "delta_error": None if actual_delta is None else predicted_delta - float(actual_delta),
        },
        "components": {},
    }
    if metrics["roa"]["level_error"] is not None:
        metrics["roa"]["level_squared_error"] = metrics["roa"]["level_error"] ** 2
        metrics["roa"]["squared_error"] = metrics["roa"]["level_squared_error"]
    if metrics["roa"]["delta_error"] is not None:
        metrics["roa"]["delta_squared_error"] = metrics["roa"]["delta_error"] ** 2

    for component, values in component_feedback.get("components", {}).items():
        metrics["components"][component] = {
            "squared_error": values["squared_error"],
            "level_squared_error": values["level_squared_error"],
            "delta_squared_error": values["delta_squared_error"],
            "level_error": values["level_error"],
            "delta_error": values["delta_error"],
        }
    return metrics


def _label_from_error(error: float, tolerance: float) -> str:
    if abs(error) <= tolerance:
        return "reasonable"
    if error > 0:
        return "too_optimistic"
    return "too_pessimistic"


def _component_explanation(component: str, label: str, level_error: float, delta_error: float) -> str:
    if label == "reasonable":
        return f"Predicted {component} level is within tolerance of the reported state."
    if label == "too_optimistic":
        return (
            f"Predicted {component} is too high versus the reported state "
            f"(level_error={level_error:.4f}, delta_error={delta_error:.4f})."
        )
    return (
        f"Predicted {component} is too low versus the reported state "
        f"(level_error={level_error:.4f}, delta_error={delta_error:.4f})."
    )


def _extract_supervision_signal(
    supervision_data: Any,
    baseline_roa: float,
) -> tuple[Direction | None, float | None, float | None, str]:
    if supervision_data is None:
        return None, None, None, "missing"
    if not isinstance(supervision_data, dict):
        return None, None, None, "invalid"

    raw_direction = supervision_data.get("direction")
    if raw_direction in {"increase", "decrease", "neutral"}:
        direction = raw_direction
    else:
        direction = None

    export_change_rate = _extract_numeric(supervision_data, ("export_change_rate", "change_rate"))
    actual_roa = _extract_numeric(supervision_data, ("actual_roa", "roa", "value"))
    if actual_roa is not None:
        derived_direction = _derive_direction(actual_roa - baseline_roa)
        return derived_direction, actual_roa, export_change_rate, "ok"

    if export_change_rate is not None:
        derived_direction = _derive_direction(export_change_rate)
        return derived_direction, None, export_change_rate, "ok"

    if direction is not None:
        return direction, None, export_change_rate, "ok"

    return None, None, export_change_rate, "invalid"


def _extract_numeric(supervision_data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = supervision_data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_baseline_metric_value(supervision_data: Any) -> float | None:
    if not isinstance(supervision_data, dict):
        return None
    value = supervision_data.get("baseline_roa")
    return float(value) if isinstance(value, (int, float)) else None


def _compare_prediction(
    baseline_roa: float,
    predicted_roa: float,
    actual_roa: float | None,
    predicted_direction: Direction,
    supervision_direction: Direction | None,
    export_change_rate: float | None,
    tolerance: float,
) -> tuple[str, str, str]:
    if supervision_direction is None:
        raise SchemaValidationError("supervision_direction must be present for comparison.")

    if predicted_direction != supervision_direction:
        if predicted_roa > baseline_roa:
            return (
                "too_optimistic",
                "direction_error",
                "Predicted ROA direction is more positive than the supervision signal.",
            )
        if predicted_roa < baseline_roa:
            return (
                "too_pessimistic",
                "direction_error",
                "Predicted ROA direction is more negative than the supervision signal.",
            )
        if supervision_direction == "increase":
            return (
                "too_pessimistic",
                "direction_error",
                "Predicted no ROA change while supervision indicates improvement.",
            )
        return (
            "too_optimistic",
            "direction_error",
            "Predicted no ROA change while supervision indicates deterioration.",
        )

    if export_change_rate is not None:
        predicted_change_rate = _safe_change_rate(predicted_roa, baseline_roa)
        gap = abs(predicted_change_rate - export_change_rate)
        if gap <= tolerance:
            return (
                "reasonable",
                "none",
                f"Predicted ROA change rate is within tolerance of the export change rate (gap={gap:.4f}).",
            )
        if predicted_change_rate > export_change_rate:
            return (
                "too_optimistic",
                "magnitude_error",
                f"Predicted ROA change rate is too high versus export change rate (gap={gap:.4f}).",
            )
        return (
            "too_pessimistic",
            "magnitude_error",
            f"Predicted ROA change rate is too low versus export change rate (gap={gap:.4f}).",
        )

    if actual_roa is not None:
        gap = abs(predicted_roa - actual_roa)
        if gap <= tolerance:
            return (
                "reasonable",
                "none",
                f"Predicted quarterly ROA is within tolerance of the supervised outcome (gap={gap:.4f}).",
            )
        if predicted_roa > actual_roa:
            return (
                "too_optimistic",
                "magnitude_error",
                f"Predicted quarterly ROA is too high versus the supervised outcome (gap={gap:.4f}).",
            )
        if predicted_roa < actual_roa:
            return (
                "too_pessimistic",
                "magnitude_error",
                f"Predicted quarterly ROA is too low versus the supervised outcome (gap={gap:.4f}).",
            )

    return (
        "reasonable",
        "none",
        "Predicted ROA direction is aligned with supervision data.",
    )


def _derive_direction(delta: float) -> Direction:
    tolerance = 1e-12
    if delta > tolerance:
        return "increase"
    if delta < -tolerance:
        return "decrease"
    return "neutral"


def _safe_change_rate(new_value: float, base_value: float) -> float:
    if abs(base_value) <= 1e-12:
        return 0.0
    return (new_value - base_value) / abs(base_value)
