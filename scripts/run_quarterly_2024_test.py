from __future__ import annotations

import copy
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from forecasting_system.agents.student import Student
from forecasting_system.agents.teacher import Teacher
from forecasting_system.exceptions import SchemaValidationError
from forecasting_system.tools.cycle_state import infer_company_cycle_state
from forecasting_system.tools.events import extract_quarter_events
from forecasting_system.tools.news import preprocess_quarterly_news
from forecasting_system.tools.rules import load_rules, match_rules_for_event


NEWS_PATH = PROJECT_ROOT / "data" / "news" / "news_2024.xlsx"
FUNDAMENTAL_OUTPUT_DIR = PROJECT_ROOT / "logs" / "fundamental_analyst_report_test" / "fundamental_analyst_outputs"
DEFAULT_RULES_PATH = PROJECT_ROOT / "data" / "rules" / "rules.json"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "quarterly_2024_test_final_rules"
TEACHER_TOLERANCE = 0.03

TEST_PERIODS = [
    (0, "2024_START", "year_start", "2023FY"),
    (1, "2024Q1", "quarter", "2024Q1"),
    (2, "2024H1", "quarter", "2024H1"),
    (3, "2024Q3", "quarter", "2024Q3"),
    (4, "2024FY", "quarter", "2024FY"),
]


def _time_stage(label: str, fn, *args, **kwargs):
    started_at = time.perf_counter()
    print(f"[STAGE START] {label}", flush=True)
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - started_at
    print(f"[STAGE DONE] {label} | elapsed={elapsed:.2f}s", flush=True)
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rules_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RULES_PATH
    if not rules_path.is_absolute():
        rules_path = PROJECT_ROOT / rules_path
    eda_payload = _time_stage("load_or_build_2024_eda", _load_or_build_2024_eda)
    report_metrics = _time_stage("load_report_metrics", _load_report_metrics)
    rules = _time_stage("load_rules", load_rules, rules_path)
    records, event_counter = _time_stage("run_student_teacher_test", _run_test, eda_payload, report_metrics, rules, rules_path)

    summary = _build_summary(records)
    summary["rules_path"] = str(rules_path)
    _time_stage("write_quarterly_records", _write_json, OUTPUT_DIR / "quarterly_records.json", records)
    _time_stage("write_cycle_states", _write_json, OUTPUT_DIR / "fundamental_cycle_states.json", _collect_cycle_states(records))
    _time_stage("write_summary_metrics", _write_json, OUTPUT_DIR / "summary_metrics.json", summary)
    _time_stage("write_summary_csv", _write_summary_csv, records, OUTPUT_DIR / "quarterly_summary.csv")
    _time_stage("write_component_metrics", _write_component_metrics, records, OUTPUT_DIR / "teacher_component_metrics.csv")
    _time_stage("write_event_frequency", _write_event_frequency, event_counter, OUTPUT_DIR / "event_frequency.csv")
    _time_stage("plot_prediction", _plot_prediction, records, OUTPUT_DIR / "roa_component_prediction_vs_actual.png")
    _time_stage("plot_mse", _plot_mse, summary, OUTPUT_DIR / "roa_component_mse.png")
    print(OUTPUT_DIR)


def _load_or_build_2024_eda() -> dict[str, Any]:
    cache_path = OUTPUT_DIR / "quarterly_event_classification.json"
    payload = _load_json(cache_path, default={}) if cache_path.exists() else {}
    quarterly_news = preprocess_quarterly_news([NEWS_PATH])
    summary_rows = []
    deduplication_log = []
    event_counter: Counter[str] = Counter()

    for quarter, suffix in ((1, "Q1"), (2, "H1"), (3, "Q3"), (4, "FY")):
        period = f"2024{suffix}"
        raw_titles = quarterly_news.get((2024, quarter), [])
        if period in payload:
            print(f"SKIP {period}: loaded cached EDA output")
            period_payload = payload[period]
        else:
            print(f"START {period}: raw_news={len(raw_titles)}")
            batch_payload = extract_quarter_events(raw_titles) if raw_titles else {
                "deduped_news": [],
                "duplicate_groups": [],
                "events": [],
            }
            period_payload = {
                "period": period,
                "year": 2024,
                "quarter": quarter,
                "raw_news": raw_titles,
                "deduped_news": batch_payload["deduped_news"],
                "duplicate_groups": batch_payload["duplicate_groups"],
                "events": batch_payload["events"],
            }
            payload[period] = period_payload
            _write_json(cache_path, payload)

        non_noise_events = [event for event in period_payload["events"] if not event.get("noise", False)]
        for event in non_noise_events:
            event_counter[f"{event['scenario']}::{event['event_type']}::{event['direction']}"] += 1
        summary_rows.append(
            {
                "period": period,
                "raw_news_count": len(raw_titles),
                "deduped_news_count": len(period_payload["deduped_news"]),
                "duplicate_group_count": len(period_payload["duplicate_groups"]),
                "event_count": len(period_payload["events"]),
                "non_noise_event_count": len(non_noise_events),
            }
        )
        deduplication_log.append(
            {
                "period": period,
                "raw_news_count": len(raw_titles),
                "deduped_news_count": len(period_payload["deduped_news"]),
                "duplicate_group_count": len(period_payload["duplicate_groups"]),
                "duplicate_groups": period_payload["duplicate_groups"],
            }
        )

    _write_csv(OUTPUT_DIR / "quarterly_eda_summary.csv", summary_rows)
    _write_json(OUTPUT_DIR / "deduplication_log.json", deduplication_log)
    _write_event_frequency(event_counter, OUTPUT_DIR / "eda_event_frequency.csv")
    return payload


def _run_test(
    eda_payload: dict[str, Any],
    report_metrics: dict[str, dict[str, Any]],
    rules: list[dict[str, Any]],
    rules_path: Path | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    student = Student()
    teacher = Teacher()
    records: list[dict[str, Any]] = []
    event_counter: Counter[str] = Counter()
    scheduled_events_by_t: dict[int, list[dict[str, Any]]] = defaultdict(list)
    current_state: dict[str, float] | None = None
    current_roa = 0.0
    cycle_state_by_period = _initial_cycle_state_map(report_metrics)

    for t, period, kind, report_period in TEST_PERIODS:
        report = report_metrics[report_period]
        if kind == "year_start":
            current_state = _year_start_state(report)
            current_roa = 0.0
            records.append(
                {
                    "t": t,
                    "period": period,
                    "kind": kind,
                    "report_period_used": report_period,
                    "raw_news_count": 0,
                    "deduped_news_count": 0,
                    "non_noise_event_count": 0,
                    "events": [],
                    "baseline_state": copy.deepcopy(current_state),
                    "predicted_state": copy.deepcopy(current_state),
                    "actual_state": None,
                    "predicted_roa": 0.0,
                    "actual_roa": 0.0,
                    "teacher_feedback": None,
                    "student_record": None,
                }
            )
            continue

        if current_state is None:
            raise SchemaValidationError("Quarter reached before a year-start state.")
        quarter_payload = eda_payload[period]
        cycle_state = cycle_state_by_period.get(period)
        enriched_events = _enrich_events_with_news_names(quarter_payload)
        _schedule_eda_timed_events(
            source_t=t,
            source_period=period,
            events=enriched_events,
            rules=rules,
            scheduled_events_by_t=scheduled_events_by_t,
        )
        active_events = scheduled_events_by_t.get(t, [])
        non_noise_events = [event for event in active_events if not event.get("noise", False)]
        for event in non_noise_events:
            event_counter[f"{event['scenario']}::{event['event_type']}::{event['direction']}"] += 1

        student_record = _time_stage(f"{period} student", student.run, non_noise_events, current_state, rules)
        predicted_state = _predicted_state_from_student_record(student_record)
        actual_state = copy.deepcopy(report["derived_state"])
        actual_roa = float(actual_state["reported_roa"])
        predicted_roa = float(student_record["final_prediction"]["roa"])
        student_record["quarterly_2024_test_context"] = {
            "period": period,
            "t": t,
            "supervision_frequency": "quarterly",
            "actual_roa": actual_roa,
            "fundamental_analyst_report": report.get("fundamental_analyst_report"),
            "derived_state": actual_state,
            "uses_reasoning": False,
            "rules_path": str(rules_path or DEFAULT_RULES_PATH),
        }
        teacher_feedback = _time_stage(
            f"{period} teacher",
            teacher.run,
            student_record,
            {
                "actual_roa": actual_roa,
                "actual_state": actual_state,
                "baseline_roa": current_roa,
                "tolerance": TEACHER_TOLERANCE,
                "component_tolerance": TEACHER_TOLERANCE,
                "frequency": "quarterly",
                "period": period,
                "supervision_source": "reported_actual_roa",
            },
        )
        records.append(
            {
                "t": t,
                "period": period,
                "kind": kind,
                "report_period_used": report_period,
                "raw_news_count": len(quarter_payload["raw_news"]),
                "deduped_news_count": len(quarter_payload["deduped_news"]),
                "non_noise_event_count": len(non_noise_events),
                "events": active_events,
                "source_period_events": enriched_events,
                "cycle_state": cycle_state,
                "baseline_state": copy.deepcopy(current_state),
                "predicted_state": predicted_state,
                "actual_state": actual_state,
                "predicted_roa": predicted_roa,
                "actual_roa": actual_roa,
                "predicted_delta": predicted_roa - current_roa,
                "supervision_delta": actual_roa - current_roa,
                "student_record": student_record,
                "teacher_feedback": teacher_feedback,
            }
        )
        current_state = {
            "profit_margin": float(actual_state["profit_margin"]),
            "asset_turnover": float(actual_state["asset_turnover"]),
            "equity_multiplier": float(actual_state["equity_multiplier"]),
        }
        current_roa = actual_roa
        next_period = _next_quarter_period(period)
        if next_period is not None:
            cycle_state_by_period[next_period] = infer_company_cycle_state(
                source_period=period,
                target_period=next_period,
                report_payload=report,
            )

    return records, event_counter


def _initial_cycle_state_map(report_metrics: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "2024Q1": infer_company_cycle_state(
            source_period="2023FY",
            target_period="2024Q1",
            report_payload=report_metrics["2023FY"],
        )
    }


def _next_quarter_period(period: str) -> str | None:
    order = ["2024Q1", "2024H1", "2024Q3", "2024FY"]
    if period not in order:
        return None
    index = order.index(period)
    if index + 1 >= len(order):
        return None
    return order[index + 1]


def _collect_cycle_states(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "period": record["period"],
            "cycle_state": record.get("cycle_state"),
        }
        for record in records
        if record.get("kind") == "quarter"
    ]


def _schedule_eda_timed_events(
    *,
    source_t: int,
    source_period: str,
    events: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    scheduled_events_by_t: dict[int, list[dict[str, Any]]],
) -> None:
    max_t = max(t for t, _, kind, _ in TEST_PERIODS if kind == "quarter")
    for event in events:
        if event.get("noise", False):
            continue
        for rule in match_rules_for_event(event, rules):
            lag = _event_lag(event)
            duration = _event_duration(event)
            for offset in range(duration):
                active_t = source_t + lag + offset
                if active_t > max_t:
                    continue
                scheduled = copy.deepcopy(event)
                scheduled["active_rule_ids"] = [rule["rule_id"]]
                scheduled["timing"] = {
                    "source_t": source_t,
                    "source_period": source_period,
                    "active_t": active_t,
                    "rule_id": rule["rule_id"],
                    "event_lag": lag,
                    "event_duration": duration,
                    "duration_offset": offset,
                }
                scheduled_events_by_t[active_t].append(scheduled)


def _event_lag(event: dict[str, Any]) -> int:
    value = event.get("lag", 0)
    return min(int(value), 4) if isinstance(value, int) and value >= 0 else 0


def _event_duration(event: dict[str, Any]) -> int:
    value = event.get("duration", 1)
    if not isinstance(value, int) or value <= 0:
        return 1
    return min(value, 4)


def _enrich_events_with_news_names(quarter_payload: dict[str, Any]) -> list[dict[str, Any]]:
    deduped_news = quarter_payload.get("deduped_news", [])
    duplicate_groups = quarter_payload.get("duplicate_groups", [])
    events = []
    for index, event in enumerate(quarter_payload.get("events", [])):
        enriched = copy.deepcopy(event)
        canonical = deduped_news[index] if index < len(deduped_news) else None
        if canonical:
            enriched.setdefault("canonical_news", str(canonical))
            enriched.setdefault("raw_news", str(canonical))
            enriched.setdefault("duplicate_news", _duplicates_for_canonical(str(canonical), duplicate_groups))
        enriched.setdefault("event_key", f"{enriched.get('scenario')}::{enriched.get('event_type')}::{enriched.get('direction')}")
        events.append(enriched)
    return events


def _duplicates_for_canonical(canonical: str, duplicate_groups: list[dict[str, Any]]) -> list[str]:
    for group in duplicate_groups:
        if group.get("canonical") == canonical:
            return [str(item) for item in group.get("duplicates", [])]
    return []


def _year_start_state(previous_fy_report: dict[str, Any]) -> dict[str, float]:
    previous_state = previous_fy_report["derived_state"]
    return {
        "profit_margin": 0.0,
        "asset_turnover": 0.0,
        "equity_multiplier": float(previous_state["equity_multiplier"]),
    }


def _predicted_state_from_student_record(student_record: dict[str, Any]) -> dict[str, float]:
    intermediate = student_record["intermediate_values"]
    return {
        "profit_margin": float(intermediate["updated_profit_margin"]),
        "asset_turnover": float(intermediate["updated_asset_turnover"]),
        "equity_multiplier": float(intermediate["updated_equity_multiplier"]),
        "reported_roa": float(student_record["final_prediction"]["roa"]),
    }


def _load_report_metrics() -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for path in FUNDAMENTAL_OUTPUT_DIR.glob("*.json"):
        payload = _load_json(path, default={})
        _ensure_reported_roa(payload)
        metrics[payload["period"]] = payload
    required = sorted({report_period for _, _, _, report_period in TEST_PERIODS})
    missing = [period for period in required if period not in metrics]
    if missing:
        raise SystemExit(f"Missing Fundamental Analyst JSON outputs: {missing}.")
    return metrics


def _ensure_reported_roa(report: dict[str, Any]) -> None:
    derived = report.setdefault("derived_state", {})
    if "reported_roa" in derived:
        return
    profit_margin = float(derived.get("profit_margin", 0.0))
    asset_turnover = float(derived.get("asset_turnover", 0.0))
    derived["reported_roa"] = profit_margin * asset_turnover
    report.setdefault("key_indicators", {})["reported_roa"] = round(float(derived["reported_roa"]), 6)


def _build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    quarters = [record for record in records if record["kind"] == "quarter"]
    labels = [record["teacher_feedback"]["evaluation_label"] for record in quarters]
    mse = _aggregate_mse(quarters)
    return {
        "quarter_count": len(quarters),
        "year_start_count": sum(1 for record in records if record["kind"] == "year_start"),
        "reasonable_count": labels.count("reasonable"),
        "too_optimistic_count": labels.count("too_optimistic"),
        "too_pessimistic_count": labels.count("too_pessimistic"),
        "mse": mse,
        "final_predicted_roa": float(quarters[-1]["predicted_roa"]) if quarters else 0.0,
        "final_actual_roa": float(quarters[-1]["actual_roa"]) if quarters else 0.0,
        "rules_path": str(DEFAULT_RULES_PATH),
        "uses_reasoning": False,
        "uses_eda_lag_duration": True,
    }


def _aggregate_mse(quarters: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets = {metric: [] for metric in ("roa", "profit_margin", "asset_turnover")}
    for record in quarters:
        metrics = record["teacher_feedback"]["evaluation_metrics"]
        buckets["roa"].append(float(metrics["roa"]["squared_error"]))
        for component, values in metrics["components"].items():
            buckets[component].append(float(values["squared_error"]))
    return {metric: {"mse": sum(values) / len(values) if values else 0.0} for metric, values in buckets.items()}


def _write_summary_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    rows = []
    for record in records:
        feedback = record.get("teacher_feedback") or {}
        comparison = feedback.get("comparison_summary", {})
        rows.append(
            {
                "t": record["t"],
                "period": record["period"],
                "kind": record["kind"],
                "report_period_used": record["report_period_used"],
                "raw_news_count": record["raw_news_count"],
                "deduped_news_count": record["deduped_news_count"],
                "non_noise_event_count": record["non_noise_event_count"],
                "predicted_roa": record["predicted_roa"],
                "actual_roa": record["actual_roa"],
                "predicted_delta": comparison.get("predicted_delta"),
                "supervision_delta": comparison.get("supervision_delta"),
                "teacher_label": feedback.get("evaluation_label"),
                "error_type": feedback.get("error_type"),
                "primary_error_driver": feedback.get("component_feedback", {}).get("primary_error_driver"),
            }
        )
    _write_csv(output_path, rows)


def _write_component_metrics(records: list[dict[str, Any]], output_path: Path) -> None:
    rows = []
    for record in records:
        if record["kind"] != "quarter":
            continue
        feedback = record["teacher_feedback"]
        comparison = feedback["comparison_summary"]
        roa_metrics = feedback["evaluation_metrics"]["roa"]
        rows.append(_metric_row(record, "roa", comparison, roa_metrics, feedback["evaluation_label"], feedback))
        for metric, values in feedback.get("component_feedback", {}).get("components", {}).items():
            rows.append(
                {
                    "t": record["t"],
                    "period": record["period"],
                    "metric": metric,
                    "baseline": values["baseline"],
                    "predicted": values["predicted"],
                    "actual": values["actual"],
                    "predicted_delta": values["predicted_delta"],
                    "actual_delta": values["actual_delta"],
                    "level_error": values["level_error"],
                    "delta_error": values["delta_error"],
                    "squared_error": values["squared_error"],
                    "teacher_label": values["evaluation_label"],
                    "primary_error_driver": feedback["component_feedback"].get("primary_error_driver"),
                }
            )
    _write_csv(output_path, rows)


def _metric_row(
    record: dict[str, Any],
    metric: str,
    comparison: dict[str, Any],
    values: dict[str, Any],
    label: str,
    feedback: dict[str, Any],
) -> dict[str, Any]:
    return {
        "t": record["t"],
        "period": record["period"],
        "metric": metric,
        "baseline": comparison["baseline_roa"],
        "predicted": comparison["predicted_roa"],
        "actual": comparison["actual_roa"],
        "predicted_delta": comparison["predicted_delta"],
        "actual_delta": comparison["supervision_delta"],
        "level_error": values.get("level_error"),
        "delta_error": values.get("delta_error"),
        "squared_error": values.get("squared_error"),
        "teacher_label": label,
        "primary_error_driver": feedback["component_feedback"].get("primary_error_driver"),
    }


def _write_event_frequency(counter: Counter[str], output_path: Path) -> None:
    rows = []
    for key, count in counter.most_common():
        scenario, event_type, direction = key.split("::")
        rows.append({"scenario": scenario, "event_type": event_type, "direction": direction, "count": count})
    _write_csv(output_path, rows)


def _plot_prediction(records: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    quarters = [record for record in records if record["kind"] == "quarter"]
    x = [int(record["t"]) for record in quarters]
    labels = [record["period"] for record in quarters]
    metrics = [("roa", "ROA"), ("profit_margin", "Profit Margin"), ("asset_turnover", "Asset Turnover")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        if metric == "roa":
            predicted = [float(record["predicted_roa"]) for record in quarters]
            actual = [float(record["actual_roa"]) for record in quarters]
        else:
            predicted = [float(record["predicted_state"][metric]) for record in quarters]
            actual = [float(record["actual_state"][metric]) for record in quarters]
        ax.plot(x, actual, color="#111111", marker="o", linewidth=2.4, label="actual")
        ax.plot(x, predicted, color="#4C78A8", marker="s", linewidth=2.0, label="predicted")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_mse(summary: dict[str, Any], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    metrics = ["roa", "profit_margin", "asset_turnover"]
    values = [float(summary["mse"][metric]["mse"]) for metric in metrics]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(metrics, values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title("2024 Test MSE")
    ax.set_ylabel("MSE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
