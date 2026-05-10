"""Shared domain types for phase 1."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

try:
    from typing import NotRequired
except ImportError:  # pragma: no cover - Python < 3.11 compatibility
    from typing_extensions import NotRequired


Direction = Literal["increase", "decrease", "neutral"]
TargetComponent = Literal["profit_margin", "asset_turnover"]
MetricName = Literal["roa"]
ErrorType = Literal["direction_error", "magnitude_error", "attribution_error", "invalid_supervision", "none"]


class Event(TypedDict):
    scenario: str
    event_type: str
    direction: Direction
    strength: float
    relevance: float
    lag: int
    duration: int
    noise: bool


class RuleTrigger(TypedDict):
    scenario: str
    event_type: str
    direction: Direction


class Rule(TypedDict):
    rule_id: str
    trigger: RuleTrigger
    target_component: TargetComponent
    function_name: Literal["linear_adjustment"]
    params: dict[str, float]
    explanation: str
    lag: NotRequired[int]
    duration: NotRequired[int]
    business_chain: NotRequired[str]
    fundamental_basis: NotRequired[str]
    operating_metric_links: NotRequired[list[str]]


class BaselineState(TypedDict):
    profit_margin: float
    asset_turnover: float
    equity_multiplier: float


class FundamentalBaseline(TypedDict, total=False):
    profit_margin: float
    asset_turnover: float
    equity_multiplier: float
    source_summary: str
    confidence: float


class MatchedRuleRecord(TypedDict):
    rule_id: str
    target_component: TargetComponent
    function_name: str
    inputs: dict[str, Any]
    delta: float


class StudentRecord(TypedDict):
    metric: MetricName
    input_events: list[Event]
    baseline_state: BaselineState
    matched_rules: list[MatchedRuleRecord]
    function_calls: list[dict[str, Any]]
    intermediate_values: dict[str, float]
    final_prediction: dict[str, float]


class TeacherFeedback(TypedDict):
    metric: MetricName
    evaluation_label: str
    error_type: ErrorType
    explanation: str
    supervision_snapshot: dict[str, Any]
    comparison_summary: dict[str, Any]


class RuleChange(TypedDict):
    rule_id: str
    field: str
    previous_value: float
    new_value: float
    source_month: NotRequired[int | None]
    rule_version: NotRequired[int | None]
    teacher_label: NotRequired[str | None]
    error_type: NotRequired[str | None]
    reason: str


class RuleUpdateRecord(TypedDict):
    updated_rules: list[Rule]
    rule_update_records: list[RuleChange]
    explanation: str
