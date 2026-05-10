"""Company cycle state inference from Fundamental Analyst report outputs."""

from __future__ import annotations

from typing import Any


ALLOWED_COMPANY_CYCLE_STATES = {
    "early_recovery",
    "demand_expansion",
    "capacity_expansion",
    "margin_pressure",
    "high_growth_operating_leverage",
    "working_capital_stress",
    "mature_normalization",
    "downturn_shock",
}


def infer_company_cycle_state(
    *,
    source_period: str,
    target_period: str,
    report_payload: dict[str, Any],
) -> dict[str, Any]:
    """Infer the cycle state to use in the next forecasting period.

    This is intentionally deterministic and conservative. It consumes the
    Fundamental Analyst report text plus derived DuPont state, then emits one
    fixed-cycle-state label for the next period.
    """
    derived = report_payload.get("derived_state", {}) or {}
    report_text = str(report_payload.get("fundamental_analyst_report", "")).lower()
    roe = _safe_float(derived.get("reported_roe"))
    profit_margin = _safe_float(derived.get("profit_margin"))
    asset_turnover = _safe_float(derived.get("asset_turnover"))
    equity_multiplier = _safe_float(derived.get("equity_multiplier"))

    primary = "demand_expansion"
    secondary: str | None = None
    evidence = []

    if _mentions(report_text, ("negative operating cash", "cash flow", "inventory", "receivable", "working capital")):
        secondary = "working_capital_stress"
        evidence.append("Fundamental Analyst mentioned cash-flow, inventory, receivable, or working-capital pressure.")

    if _mentions(report_text, ("margin pressure", "gross margin compressed", "compressed", "profitability pressure")) or (
        profit_margin is not None and profit_margin < 0.08
    ):
        primary = "margin_pressure"
        evidence.append("Margin/profitability pressure is visible in the report or profit_margin is low.")
    elif _mentions(report_text, ("capacity", "factory", "capex", "production base", "construction", "investment")):
        primary = "capacity_expansion"
        evidence.append("Report highlights capacity expansion, factory construction, capex, or investment activity.")
    elif (
        (roe is not None and roe >= 0.20)
        and (profit_margin is not None and profit_margin >= 0.10)
        and _mentions(report_text, ("revenue", "profit growth", "net profit", "growth"))
    ):
        primary = "high_growth_operating_leverage"
        evidence.append("High ROE/profit margin plus report language indicates strong growth leverage.")
    elif _mentions(report_text, ("demand", "order", "revenue growth", "overseas", "shipment")):
        primary = "demand_expansion"
        evidence.append("Report emphasizes demand, orders, shipments, revenue growth, or overseas expansion.")
    elif _mentions(report_text, ("balanced", "stable", "normal", "mature")) or (
        roe is not None and roe >= 0.25 and asset_turnover is not None and asset_turnover >= 0.70
    ):
        primary = "mature_normalization"
        evidence.append("The report/state suggests balanced high-level operation and normalization.")
    elif roe is not None and roe < 0:
        primary = "downturn_shock"
        evidence.append("Reported ROE is negative.")
    elif roe is not None and 0 <= roe < 0.08:
        primary = "early_recovery"
        evidence.append("ROE is positive but still low, suggesting an early recovery stage.")

    if secondary == primary:
        secondary = None

    if not evidence:
        evidence.append(
            "Conservative default based on Fundamental Analyst derived state: "
            f"ROE={roe}, PM={profit_margin}, ATO={asset_turnover}, EM={equity_multiplier}."
        )

    return {
        "source": "fundamental_analyst",
        "source_period": source_period,
        "target_period": target_period,
        "primary_cycle_state": primary,
        "secondary_cycle_state": secondary,
        "evidence": " ".join(evidence),
        "derived_state_snapshot": {
            "reported_roe": roe,
            "profit_margin": profit_margin,
            "asset_turnover": asset_turnover,
            "equity_multiplier": equity_multiplier,
        },
    }


def _mentions(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
