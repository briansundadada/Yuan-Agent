"""PDF report extraction and Fundamental Analyst analysis for ROA calibration."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from forecasting_system.config import DEEPSEEK_API_KEY
from forecasting_system.exceptions import SchemaValidationError
from forecasting_system.tools.deepseek_client import chat_json


REPORT_PERIOD_FILES = {
    "2020FY": Path("data/reports/sungrow_2020_annual_summary.pdf"),
    "2021Q1": Path("data/reports/sungrow_2021_q1_full.pdf"),
    "2021H1": Path("data/reports/sungrow_2021_h1_summary.pdf"),
    "2021Q3": Path("data/reports/sungrow_2021_q3.pdf"),
    "2021FY": Path("data/reports/sungrow_2021_annual_summary.pdf"),
    "2022Q1": Path("data/reports/sungrow_2022_q1.pdf"),
    "2022H1": Path("data/reports/sungrow_2022_h1_summary.pdf"),
    "2022Q3": Path("data/reports/sungrow_2022_q3.pdf"),
    "2022FY": Path("data/reports/sungrow_2022_annual_summary.pdf"),
    "2023Q1": Path("data/reports/sungrow_2023_q1.pdf"),
    "2023H1": Path("data/reports/sungrow_2023_h1_summary.pdf"),
    "2023Q3": Path("data/reports/sungrow_2023_q3.pdf"),
    "2023FY": Path("data/reports/sungrow_2023_annual_summary.pdf"),
    "2024Q1": Path("data/reports/sungrow_2024_q1.pdf"),
    "2024H1": Path("data/reports/sungrow_2024_h1_summary.pdf"),
    "2024Q3": Path("data/reports/sungrow_2024_q3.pdf"),
    "2024FY": Path("data/reports/sungrow_2024_annual_summary.pdf"),
}

REPORT_FILES = {
    0: REPORT_PERIOD_FILES["2023FY"],
    3: REPORT_PERIOD_FILES["2024Q1"],
    6: REPORT_PERIOD_FILES["2024H1"],
    9: REPORT_PERIOD_FILES["2024Q3"],
    12: REPORT_PERIOD_FILES["2024FY"],
}


def analyze_report_pdf(report_path: str | Path, period_label: str, api_key: str | None = None) -> dict[str, Any]:
    """Extract report text locally and ask the Fundamental Analyst LLM for structured indicators."""
    text = extract_pdf_text(report_path)
    if not text.strip():
        raise SchemaValidationError(f"No readable text extracted from report: {report_path}")

    active_api_key = api_key or DEEPSEEK_API_KEY
    if active_api_key:
        try:
            return _normalize_report_analysis(
                _analyze_with_deepseek(text, Path(report_path).name, period_label, active_api_key)
            )
        except Exception:
            pass
    return _normalize_report_analysis(
        _heuristic_report_analysis(text=text, filename=Path(report_path).name, period_label=period_label)
    )


def load_or_analyze_report_pdf(
    report_path: str | Path,
    period_label: str,
    cache_path: str | Path | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    if cache_path is not None:
        cache_file = Path(cache_path)
        if cache_file.exists():
            return _normalize_report_analysis(json.loads(cache_file.read_text(encoding="utf-8-sig")))

    analysis = analyze_report_pdf(report_path=report_path, period_label=period_label, api_key=api_key)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return analysis


def extract_pdf_text(report_path: str | Path) -> str:
    """Extract text from a PDF using pypdf and keep page text boundaries."""
    path = Path(report_path)
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise SchemaValidationError("PDF extraction requires pypdf. Install it before running real report parsing.") from exc

    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(f"\n[page {page_number}]\n{text.strip()}")
    return "\n".join(pages)


def get_report_path_for_month(month: int) -> Path | None:
    return REPORT_FILES.get(month)


def iter_report_period_files() -> list[tuple[str, Path]]:
    return list(REPORT_PERIOD_FILES.items())


def _analyze_with_deepseek(text: str, filename: str, period_label: str, api_key: str) -> dict[str, Any]:
    return chat_json(
        system_prompt=(
            "You are the Fundamental Analyst Agent for an auditable financial forecasting system. "
            "Read extracted PDF text carefully and return one JSON object only."
        ),
        user_prompt=(
            "You are analyzing Sungrow's financial report text extracted from PDF.\n"
            "Use factual, disclosed, cumulative-to-date indicators. For Q1, H1, Q3, and FY, ROE must be "
            "year-to-date/cumulative when the report discloses weighted average ROE; do not convert it to a single-quarter flow.\n"
            "Treat weighted average ROE as the authoritative real ROE field whenever it is present.\n"
            "Keep equity_multiplier for fundamental analysis, and calculate reported_roa as profit_margin * asset_turnover.\n"
            "Distinguish flow metrics such as revenue/net profit/operating cash flow from stock metrics such as total assets, "
            "total equity, inventory, and accounts receivable.\n"
            "Calculate equity_multiplier as total_assets / total_equity when both are available. "
            "If exact values are unavailable, infer cautiously and explain uncertainty in calculation_notes.\n"
            "Return JSON only with exactly these keys:\n"
            "{\n"
            '  "period": "string",\n'
            '  "report_name": "string",\n'
            '  "summary": "string",\n'
            '  "key_indicators": {\n'
            '    "revenue": "string",\n'
            '    "net_profit": "string",\n'
            '    "total_assets": "string",\n'
            '    "total_equity": "string",\n'
            '    "operating_cash_flow": "string",\n'
            '    "inventory": "string",\n'
            '    "accounts_receivable": "string",\n'
            '    "gross_margin": "string",\n'
            '    "net_margin": "string",\n'
            '    "asset_turnover": "string",\n'
            '    "equity_multiplier": "string",\n'
            '    "weighted_average_roe": number,\n'
            '    "overseas_revenue_share": "string"\n'
            "  },\n"
            '  "derived_state": {\n'
            '    "profit_margin": number,\n'
            '    "asset_turnover": number,\n'
            '    "equity_multiplier": number,\n'
            '    "reported_roa": number,\n'
            '    "reported_roe": number\n'
            "  },\n"
            '  "fundamental_analyst_report": "string",\n'
            '  "calculation_notes": "string"\n'
            "}\n"
            f"Report name: {filename}\n"
            f"Period: {period_label}\n"
            "Extracted PDF text:\n"
            f"{text[:26000]}"
        ),
        api_key=api_key,
        timeout=90,
        purpose="DeepSeek Fundamental Analyst report request",
    )


def _heuristic_report_analysis(text: str, filename: str, period_label: str) -> dict[str, Any]:
    roe = _extract_percentage(text, ("weighted average ROE", "weighted average return on equity", "ROE"))
    net_margin = _extract_percentage(text, ("net margin", "net profit margin"))
    gross_margin = _extract_percentage(text, ("gross margin",))
    asset_turnover = _extract_ratio(text, ("asset turnover",))
    revenue = _extract_amount(text, ("operating revenue", "revenue"))
    net_profit = _extract_amount(text, ("net profit",))
    total_assets = _extract_amount(text, ("total assets",))
    total_equity = _extract_amount(text, ("total equity", "shareholders' equity", "owners' equity"))
    operating_cash_flow = _extract_amount(text, ("net cash flow from operating activities", "operating cash flow"))
    inventory = _extract_amount(text, ("inventory",))
    accounts_receivable = _extract_amount(text, ("accounts receivable",))
    overseas_share = _extract_percentage(text, ("overseas revenue share", "overseas"))

    pm_value = net_margin if net_margin is not None else 0.12
    ato_value = asset_turnover if asset_turnover is not None else 0.8
    em_value = _compute_em_from_indicator_strings(total_assets, total_equity) or 1.8
    reported_roa = pm_value * ato_value
    reported_roe = roe if roe is not None else reported_roa * em_value

    return {
        "period": period_label,
        "report_name": filename,
        "summary": "Used local pypdf text extraction fallback because Fundamental Analyst LLM was unavailable.",
        "key_indicators": {
            "revenue": revenue or "unavailable",
            "net_profit": net_profit or "unavailable",
            "total_assets": total_assets or "unavailable",
            "total_equity": total_equity or "unavailable",
            "operating_cash_flow": operating_cash_flow or "unavailable",
            "inventory": inventory or "unavailable",
            "accounts_receivable": accounts_receivable or "unavailable",
            "gross_margin": _fmt_percent(gross_margin),
            "net_margin": _fmt_percent(net_margin),
            "asset_turnover": _fmt_number(asset_turnover),
            "equity_multiplier": _fmt_number(em_value),
            "reported_roa": round(float(reported_roa), 6),
            "weighted_average_roe": round(float(reported_roe), 6),
            "overseas_revenue_share": _fmt_percent(overseas_share),
        },
        "derived_state": {
            "profit_margin": float(pm_value),
            "asset_turnover": float(ato_value),
            "equity_multiplier": float(em_value),
            "reported_roa": float(reported_roa),
            "reported_roe": float(reported_roe),
        },
        "fundamental_analyst_report": (
            f"{period_label} fallback Fundamental Analyst report. Parsed values are cumulative/stock-aware when available; "
            "defaults are used where pypdf text lacks machine-readable figures."
        ),
        "calculation_notes": "Fallback heuristic: ROE is cumulative if directly parsed; EM is assets/equity when both are parsed.",
    }


def _normalize_report_analysis(report_analysis: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(report_analysis)
    indicators = dict(normalized.get("key_indicators", {}))
    derived_state = dict(normalized.get("derived_state", {}))

    profit_margin = _coerce_optional_ratio(derived_state.get("profit_margin"))
    asset_turnover = _coerce_optional_number(derived_state.get("asset_turnover"))
    equity_multiplier = _coerce_optional_number(derived_state.get("equity_multiplier"))
    reported_roe = _coerce_optional_ratio(
        derived_state.get("reported_roe", indicators.get("weighted_average_roe", indicators.get("roe")))
    )
    reported_roa = _coerce_optional_ratio(derived_state.get("reported_roa", indicators.get("reported_roa")))

    if profit_margin is None:
        profit_margin = _coerce_optional_ratio(indicators.get("net_margin")) or 0.12
    if asset_turnover is None:
        asset_turnover = _coerce_optional_number(indicators.get("asset_turnover")) or 0.8
    if equity_multiplier is None:
        equity_multiplier = _coerce_optional_number(indicators.get("equity_multiplier"))
    if equity_multiplier is None:
        equity_multiplier = _compute_em_from_indicator_strings(
            indicators.get("total_assets"),
            indicators.get("total_equity"),
        )
    if equity_multiplier is None:
        if reported_roe is not None and profit_margin and asset_turnover:
            denominator = profit_margin * asset_turnover
            equity_multiplier = reported_roe / denominator if abs(denominator) > 1e-12 else 1.8
        else:
            equity_multiplier = 1.8
    if reported_roe is None:
        reported_roe = profit_margin * asset_turnover * equity_multiplier
    if reported_roa is None:
        reported_roa = profit_margin * asset_turnover

    normalized["key_indicators"] = indicators
    normalized["key_indicators"]["reported_roa"] = round(float(reported_roa), 6)
    normalized["derived_state"] = {
        "profit_margin": float(profit_margin),
        "asset_turnover": float(asset_turnover),
        "equity_multiplier": float(equity_multiplier),
        "reported_roa": float(reported_roa),
        "reported_roe": float(reported_roe),
    }
    if "fundamental_analyst_report" not in normalized and "student_fundamental_report" in normalized:
        normalized["fundamental_analyst_report"] = normalized["student_fundamental_report"]
    normalized["student_fundamental_report"] = normalized.get("fundamental_analyst_report", "")
    normalized.setdefault("calculation_notes", "No calculation notes provided.")
    return normalized


def _extract_percentage(text: str, labels: tuple[str, ...]) -> float | None:
    for label in labels:
        match = re.search(rf"{re.escape(label)}[^\d\-]{{0,30}}(-?\d+(?:\.\d+)?)\s*%", text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)) / 100.0
    return None


def _extract_ratio(text: str, labels: tuple[str, ...]) -> float | None:
    for label in labels:
        match = re.search(rf"{re.escape(label)}[^\d\-]{{0,30}}(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _extract_amount(text: str, labels: tuple[str, ...]) -> str | None:
    for label in labels:
        match = re.search(rf"{re.escape(label)}(.{{0,40}}?\d[\d,\.]*\s*[A-Za-z%]*)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _fmt_percent(value: float | None) -> str:
    return "unavailable" if value is None else f"{value * 100:.2f}%"


def _fmt_number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def _coerce_optional_ratio(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric / 100.0 if abs(numeric) > 1.5 and abs(numeric) <= 100 else numeric
    if isinstance(value, str):
        match = re.search(r"(-?\d+(?:\.\d+)?)", value.replace(",", ""))
        if not match:
            return None
        numeric = float(match.group(1))
        if "%" in value:
            return numeric / 100.0
        return numeric / 100.0 if abs(numeric) > 1.5 and abs(numeric) <= 100 else numeric
    return None


def _coerce_optional_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"(-?\d+(?:\.\d+)?)", value.replace(",", ""))
        if not match:
            return None
        return float(match.group(1))
    return None


def _compute_em_from_indicator_strings(total_assets: Any, total_equity: Any) -> float | None:
    assets = _coerce_optional_number(total_assets)
    equity = _coerce_optional_number(total_equity)
    if assets is None or equity is None or abs(equity) <= 1e-12:
        return None
    return assets / equity
