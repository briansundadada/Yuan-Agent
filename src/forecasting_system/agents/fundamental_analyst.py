"""Fundamental Analyst Agent for factual financial report analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forecasting_system.tools.report_analysis import load_or_analyze_report_pdf


class FundamentalAnalyst:
    """Analyze disclosed reports into cumulative ROE, EM, and operating indicators."""

    def run(
        self,
        report_path: str | Path,
        period_label: str,
        cache_path: str | Path | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        return load_or_analyze_report_pdf(
            report_path=report_path,
            period_label=period_label,
            cache_path=cache_path,
            api_key=api_key,
        )
