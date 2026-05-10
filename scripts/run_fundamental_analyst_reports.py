from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from forecasting_system.agents.fundamental_analyst import FundamentalAnalyst
from forecasting_system.tools.report_analysis import extract_pdf_text, iter_report_period_files


OUTPUT_DIR = PROJECT_ROOT / "logs" / "fundamental_analyst_report_test"
CACHE_DIR = OUTPUT_DIR / "fundamental_analyst_outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    analyst = FundamentalAnalyst()
    report_index = []
    metrics_rows = []
    quality_lines = ["# pypdf Text Quality", ""]

    for period, report_path in iter_report_period_files():
        absolute_path = PROJECT_ROOT / report_path
        text = extract_pdf_text(absolute_path)
        page_count = text.count("[page ")
        char_count = len(text)
        cache_path = CACHE_DIR / f"{period}.json"
        analysis = analyst.run(
            report_path=absolute_path,
            period_label=period,
            cache_path=cache_path,
        )
        indicators = analysis.get("key_indicators", {})
        derived = analysis.get("derived_state", {})

        report_index.append(
            {
                "period": period,
                "path": str(report_path),
                "page_count_with_text": page_count,
                "extracted_char_count": char_count,
                "cache_path": str(cache_path.relative_to(PROJECT_ROOT)),
            }
        )
        metrics_rows.append(
            {
                "period": period,
                "report_name": analysis.get("report_name"),
                "reported_roe_cumulative": derived.get("reported_roe"),
                "equity_multiplier": derived.get("equity_multiplier"),
                "profit_margin": derived.get("profit_margin"),
                "asset_turnover": derived.get("asset_turnover"),
                "revenue_flow_cumulative": indicators.get("revenue"),
                "net_profit_flow_cumulative": indicators.get("net_profit"),
                "total_assets_stock": indicators.get("total_assets"),
                "total_equity_stock": indicators.get("total_equity"),
                "operating_cash_flow_cumulative": indicators.get("operating_cash_flow"),
                "inventory_stock": indicators.get("inventory"),
                "accounts_receivable_stock": indicators.get("accounts_receivable"),
                "gross_margin": indicators.get("gross_margin"),
                "net_margin": indicators.get("net_margin"),
                "overseas_revenue_share": indicators.get("overseas_revenue_share"),
                "calculation_notes": analysis.get("calculation_notes"),
            }
        )
        quality_lines.extend(
            [
                f"## {period}",
                f"- File: `{report_path}`",
                f"- Text pages: `{page_count}`",
                f"- Extracted chars: `{char_count}`",
                f"- pypdf status: `{'ok' if char_count > 1000 else 'weak'}`",
                "",
            ]
        )

    (OUTPUT_DIR / "report_index.json").write_text(
        json.dumps(report_index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "per_period_metrics.json").write_text(
        json.dumps(metrics_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(OUTPUT_DIR / "per_period_metrics.csv", metrics_rows)
    (OUTPUT_DIR / "pypdf_text_quality.md").write_text("\n".join(quality_lines), encoding="utf-8")
    print(OUTPUT_DIR)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
