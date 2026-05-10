import json
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent / "src"))

from forecasting_system.tools.student_fundamentals import analyze_fundamentals


SAMPLE_SUNGROW_REPORT_SUMMARY = """
Sungrow is a leading solar inverter and energy storage system company.
The report describes stable gross margin from product mix improvement, ongoing
overseas shipment expansion, and exposure to upstream semiconductor and battery
material costs. Export sales remain important, while demand growth is driven by
utility-scale solar installations and energy storage adoption. The company
continues to invest in manufacturing efficiency and new product competitiveness.
Assume current profit_margin is around 0.20 and asset_turnover is around 1.50.
"""


def main() -> None:
    understanding = analyze_fundamentals(raw_text=SAMPLE_SUNGROW_REPORT_SUMMARY)
    print(json.dumps(understanding, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
