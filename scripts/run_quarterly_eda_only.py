from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from forecasting_system.tools.events import extract_quarter_events
from forecasting_system.tools.news import preprocess_quarterly_news


NEWS_PATHS = [
    PROJECT_ROOT / "data" / "news" / "news_2021.xlsx",
    PROJECT_ROOT / "data" / "news" / "news_2022.xlsx",
    PROJECT_ROOT / "data" / "news" / "news_2023.xlsx",
]
OUTPUT_DIR = PROJECT_ROOT / "logs" / "quarterly_eda_only"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    quarterly_news = preprocess_quarterly_news(NEWS_PATHS)
    cached_payload = _load_json(OUTPUT_DIR / "quarterly_event_classification.json", default={})
    deduplication_log = []
    event_counter: Counter[str] = Counter()
    summary_rows = []

    for year in (2021, 2022, 2023):
        for quarter, suffix in ((1, "Q1"), (2, "H1"), (3, "Q3"), (4, "FY")):
            period = f"{year}{suffix}"
            raw_titles = quarterly_news.get((year, quarter), [])
            if period in cached_payload:
                payload = cached_payload[period]
                print(f"SKIP {period}: loaded cached EDA output", flush=True)
            else:
                print(f"START {period}: raw_news={len(raw_titles)}", flush=True)
                batch_payload = extract_quarter_events(raw_titles) if raw_titles else {
                    "deduped_news": [],
                    "duplicate_groups": [],
                    "events": [],
                }
                payload = {
                    "period": period,
                    "year": year,
                    "quarter": quarter,
                    "raw_news": raw_titles,
                    "deduped_news": batch_payload["deduped_news"],
                    "duplicate_groups": batch_payload["duplicate_groups"],
                    "events": batch_payload["events"],
                }
                cached_payload[period] = payload
                _write_json(OUTPUT_DIR / "quarterly_event_classification.json", cached_payload)
                print(
                    f"END {period}: deduped_news={len(batch_payload['deduped_news'])}, events={len(batch_payload['events'])}",
                    flush=True,
                )

            non_noise_events = [event for event in payload["events"] if not event.get("noise", False)]
            for event in non_noise_events:
                event_counter[f"{event['scenario']}::{event['event_type']}::{event['direction']}"] += 1

            deduplication_log.append(
                {
                    "period": period,
                    "raw_news_count": len(payload["raw_news"]),
                    "deduped_news_count": len(payload["deduped_news"]),
                    "duplicate_group_count": len(payload["duplicate_groups"]),
                    "duplicate_groups": payload["duplicate_groups"],
                }
            )
            summary_rows.append(
                {
                    "period": period,
                    "raw_news_count": len(payload["raw_news"]),
                    "deduped_news_count": len(payload["deduped_news"]),
                    "event_count": len(payload["events"]),
                    "non_noise_event_count": len(non_noise_events),
                    "noise_event_count": len(payload["events"]) - len(non_noise_events),
                    "duplicate_group_count": len(payload["duplicate_groups"]),
                }
            )

    _write_json(OUTPUT_DIR / "deduplication_log.json", deduplication_log)
    _write_csv(OUTPUT_DIR / "quarterly_eda_summary.csv", summary_rows)
    _write_event_frequency(event_counter, OUTPUT_DIR / "event_frequency.csv")
    print(OUTPUT_DIR)


def _write_event_frequency(counter: Counter[str], output_path: Path) -> None:
    rows = []
    for key, count in counter.most_common():
        scenario, event_type, direction = key.split("::")
        rows.append({"scenario": scenario, "event_type": event_type, "direction": direction, "count": count})
    _write_csv(output_path, rows)


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
