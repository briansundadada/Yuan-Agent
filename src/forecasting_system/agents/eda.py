"""EDA agent with hard noise filtering for monthly news."""

from __future__ import annotations

import copy

from forecasting_system.tools.events import extract_event
from forecasting_system.tools.news import is_forced_noise_news
from forecasting_system.types import Event


class EDA:
    """Convert news headlines into events while enforcing hard noise rules."""

    def run(self, news_batch: list[str]) -> list[Event]:
        events: list[Event] = []
        for news_item in news_batch:
            if is_forced_noise_news(news_item):
                event = {
                    "scenario": "sales_demand",
                    "event_type": "demand",
                    "direction": "neutral",
                    "strength": 0.0,
                    "relevance": 0.0,
                    "lag": 0,
                    "duration": 0,
                    "noise": True,
                }
            else:
                event = copy.deepcopy(extract_event(news_item))
            events.append(event)
        return events
