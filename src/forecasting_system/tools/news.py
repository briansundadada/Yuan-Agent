"""News preprocessing tools for monthly and quarterly pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from forecasting_system.exceptions import PlaceholderNotImplementedError, SchemaValidationError


COMPANY_KEYWORDS = ("阳光", "阳光电源", "Sungrow", "SUNGROW")
MATERIAL_KEYWORDS = (
    "订单",
    "中标",
    "预中标",
    "项目",
    "逆变器",
    "储能",
    "PCS",
    "出货",
    "出口",
    "海外",
    "欧洲",
    "美国",
    "中东",
    "产能",
    "扩产",
    "交付",
    "签约",
    "并网",
    "投产",
    "新品",
    "专利",
    "发布",
    "业绩",
    "营业收入",
    "净利润",
    "ROE",
    "净资产收益率",
    "加权平均净资产收益率",
    "回款",
    "融资",
    "募资",
)
NOISE_KEYWORDS = (
    "融资融券",
    "融资净买入",
    "融资净偿还",
    "获融资买入",
    "融券",
    "两融",
    "成交量",
    "换手率",
    "龙虎榜",
    "收盘",
    "涨停",
    "跌停",
    "股价",
    "市值",
    "研报评级",
    "机构评级",
    "流入资金比例",
)
INSUFFICIENT_INFO_KEYWORDS = (
    "周观点",
    "周观察",
    "排名",
    "图说",
    "研究分析",
    "简报",
    "快讯",
)


def load_news(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load raw news rows from Excel."""
    if path is None:
        raise PlaceholderNotImplementedError("load_news() requires a local Excel path in this pipeline.")
    frame = pd.read_excel(path)
    normalized = _normalize_news_frame(frame)
    return normalized.to_dict(orient="records")


def preprocess_monthly_news(path: str | Path) -> dict[int, list[str]]:
    """Map one year of news into month -> material deduplicated titles."""
    frame = pd.read_excel(path)
    normalized = _normalize_news_frame(frame)
    normalized["month"] = normalized["timestamp"].dt.month

    monthly: dict[int, list[str]] = {}
    for _, row in normalized.iterrows():
        month = int(row["month"])
        title = str(row["title"]).strip()
        if not title or not is_potentially_material_news(title):
            continue
        monthly.setdefault(month, []).append(title)
    return {month: _deduplicate_titles(titles) for month, titles in monthly.items()}


def preprocess_quarterly_news(paths: list[str | Path]) -> dict[tuple[int, int], list[str]]:
    """Map news files into (year, quarter) groups with existing Python noise filtering."""
    quarterly: dict[tuple[int, int], list[str]] = {}
    for path in paths:
        frame = pd.read_excel(path)
        normalized = _normalize_news_frame(frame)
        normalized["year"] = normalized["timestamp"].dt.year
        normalized["quarter"] = normalized["timestamp"].dt.quarter

        for _, row in normalized.iterrows():
            title = str(row["title"]).strip()
            if not title or not is_potentially_material_news(title):
                continue
            key = (int(row["year"]), int(row["quarter"]))
            quarterly.setdefault(key, []).append(title)
    return quarterly


def is_forced_noise_news(news_title: str) -> bool:
    """Apply hard filters requested by the real-data experiment."""
    text = str(news_title).strip()
    if not text:
        return True
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in NOISE_KEYWORDS + INSUFFICIENT_INFO_KEYWORDS)


def is_potentially_material_news(news_title: str) -> bool:
    text = str(news_title).strip()
    if not text or is_forced_noise_news(text):
        return False
    if not any(keyword in text for keyword in COMPANY_KEYWORDS):
        return False
    return any(keyword in text for keyword in MATERIAL_KEYWORDS)


def _normalize_news_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise SchemaValidationError("News file is empty.")

    columns = {str(column).strip(): column for column in frame.columns}
    timestamp_column = _find_first_column(columns, ("时间戳", "timestamp", "date", "日期"))
    title_column = _find_first_column(columns, ("新闻标题", "title", "headline", "标题"))
    if timestamp_column is None or title_column is None:
        raise SchemaValidationError(
            "News file must contain timestamp/title columns such as '时间戳' and '新闻标题'."
        )

    normalized = frame[[timestamp_column, title_column]].copy()
    normalized.columns = ["timestamp", "title"]
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
    normalized["title"] = normalized["title"].astype(str).str.strip()
    normalized = normalized[normalized["timestamp"].notna()].copy()
    normalized = normalized[normalized["title"] != ""].copy()
    return normalized.reset_index(drop=True)


def _find_first_column(columns: dict[str, Any], candidates: tuple[str, ...]) -> Any | None:
    lowered = {name.lower(): value for name, value in columns.items()}
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _deduplicate_titles(titles: list[str], similarity_threshold: float = 0.78, max_items: int = 6) -> list[str]:
    kept: list[str] = []
    signatures: list[set[str]] = []
    for title in titles:
        signature = _character_bigrams(_normalize_title(title))
        if not signature:
            continue
        if any(_jaccard(signature, existing) >= similarity_threshold for existing in signatures):
            continue
        kept.append(title)
        signatures.append(signature)
        if len(kept) >= max_items:
            break
    return kept


def _normalize_title(title: str) -> str:
    normalized = str(title)
    for keyword in COMPANY_KEYWORDS:
        normalized = normalized.replace(keyword, "")
    return "".join(ch for ch in normalized.lower() if ch.isalnum())


def _character_bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[index : index + 2] for index in range(len(text) - 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union if union else 0.0
