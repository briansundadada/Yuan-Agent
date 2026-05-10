"""Append-only JSONL logging tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forecasting_system.config import RULE_UPDATE_LOG_PATH, STUDENT_LOG_PATH, TEACHER_LOG_PATH


def log_student_record(record: dict[str, Any], path: str | Path = STUDENT_LOG_PATH) -> None:
    """Append one student record to JSONL."""
    _append_jsonl(path, record)


def log_teacher_feedback(record: dict[str, Any], path: str | Path = TEACHER_LOG_PATH) -> None:
    """Append one teacher feedback record to JSONL."""
    _append_jsonl(path, record)


def log_rule_update(record: dict[str, Any], path: str | Path = RULE_UPDATE_LOG_PATH) -> None:
    """Append one rule update record to JSONL."""
    _append_jsonl(path, record)


def _append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
