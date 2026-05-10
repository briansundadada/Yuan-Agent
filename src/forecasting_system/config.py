"""Central project paths and configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"
EVENT_LIBRARY_PATH = DATA_DIR / "event_library" / "sungrow_event_library.json"
RULES_PATH = DATA_DIR / "rules" / "rules.json"
NEWS_DIR = DATA_DIR / "news"
SUPERVISION_DIR = DATA_DIR / "supervision"

STUDENT_LOG_PATH = LOGS_DIR / "student_records.jsonl"
TEACHER_LOG_PATH = LOGS_DIR / "teacher_feedback.jsonl"
RULE_UPDATE_LOG_PATH = LOGS_DIR / "rule_updates.jsonl"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
EDA_DEEPSEEK_MODEL = os.getenv("EDA_DEEPSEEK_MODEL", "deepseek-v4-flash")

TARGET_METRIC = "roa"
ALLOWED_FACTORS = ("profit_margin", "asset_turnover")
