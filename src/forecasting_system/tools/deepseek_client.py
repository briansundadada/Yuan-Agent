"""Shared DeepSeek chat client helpers."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from http.client import IncompleteRead
from typing import Any

from forecasting_system.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_CHAT_MODEL
from forecasting_system.exceptions import SchemaValidationError


def resolve_api_key(api_key: str | None = None, *, purpose: str = "DeepSeek request") -> str:
    resolved = api_key or DEEPSEEK_API_KEY
    if not resolved:
        raise SchemaValidationError(f"Missing DeepSeek API key. Set DEEPSEEK_API_KEY for {purpose}.")
    return resolved


def chat_completion(
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    model: str = DEEPSEEK_CHAT_MODEL,
    response_format: dict[str, str] | None = None,
    timeout: int = 30,
    purpose: str = "DeepSeek request",
) -> dict[str, Any]:
    active_api_key = resolve_api_key(api_key, purpose=purpose)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    request = urllib.request.Request(
        url=f"{DEEPSEEK_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {active_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started_at = time.perf_counter()
    print(f"[LLM START] {purpose} | model={model}", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            elapsed_seconds = time.perf_counter() - started_at
            usage = payload.get("usage") if isinstance(payload, dict) else None
            prompt_tokens = _usage_int(usage, "prompt_tokens")
            completion_tokens = _usage_int(usage, "completion_tokens")
            total_tokens = _usage_int(usage, "total_tokens")
            payload.setdefault("_diagnostics", {})
            payload["_diagnostics"].update(
                {
                    "purpose": purpose,
                    "model": model,
                    "elapsed_seconds": elapsed_seconds,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
            )
            print(
                "[LLM DONE] "
                f"{purpose} | elapsed={elapsed_seconds:.2f}s | "
                f"prompt_tokens={_fmt_token(prompt_tokens)} | "
                f"completion_tokens={_fmt_token(completion_tokens)} | "
                f"total_tokens={_fmt_token(total_tokens)}",
                flush=True,
            )
            return payload
    except (urllib.error.URLError, IncompleteRead, TimeoutError, OSError) as exc:
        elapsed_seconds = time.perf_counter() - started_at
        print(f"[LLM FAIL] {purpose} | elapsed={elapsed_seconds:.2f}s | error={exc}", flush=True)
        raise SchemaValidationError(f"{purpose} failed: {exc}") from exc


def _usage_int(usage: Any, key: str) -> int | None:
    if isinstance(usage, dict) and isinstance(usage.get(key), int):
        return int(usage[key])
    return None


def _fmt_token(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def chat_json(
    *,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
    timeout: int = 30,
    purpose: str = "DeepSeek JSON request",
) -> dict[str, Any]:
    response = chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        api_key=api_key,
        response_format={"type": "json_object"},
        timeout=timeout,
        purpose=purpose,
    )
    content = response["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise SchemaValidationError(f"{purpose} must return a JSON object.")
    return parsed


def chat_text(
    *,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
    timeout: int = 30,
    purpose: str = "DeepSeek text request",
) -> str:
    response = chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        api_key=api_key,
        timeout=timeout,
        purpose=purpose,
    )
    content = response["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise SchemaValidationError(f"{purpose} returned non-string content.")
    return content
