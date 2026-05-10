"""Event extraction tools."""

from __future__ import annotations

import json
from pathlib import Path

from forecasting_system.config import (
    EDA_DEEPSEEK_MODEL,
    EVENT_LIBRARY_PATH,
)
from forecasting_system.exceptions import SchemaValidationError
from forecasting_system.tools.deepseek_client import chat_completion, resolve_api_key
from forecasting_system.types import Event


EVENT_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "event.schema.json"
MAX_EXTRACTION_RETRIES = 3
MAX_DEDUPLICATION_RETRIES = 3
MAX_BATCH_EXTRACTION_RETRIES = 3
DEFAULT_QUARTER_BATCH_SIZE = 12


def extract_event(news_item: str, api_key: str | None = None, max_retries: int = MAX_EXTRACTION_RETRIES) -> Event:
    """Convert one raw news string into a validated structured event."""
    if not isinstance(news_item, str) or not news_item.strip():
        raise SchemaValidationError("extract_event() requires a non-empty raw news text string.")

    active_api_key = resolve_api_key(api_key, purpose="event extraction")
    event_library = load_event_library()
    prompt = _build_event_prompt(news_item, event_library)
    last_error: Exception | None = None

    for attempt in range(max_retries):
        response_payload = _post_chat_completion(prompt, active_api_key)
        try:
            event = _parse_event_response(response_payload)
            _validate_event(event, event_library)
            return event
        except (json.JSONDecodeError, KeyError, TypeError, SchemaValidationError) as exc:
            last_error = exc
            prompt = _build_retry_prompt(news_item, event_library, attempt + 1)

    raise SchemaValidationError(f"DeepSeek event extraction failed after {max_retries} attempts: {last_error}")


def deduplicate_news_semantically(
    news_items: list[str],
    api_key: str | None = None,
    max_retries: int = MAX_DEDUPLICATION_RETRIES,
) -> dict:
    """Use DeepSeek to merge only semantically duplicate news within one quarter."""
    cleaned = [str(item).strip() for item in news_items if str(item).strip()]
    if not cleaned:
        return {"kept_news": [], "duplicate_groups": []}

    active_api_key = resolve_api_key(api_key, purpose="semantic news deduplication")
    prompt = _build_semantic_dedup_prompt(cleaned)
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response_payload = _post_chat_completion(prompt, active_api_key)
            payload = _parse_semantic_dedup_response(response_payload)
            return _validate_semantic_dedup_payload(payload, cleaned)
        except (json.JSONDecodeError, KeyError, TypeError, SchemaValidationError) as exc:
            last_error = exc
            prompt = _build_semantic_dedup_retry_prompt(cleaned, attempt + 1)

    raise SchemaValidationError(f"DeepSeek semantic deduplication failed after retries: {last_error}")


def extract_quarter_events(
    news_items: list[str],
    api_key: str | None = None,
    max_retries: int = MAX_BATCH_EXTRACTION_RETRIES,
    batch_size: int = DEFAULT_QUARTER_BATCH_SIZE,
) -> dict:
    """Deduplicate and extract events for one quarter in a single LLM request."""
    cleaned = [str(item).strip() for item in news_items if str(item).strip()]
    if not cleaned:
        return {"deduped_news": [], "duplicate_groups": [], "events": []}
    if batch_size <= 0:
        raise SchemaValidationError("batch_size must be positive.")
    if len(cleaned) > batch_size:
        return _extract_quarter_events_in_batches(
            cleaned,
            api_key=api_key,
            max_retries=max_retries,
            batch_size=batch_size,
        )

    active_api_key = resolve_api_key(api_key, purpose="quarterly batch event extraction")
    event_library = load_event_library()
    prompt = _build_quarter_event_prompt(cleaned, event_library)
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response_payload = _post_chat_completion(prompt, active_api_key)
            payload = _parse_quarter_event_response(response_payload)
            return _validate_quarter_event_payload(payload, cleaned, event_library)
        except (json.JSONDecodeError, KeyError, TypeError, SchemaValidationError) as exc:
            last_error = exc
            prompt = _build_quarter_event_retry_prompt(cleaned, event_library, attempt + 1)

    raise SchemaValidationError(f"DeepSeek quarterly batch event extraction failed after retries: {last_error}")


def _extract_quarter_events_in_batches(
    news_items: list[str],
    api_key: str | None,
    max_retries: int,
    batch_size: int,
) -> dict:
    deduped_news: list[str] = []
    duplicate_groups: list[dict] = []
    events: list[Event] = []
    seen_signatures: list[set[str]] = []

    for start in range(0, len(news_items), batch_size):
        batch = news_items[start : start + batch_size]
        payload = extract_quarter_events(
            batch,
            api_key=api_key,
            max_retries=max_retries,
            batch_size=batch_size,
        )
        duplicate_groups.extend(payload["duplicate_groups"])
        for title, event in zip(payload["deduped_news"], payload["events"]):
            signature = _character_bigrams(_normalize_title_for_dedup(title))
            duplicate_index = _find_similar_signature(signature, seen_signatures)
            if duplicate_index is not None:
                canonical = deduped_news[duplicate_index]
                duplicate_groups.append(
                    {
                        "canonical": canonical,
                        "duplicates": [title],
                        "reason": "Cross-batch Python title similarity deduplication.",
                    }
                )
                continue
            deduped_news.append(title)
            events.append(event)
            seen_signatures.append(signature)

    return {
        "deduped_news": deduped_news,
        "duplicate_groups": duplicate_groups,
        "events": events,
    }


def validate_event(event: dict) -> Event:
    """Validate one event dict against the local event schema contract."""
    _validate_event(event, load_event_library())
    return event


def validate_event_library(event_library: dict) -> dict:
    """Validate one in-memory event library."""
    if not isinstance(event_library, dict) or not event_library:
        raise SchemaValidationError("Event library must be a non-empty object.")
    for scenario, scenario_data in event_library.items():
        if not isinstance(scenario, str) or not scenario:
            raise SchemaValidationError("Event library scenario keys must be non-empty strings.")
        event_types = scenario_data.get("event_types") if isinstance(scenario_data, dict) else None
        if not isinstance(event_types, list) or not event_types:
            raise SchemaValidationError(f"Event library scenario {scenario!r} must define non-empty event_types.")
        if not all(isinstance(item, str) and item.strip() for item in event_types):
            raise SchemaValidationError(f"Event library scenario {scenario!r} contains invalid event_types.")
    return event_library


def load_event_library(path: str | Path = EVENT_LIBRARY_PATH) -> dict:
    """Load the controlled event taxonomy from local offline JSON."""
    event_library = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return validate_event_library(event_library)


def _build_event_prompt(news_item: str, event_library: dict) -> str:
    taxonomy_json = json.dumps(event_library, ensure_ascii=False, indent=2)
    return (
        "Extract exactly one event from the news text.\n"
        "Return JSON only. Do not include markdown, explanation, or extra text.\n"
        "You must use the controlled event taxonomy below.\n"
        "Choose scenario only from the top-level taxonomy keys.\n"
        "Choose event_type only from the event_types list under the chosen scenario.\n"
        "Never invent new scenario or event_type labels.\n"
        "If no taxonomy category fits the news, still choose the closest valid scenario and event_type, and set noise to true.\n"
        "If a taxonomy category fits, set noise to false.\n"
        f"Controlled event taxonomy:\n{taxonomy_json}\n"
        f"{_direction_semantics_text()}\n"
        f"{_few_shot_examples_text()}\n"
        "Required JSON fields:\n"
        '- "scenario": one top-level key from the controlled taxonomy.\n'
        '- "event_type": one event_types value under the chosen scenario.\n'
        '- "direction": one of "increase", "decrease", "neutral"; it is the movement of event_type itself.\n'
        '- "strength": numeric score between 0 and 1.\n'
        '- "relevance": numeric score between 0 and 1.\n'
        '- "lag": non-negative integer number of weeks before impact starts.\n'
        '- "duration": non-negative integer number of weeks the event remains active.\n'
        '- "noise": boolean; true only when no controlled category fits.\n'
        "Return a single JSON object with those fields only.\n"
        f"News text:\n{news_item}"
    )


def _build_retry_prompt(news_item: str, event_library: dict, attempt_number: int) -> str:
    taxonomy_json = json.dumps(event_library, ensure_ascii=False, indent=2)
    return (
        f"Retry attempt {attempt_number}. "
        "Return one valid JSON object only, with keys scenario, event_type, direction, strength, relevance, lag, duration, noise. "
        "Use only the controlled taxonomy below. "
        "scenario must be a top-level taxonomy key. "
        "event_type must belong to the selected scenario's event_types. "
        "If no category fits, use the closest valid taxonomy category and set noise to true. "
        'direction must be exactly one of "increase", "decrease", "neutral". '
        "direction is the movement of the selected event_type itself, not sentiment and not company impact direction. "
        "strength and relevance must be numbers. lag and duration must be integers. noise must be boolean.\n"
        f"Controlled event taxonomy:\n{taxonomy_json}\n"
        f"{_direction_semantics_text()}\n"
        f"{_few_shot_examples_text()}\n"
        f"News text:\n{news_item}"
    )


def _direction_semantics_text() -> str:
    return (
        "Direction semantics:\n"
        "- Direction describes the movement of the selected event_type itself.\n"
        "- Direction is not news sentiment.\n"
        "- Direction is not the expected impact on Sungrow or any company financial metric.\n"
        "- For trade_barrier: increase means barriers, tariffs, or restrictions become stronger; "
        "decrease means barriers, tariffs, or restrictions become weaker.\n"
        "- For demand: increase means demand rises; decrease means demand falls.\n"
        "- For upstream_price: increase means upstream/input prices rise; decrease means upstream/input prices fall.\n"
        "- For export: increase means export volume/value/orders rise; decrease means export volume/value/orders fall.\n"
        "- Use neutral only when the event_type is present but no clear movement is stated."
    )


def _few_shot_examples_text() -> str:
    examples = [
        {
            "news": "The United States raised tariffs on imported solar equipment.",
            "output": {
                "scenario": "overseas_trade",
                "event_type": "trade_barrier",
                "direction": "increase",
                "strength": 0.8,
                "relevance": 0.9,
                "lag": 0,
                "duration": 4,
                "noise": False,
            },
        },
        {
            "news": "European regulators reduced tariff pressure on Chinese solar exports.",
            "output": {
                "scenario": "overseas_trade",
                "event_type": "trade_barrier",
                "direction": "decrease",
                "strength": 0.7,
                "relevance": 0.8,
                "lag": 0,
                "duration": 4,
                "noise": False,
            },
        },
        {
            "news": "Sungrow reported a decline in export orders from overseas markets.",
            "output": {
                "scenario": "overseas_trade",
                "event_type": "export",
                "direction": "decrease",
                "strength": 0.7,
                "relevance": 0.9,
                "lag": 0,
                "duration": 3,
                "noise": False,
            },
        },
        {
            "news": "Domestic demand for solar inverters continued to rise.",
            "output": {
                "scenario": "sales_demand",
                "event_type": "demand",
                "direction": "increase",
                "strength": 0.7,
                "relevance": 0.8,
                "lag": 0,
                "duration": 3,
                "noise": False,
            },
        },
        {
            "news": "Polysilicon and other upstream material prices fell this week.",
            "output": {
                "scenario": "raw_materials",
                "event_type": "upstream_price",
                "direction": "decrease",
                "strength": 0.6,
                "relevance": 0.8,
                "lag": 0,
                "duration": 2,
                "noise": False,
            },
        },
    ]
    return "Few-shot examples:\n" + json.dumps(examples, ensure_ascii=False, indent=2)


def _build_semantic_dedup_prompt(news_items: list[str]) -> str:
    return (
        "You are deduplicating financial news within one quarter before event extraction.\n"
        "Return JSON only. Merge two items only when they describe the same concrete news fact "
        "or the same announcement in semantically similar wording.\n"
        "Do not merge merely because they share the same event type, topic, company, or industry.\n"
        "Prefer keeping the most specific title in each duplicate group.\n"
        "Required JSON shape:\n"
        "{\n"
        '  "kept_news": ["string"],\n'
        '  "duplicate_groups": [\n'
        '    {"canonical": "string", "duplicates": ["string"], "reason": "string"}\n'
        "  ]\n"
        "}\n"
        "News items:\n"
        f"{json.dumps(news_items, ensure_ascii=False, indent=2)}"
    )


def _build_semantic_dedup_retry_prompt(news_items: list[str], attempt_number: int) -> str:
    return (
        f"Retry attempt {attempt_number}. Return one strict JSON object only with keys kept_news and duplicate_groups. "
        "Every kept_news item must be copied verbatim from the input list. "
        "Only merge semantically duplicate reports about the same concrete news fact.\n"
        + _build_semantic_dedup_prompt(news_items)
    )


def _parse_semantic_dedup_response(response_payload: dict) -> dict:
    content = response_payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise SchemaValidationError("Semantic dedup response must decode to a JSON object.")
    return parsed


def _validate_semantic_dedup_payload(payload: dict, original_news: list[str]) -> dict:
    if set(payload) != {"kept_news", "duplicate_groups"}:
        raise SchemaValidationError("Semantic dedup payload must contain only kept_news and duplicate_groups.")
    if not isinstance(payload["kept_news"], list):
        raise SchemaValidationError("kept_news must be a list.")
    original_set = set(original_news)
    kept = []
    for item in payload["kept_news"]:
        if not isinstance(item, str) or item not in original_set:
            raise SchemaValidationError("kept_news items must be copied verbatim from input news.")
        if item not in kept:
            kept.append(item)
    if not kept:
        kept = original_news[:]

    groups = []
    if not isinstance(payload["duplicate_groups"], list):
        raise SchemaValidationError("duplicate_groups must be a list.")
    for group in payload["duplicate_groups"]:
        if not isinstance(group, dict):
            continue
        canonical = group.get("canonical")
        duplicates = group.get("duplicates", [])
        reason = group.get("reason", "")
        if canonical not in original_set or not isinstance(duplicates, list):
            continue
        valid_duplicates = [item for item in duplicates if isinstance(item, str) and item in original_set and item != canonical]
        if valid_duplicates:
            groups.append({"canonical": canonical, "duplicates": valid_duplicates, "reason": str(reason)})
    return {"kept_news": kept, "duplicate_groups": groups}


def _build_quarter_event_prompt(news_items: list[str], event_library: dict) -> str:
    taxonomy_json = json.dumps(event_library, ensure_ascii=False, indent=2)
    return (
        "You are the EDA Agent for a quarterly financial forecasting training set.\n"
        "Perform semantic deduplication and event extraction in one step.\n"
        "Return JSON only. Do not include markdown or commentary.\n"
        "Deduplication rule: merge items only when they describe the same concrete news fact or same announcement. "
        "Do not merge merely because they share the same company, industry, topic, scenario, or event_type.\n"
        "For each kept canonical news title, extract exactly one event using only the controlled taxonomy.\n"
        "If no taxonomy category fits a canonical title, choose the closest valid scenario/event_type and set noise=true.\n"
        "Lag/duration are quarterly timing fields, not weeks. Be conservative and stable:\n"
        "- For ordinary recurring news, use lag=0 and duration=1.\n"
        "- Use lag=1 only when the event normally affects fundamentals after revenue recognition, capacity ramp, financing completion, or balance-sheet settlement.\n"
        "- Use lag=2 only for clearly delayed projects such as large factories, long-cycle overseas orders, or construction-heavy capacity.\n"
        "- Use duration=2 only for structural events likely to affect more than one quarter.\n"
        "- Use duration=3 or 4 only for very large capacity, financing, policy, litigation, or business-model events with explicit evidence.\n"
        "- For noise=true, set lag=0 and duration=0.\n"
        "- Do not invent lag/duration just to make the event look sophisticated; timing must be justified by the title.\n"
        f"Controlled event taxonomy:\n{taxonomy_json}\n"
        f"{_direction_semantics_text()}\n"
        f"{_few_shot_examples_text()}\n"
        "Required JSON shape:\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "canonical_news": "string copied verbatim from input",\n'
        '      "duplicates": ["string copied verbatim from input"],\n'
        '      "dedup_reason": "string",\n'
        '      "event": {\n'
        '        "scenario": "string",\n'
        '        "event_type": "string",\n'
        '        "direction": "increase" | "decrease" | "neutral",\n'
        '        "strength": 0.0,\n'
        '        "relevance": 0.0,\n'
        '        "lag": 0,\n'
        '        "duration": 0,\n'
        '        "noise": false\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Input news titles:\n"
        f"{json.dumps(news_items, ensure_ascii=False, indent=2)}"
    )


def _build_quarter_event_retry_prompt(news_items: list[str], event_library: dict, attempt_number: int) -> str:
    return (
        f"Retry attempt {attempt_number}. Return strict JSON only with top-level key items. "
        "Every canonical_news and duplicate must be copied verbatim from input. "
        "Every event must contain exactly scenario, event_type, direction, strength, relevance, lag, duration, noise. "
        "Use only the controlled taxonomy; never invent scenario/event_type.\n"
        + _build_quarter_event_prompt(news_items, event_library)
    )


def _parse_quarter_event_response(response_payload: dict) -> dict:
    content = response_payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise SchemaValidationError("Quarter event extraction response must decode to a JSON object.")
    return parsed


def _validate_quarter_event_payload(payload: dict, original_news: list[str], event_library: dict) -> dict:
    if set(payload) != {"items"}:
        raise SchemaValidationError("Quarter event extraction payload must contain only items.")
    if not isinstance(payload["items"], list):
        raise SchemaValidationError("Quarter event extraction items must be a list.")

    original_set = set(original_news)
    seen: set[str] = set()
    deduped_news: list[str] = []
    duplicate_groups: list[dict] = []
    events: list[Event] = []

    for item in payload["items"]:
        if not isinstance(item, dict):
            raise SchemaValidationError("Each quarter event item must be an object.")
        required = {"canonical_news", "duplicates", "dedup_reason", "event"}
        missing = required.difference(item)
        if missing:
            raise SchemaValidationError(f"Quarter event item is missing keys: {sorted(missing)}")
        canonical = item["canonical_news"]
        if not isinstance(canonical, str) or canonical not in original_set:
            raise SchemaValidationError("canonical_news must be copied verbatim from input.")
        if canonical in seen:
            continue
        seen.add(canonical)
        deduped_news.append(canonical)

        duplicates = item["duplicates"]
        if not isinstance(duplicates, list):
            raise SchemaValidationError("duplicates must be a list.")
        valid_duplicates = []
        for duplicate in duplicates:
            if isinstance(duplicate, str) and duplicate in original_set and duplicate != canonical:
                valid_duplicates.append(duplicate)
        if valid_duplicates:
            duplicate_groups.append(
                {
                    "canonical": canonical,
                    "duplicates": valid_duplicates,
                    "reason": str(item.get("dedup_reason", "")),
                }
            )

        event = item["event"]
        if not isinstance(event, dict):
            raise SchemaValidationError("event must be an object.")
        _validate_event(event, event_library)
        events.append(event)

    if not deduped_news:
        raise SchemaValidationError("Quarter event extraction returned no canonical news.")
    return {"deduped_news": deduped_news, "duplicate_groups": duplicate_groups, "events": events}


def _normalize_title_for_dedup(title: str) -> str:
    return "".join(ch for ch in str(title).lower() if ch.isalnum())


def _character_bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[index : index + 2] for index in range(len(text) - 1)}


def _find_similar_signature(signature: set[str], signatures: list[set[str]], threshold: float = 0.86) -> int | None:
    if not signature:
        return None
    for index, existing in enumerate(signatures):
        if _jaccard(signature, existing) >= threshold:
            return index
    return None


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def _post_chat_completion(prompt: str, api_key: str) -> dict:
    return chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "You are an event extraction service. "
                    "Return only one compact JSON object and no additional text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        api_key=api_key,
        model=EDA_DEEPSEEK_MODEL,
        response_format={"type": "json_object"},
        timeout=60,
        purpose="DeepSeek event extraction request",
    )


def _parse_event_response(response_payload: dict) -> Event:
    content = response_payload["choices"][0]["message"]["content"]
    event = json.loads(content)
    if not isinstance(event, dict):
        raise SchemaValidationError("DeepSeek response content must decode to a JSON object.")
    return event


def _validate_event(event: dict, event_library: dict) -> None:
    schema = json.loads(EVENT_SCHEMA_PATH.read_text(encoding="utf-8"))
    required_fields = schema["required"]
    missing = [field for field in required_fields if field not in event]
    if missing:
        raise SchemaValidationError(f"Event is missing required fields: {missing}")
    allowed_fields = set(schema["properties"])
    extra_fields = sorted(set(event).difference(allowed_fields))
    if extra_fields:
        raise SchemaValidationError(f"Event contains unexpected fields: {extra_fields}")

    if not isinstance(event["scenario"], str) or not event["scenario"].strip():
        raise SchemaValidationError("Event field 'scenario' must be a non-empty string.")
    if not isinstance(event["event_type"], str) or not event["event_type"].strip():
        raise SchemaValidationError("Event field 'event_type' must be a non-empty string.")
    if event["direction"] not in {"increase", "decrease", "neutral"}:
        raise SchemaValidationError("Event field 'direction' must be increase, decrease, or neutral.")
    if not isinstance(event["strength"], (int, float)):
        raise SchemaValidationError("Event field 'strength' must be numeric.")
    if not 0.0 <= float(event["strength"]) <= 1.0:
        raise SchemaValidationError("Event field 'strength' must be between 0 and 1.")
    if not isinstance(event["relevance"], (int, float)):
        raise SchemaValidationError("Event field 'relevance' must be numeric.")
    if not 0.0 <= float(event["relevance"]) <= 1.0:
        raise SchemaValidationError("Event field 'relevance' must be between 0 and 1.")
    if not isinstance(event["lag"], int) or event["lag"] < 0:
        raise SchemaValidationError("Event field 'lag' must be a non-negative integer.")
    if not isinstance(event["duration"], int) or event["duration"] < 0:
        raise SchemaValidationError("Event field 'duration' must be a non-negative integer.")
    if not isinstance(event["noise"], bool):
        raise SchemaValidationError("Event field 'noise' must be a boolean.")
    if event["noise"]:
        event["lag"] = 0
        event["duration"] = 0
    else:
        event["lag"] = min(int(event["lag"]), 4)
        event["duration"] = min(max(int(event["duration"]), 1), 4)

    scenario = event["scenario"]
    event_type = event["event_type"]
    if scenario not in event_library:
        raise SchemaValidationError(f"Event scenario {scenario!r} is not in the controlled event library.")
    allowed_event_types = event_library[scenario]["event_types"]
    if event_type not in allowed_event_types:
        raise SchemaValidationError(
            f"Event type {event_type!r} does not belong to controlled scenario {scenario!r}."
        )
