from forecasting_system.agents.eda import EDA
from forecasting_system.tools import events as event_tools


def test_event_prompt_defines_direction_as_event_type_movement():
    prompt = event_tools._build_event_prompt(
        "The United States raised tariffs on imported solar equipment.",
        event_tools.load_event_library(),
    )

    assert "movement of the selected event_type itself" in prompt
    assert "not news sentiment" in prompt
    assert "not the expected impact on Sungrow" in prompt
    assert '"event_type": "trade_barrier"' in prompt
    assert '"direction": "increase"' in prompt
    assert "reduced tariff pressure" in prompt
    assert '"direction": "decrease"' in prompt
    assert '"event_type": "export"' in prompt


def test_extract_event_returns_valid_structure_with_mocked_deepseek(monkeypatch):
    def fake_post_chat_completion(prompt: str, api_key: str) -> dict:
        assert "Controlled event taxonomy" in prompt
        assert "policy_regulation" in prompt
        assert "subsidy" in prompt
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"scenario":"policy_regulation","event_type":"subsidy","direction":"increase",'
                            '"strength":0.8,"relevance":0.9,"lag":1,"duration":4,"noise":false}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(event_tools, "_post_chat_completion", fake_post_chat_completion)

    event = event_tools.extract_event(
        "China announced new renewable-energy subsidy support for inverter makers.",
        api_key="test-key",
    )

    assert event == {
        "scenario": "policy_regulation",
        "event_type": "subsidy",
        "direction": "increase",
        "strength": 0.8,
        "relevance": 0.9,
        "lag": 1,
        "duration": 4,
        "noise": False,
    }


def test_extract_event_retries_until_json_is_valid(monkeypatch):
    responses = iter(
        [
            {"choices": [{"message": {"content": "not json"}}]},
            {
                "choices": [
                    {
                        "message": {
                        "content": (
                                '{"scenario":"overseas_trade","event_type":"export","direction":"decrease",'
                                '"strength":0.7,"relevance":0.8,"lag":0,"duration":2,"noise":false}'
                            )
                        }
                    }
                ]
            },
        ]
    )

    def fake_post_chat_completion(prompt: str, api_key: str) -> dict:
        return next(responses)

    monkeypatch.setattr(event_tools, "_post_chat_completion", fake_post_chat_completion)

    event = event_tools.extract_event(
        "A logistics disruption delayed overseas shipments for solar equipment exporters.",
        api_key="test-key",
    )

    assert event["direction"] == "decrease"
    assert event["scenario"] == "overseas_trade"
    assert event["event_type"] == "export"
    assert event["noise"] is False


def test_extract_event_retries_when_taxonomy_labels_are_invalid(monkeypatch):
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"scenario":"invented","event_type":"made_up","direction":"increase",'
                                '"strength":0.7,"relevance":0.8,"lag":0,"duration":2,"noise":false}'
                            )
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"scenario":"raw_materials","event_type":"upstream_price","direction":"increase",'
                                '"strength":0.7,"relevance":0.8,"lag":0,"duration":2,"noise":false}'
                            )
                        }
                    }
                ]
            },
        ]
    )

    def fake_post_chat_completion(prompt: str, api_key: str) -> dict:
        return next(responses)

    monkeypatch.setattr(event_tools, "_post_chat_completion", fake_post_chat_completion)

    event = event_tools.extract_event(
        "Polysilicon prices increased and may affect inverter sector costs.",
        api_key="test-key",
    )

    event_library = event_tools.load_event_library()
    assert event["scenario"] in event_library
    assert event["event_type"] in event_library[event["scenario"]]["event_types"]


def test_extract_event_noise_fallback_uses_valid_taxonomy(monkeypatch):
    def fake_post_chat_completion(prompt: str, api_key: str) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"scenario":"technology_competition","event_type":"new_product","direction":"neutral",'
                            '"strength":0.0,"relevance":0.0,"lag":0,"duration":0,"noise":true}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(event_tools, "_post_chat_completion", fake_post_chat_completion)

    event = event_tools.extract_event(
        "The company cafeteria added a new lunch menu unrelated to business fundamentals.",
        api_key="test-key",
    )

    event_library = event_tools.load_event_library()
    assert event["noise"] is True
    assert event["scenario"] in event_library
    assert event["event_type"] in event_library[event["scenario"]]["event_types"]


def test_eda_run_extracts_events_from_news_batch(monkeypatch):
    def fake_extract_event(news_item: str, api_key=None, max_retries=3):
        return {
            "scenario": "policy_regulation",
            "event_type": "subsidy",
            "direction": "increase",
            "strength": 0.6,
            "relevance": 0.7,
            "lag": 1,
            "duration": 3,
            "noise": False,
        }

    monkeypatch.setattr("forecasting_system.agents.eda.extract_event", fake_extract_event)

    events = EDA().run(
        [
            "Sample news one.",
            "Sample news two.",
        ]
    )

    assert len(events) == 2
    assert all(event["scenario"] == "policy_regulation" for event in events)
    assert all(event["event_type"] == "subsidy" for event in events)
