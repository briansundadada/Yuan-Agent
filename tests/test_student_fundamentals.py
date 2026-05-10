import pytest

from forecasting_system.exceptions import SchemaValidationError
from forecasting_system.tools import student_fundamentals


def _mock_response(content: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                }
            }
        ]
    }


def test_analyze_fundamentals_returns_valid_structure(monkeypatch):
    def fake_post_chat_completion(prompt: str, api_key: str) -> dict:
        assert "Do NOT predict ROA directly" in prompt
        assert "Controlled event taxonomy" in prompt
        return _mock_response(
            """
            {
              "company_name": "Sungrow",
              "fundamental_summary": "Sungrow has export exposure, cost sensitivity, and growth from solar storage demand.",
              "baseline_state": {
                "profit_margin": 0.2,
                "asset_turnover": 1.5,
                "equity_multiplier": 2.0
              },
              "core_metrics": {
                "profit_margin": 0.2,
                "asset_turnover": 1.5,
                "equity_multiplier": 2.0,
                "reported_roa": 0.3,
                "reported_roe": 0.6
              },
              "candidate_scenarios": [
                {
                  "scenario": "overseas_trade",
                  "event_type": "export",
                  "reason": "Exports are described as important to revenue."
                },
                {
                  "scenario": "raw_materials",
                  "event_type": "upstream_price",
                  "reason": "Input cost exposure is mentioned."
                }
              ],
              "qualitative_assessment": {
                "export_exposure": "High overseas shipment exposure.",
                "cost_structure": "Sensitive to upstream materials and components.",
                "competitive_position": "Strong inverter and storage position.",
                "growth_driver": "Solar and energy storage demand."
              }
            }
            """
        )

    monkeypatch.setattr(student_fundamentals, "_post_chat_completion", fake_post_chat_completion)

    result = student_fundamentals.analyze_fundamentals(
        raw_text="Sungrow report text with profit margin around 0.20 and asset turnover around 1.50.",
        api_key="test-key",
    )

    assert result["company_name"] == "Sungrow"
    assert result["baseline_state"] == {"profit_margin": 0.2, "asset_turnover": 1.5, "equity_multiplier": 2.0}
    assert result["core_metrics"] == {
        "profit_margin": 0.2,
        "asset_turnover": 1.5,
        "equity_multiplier": 2.0,
        "reported_roa": 0.3,
        "reported_roe": 0.6,
    }
    assert result["candidate_scenarios"][0]["scenario"] == "overseas_trade"
    assert "export_exposure" in result["qualitative_assessment"]


def test_analyze_fundamentals_accepts_local_file(monkeypatch, tmp_path):
    source_file = tmp_path / "report.txt"
    source_file.write_text("Sungrow report summary.", encoding="utf-8")

    def fake_post_chat_completion(prompt: str, api_key: str) -> dict:
        assert "Sungrow report summary." in prompt
        return _mock_response(
            """
            {
              "company_name": "Sungrow",
              "fundamental_summary": "Local file summary was analyzed.",
          "baseline_state": {"profit_margin": 0.18, "asset_turnover": 1.4, "equity_multiplier": 2.1},
          "core_metrics": {"profit_margin": 0.18, "asset_turnover": 1.4, "equity_multiplier": 2.1, "reported_roa": 0.252, "reported_roe": 0.5292},
              "candidate_scenarios": [
                {"scenario": "sales_demand", "event_type": "demand", "reason": "Demand is discussed."}
              ],
              "qualitative_assessment": {
                "export_exposure": "Moderate.",
                "cost_structure": "Component costs matter.",
                "competitive_position": "Competitive.",
                "growth_driver": "Demand growth."
              }
            }
            """
        )

    monkeypatch.setattr(student_fundamentals, "_post_chat_completion", fake_post_chat_completion)

    result = student_fundamentals.analyze_fundamentals_from_file(source_file, api_key="test-key")

    assert result["baseline_state"]["profit_margin"] == 0.18
    assert result["candidate_scenarios"][0]["event_type"] == "demand"


def test_validate_fundamental_understanding_rejects_missing_baseline_metric():
    with pytest.raises(SchemaValidationError):
        student_fundamentals.validate_fundamental_understanding(
            {
                "company_name": "Sungrow",
                "fundamental_summary": "summary",
                "baseline_state": {"profit_margin": 0.2},
                "core_metrics": {"profit_margin": 0.2, "asset_turnover": 1.5, "equity_multiplier": 2.0, "reported_roa": 0.3, "reported_roe": 0.6},
                "candidate_scenarios": [],
                "qualitative_assessment": {
                    "export_exposure": "high",
                    "cost_structure": "cost sensitive",
                    "competitive_position": "strong",
                    "growth_driver": "storage",
                },
            }
        )
