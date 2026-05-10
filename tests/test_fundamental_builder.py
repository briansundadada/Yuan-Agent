import pytest

from forecasting_system.exceptions import SchemaValidationError
from forecasting_system.tools.fundamentals import build_baseline_state, load_sungrow_baseline


def test_load_sungrow_baseline_returns_valid_baseline_state():
    baseline = load_sungrow_baseline()

    assert baseline["profit_margin"] == pytest.approx(0.2)
    assert baseline["asset_turnover"] == pytest.approx(1.5)
    assert "source_summary" in baseline
    assert baseline["confidence"] == pytest.approx(0.85)


def test_build_baseline_state_from_text_summary(tmp_path):
    source_file = tmp_path / "sungrow_summary.txt"
    source_file.write_text(
        "company: Sungrow\nprofit_margin: 0.18\nasset_turnover: 1.42\nconfidence: 0.7",
        encoding="utf-8",
    )

    baseline = build_baseline_state(source_file)

    assert baseline["profit_margin"] == pytest.approx(0.18)
    assert baseline["asset_turnover"] == pytest.approx(1.42)
    assert baseline["source_summary"].startswith("company: Sungrow")
    assert baseline["confidence"] == pytest.approx(0.7)


@pytest.mark.parametrize(
    "payload",
    [
        '{"profit_margin": 0.2, "source_summary": "missing asset turnover"}',
        '{"profit_margin": "bad", "asset_turnover": 1.5, "source_summary": "bad type"}',
    ],
)
def test_build_baseline_state_rejects_invalid_json_input(tmp_path, payload):
    source_file = tmp_path / "invalid.json"
    source_file.write_text(payload, encoding="utf-8")

    with pytest.raises(SchemaValidationError):
        build_baseline_state(source_file)


def test_build_baseline_state_rejects_invalid_text_input(tmp_path):
    source_file = tmp_path / "invalid.txt"
    source_file.write_text("company: Sungrow\nprofit_margin: 0.18", encoding="utf-8")

    with pytest.raises(SchemaValidationError):
        build_baseline_state(source_file)
