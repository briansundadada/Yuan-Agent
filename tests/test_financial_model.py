from forecasting_system.tools.financial_model import (
    apply_factor_update,
    compute_roa,
    linear_adjustment,
)


def test_linear_adjustment_is_explicit_and_deterministic():
    assert linear_adjustment(0.1, 0.5, 0.4) == 0.020000000000000004


def test_apply_factor_update_adds_delta():
    assert apply_factor_update(0.12, -0.02) == 0.09999999999999999


def test_compute_roa_uses_profit_margin_times_asset_turnover():
    assert compute_roa(0.2, 1.5) == 0.30000000000000004
