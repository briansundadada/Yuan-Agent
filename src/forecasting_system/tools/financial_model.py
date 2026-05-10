"""Core financial modeling functions for phase 1."""

from __future__ import annotations


def linear_adjustment(base_impact: float, relevance: float, strength: float) -> float:
    """Compute a linear impact delta from base impact, relevance, and strength."""
    return base_impact * relevance * strength


def apply_factor_update(current_value: float, delta: float) -> float:
    """Apply one factor delta to the current value."""
    return current_value + delta


def compute_roe(profit_margin: float, asset_turnover: float, equity_multiplier: float) -> float:
    """Compute ROE using a DuPont-style formula."""
    return profit_margin * asset_turnover * equity_multiplier


def compute_roa(profit_margin: float, asset_turnover: float) -> float:
    """Backward-compatible ROA helper retained for older callers."""
    return profit_margin * asset_turnover
