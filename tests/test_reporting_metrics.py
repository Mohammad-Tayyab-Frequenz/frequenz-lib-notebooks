# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for the reporting metrics helpers."""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from frequenz.lib.notebooks.reporting.metrics import reporting_metrics as metrics


def test_asset_production_respects_sign_and_missing_values() -> None:
    """Production is negated and NaNs stay missing."""
    production = pd.Series([-10, 5, None], index=pd.RangeIndex(3))

    result = metrics.asset_production(production)
    expected = pd.Series([10.0, 0.0, float("nan")], index=production.index)
    assert_series_equal(result, expected)


def test_production_excess_clips_negative_surplus() -> None:
    """Excess is computed after sign correction and clipped at zero."""
    production = pd.Series([-5.0, -1.0, -10.0], index=pd.RangeIndex(3))
    consumption = pd.Series([3.0, 2.0, 1.0], index=production.index)

    result = metrics.production_excess(production, consumption)
    expected = pd.Series([2.0, 0.0, 9.0], index=production.index)
    assert_series_equal(result, expected)


def test_grid_feed_in_prefers_measured_grid_and_infers_when_missing() -> None:
    """Measured grid export is used; otherwise it is inferred from PSC inputs."""
    production = pd.Series([-8, -3], index=pd.RangeIndex(2))
    consumption = pd.Series([2, 1], index=production.index)
    battery = pd.Series([5, 10], index=production.index)

    grid = pd.Series([-4, 2], index=production.index)
    expected_measured = pd.Series([4.0, 0.0], index=production.index)
    measured = metrics.grid_feed_in(production, consumption, battery, grid=grid)
    assert_series_equal(measured, expected_measured)

    inferred = metrics.grid_feed_in(production, consumption, battery, grid=None)
    expected_inferred = pd.Series([1.0, 0.0], index=production.index)
    assert_series_equal(inferred, expected_inferred)

    with pytest.raises(ValueError):
        metrics.grid_feed_in(None, None, None, grid=None)


def test_production_self_consumption_warns_on_negative_values() -> None:
    """A warning is raised when computed self-consumption drops below zero."""
    production = pd.Series([-5.0, -5.0], index=pd.RangeIndex(2))
    consumption = pd.Series([-10.0, 1.0], index=production.index)

    with pytest.warns(UserWarning):
        result = metrics.production_self_consumption(production, consumption)

    expected = pd.Series([-10.0, 1.0], index=production.index)
    assert_series_equal(result, expected)


def test_production_self_share_masks_zero_or_negative_production() -> None:
    """Self-consumption share returns NaN where production is not positive."""
    production = pd.Series([-4.0, 0.0], index=pd.RangeIndex(2))
    consumption = pd.Series([3.0, 3.0], index=production.index)

    result = metrics.production_self_share(production, consumption)
    expected = pd.Series([0.75, float("nan")], index=production.index)
    assert_series_equal(result, expected)


def test_production_self_usage_masks_zero_or_negative_consumption() -> None:
    """Self-usage returns NaN where consumption is not positive."""
    production = pd.Series([-4.0, -2.0], index=pd.RangeIndex(2))
    consumption = pd.Series([0.0, -1.0], index=production.index)

    with pytest.warns(UserWarning):
        result = metrics.production_self_usage(production, consumption)
    expected = pd.Series([float("nan"), float("nan")], index=production.index)
    assert_series_equal(result, expected)


def test_production_self_usage_computes_expected_values() -> None:
    """Self-usage computes expected ratios with PSC inputs."""
    production = pd.Series([-10.0, -5.0], index=pd.RangeIndex(2))
    consumption = pd.Series([8.0, 2.0], index=production.index)

    usage = metrics.production_self_usage(production, consumption)
    expected = pd.Series([1.0, 1.0], index=production.index)
    assert_series_equal(usage, expected)


def test_production_self_share_computes_expected_values() -> None:
    """Self-share computes expected ratios with PSC inputs."""
    production = pd.Series([-10.0, -5.0], index=pd.RangeIndex(2))
    consumption = pd.Series([8.0, 2.0], index=production.index)

    share = metrics.production_self_share(production, consumption)
    expected = pd.Series([0.8, 0.4], index=production.index)
    assert_series_equal(share, expected)


def test_consumption_infers_total_and_sets_series_name() -> None:
    """Grid, production, and battery power are combined to infer consumption."""
    grid = pd.Series([5, 4], index=pd.RangeIndex(2))
    production = pd.Series([-1, -2], index=grid.index)
    battery = pd.Series([0.5, -1], index=grid.index)

    result = metrics.consumption(grid, production=production, battery=battery)
    expected = pd.Series([5.5, 7.0], index=grid.index, name="consumption")
    assert_series_equal(result, expected)


def test_consumption_requires_grid_series() -> None:
    """Passing no grid series raises a ValueError."""
    with pytest.raises(ValueError):
        metrics.consumption(None)  # type: ignore[arg-type]


def test_grid_consumption_prefers_measured_grid_and_infers_when_missing() -> None:
    """Measured grid import is clipped, otherwise it is inferred from PSC inputs."""
    grid = pd.Series([-1, 2], index=pd.RangeIndex(2))
    expected_measured = pd.Series([0.0, 2.0], index=grid.index)
    assert_series_equal(
        metrics.grid_consumption(grid, None, None, None), expected_measured
    )

    production = pd.Series([5, 0], index=grid.index)
    consumption = pd.Series([3, 1], index=grid.index)
    battery = pd.Series([-1, 2], index=grid.index)
    inferred = metrics.grid_consumption(None, production, consumption, battery)
    expected_inferred = pd.Series([7.0, 3.0], index=grid.index)
    assert_series_equal(inferred, expected_inferred)

    with pytest.raises(ValueError):
        metrics.grid_consumption(None, None, None, None)


def test_battery_power_flows_splits_charge_and_discharge_sources() -> None:
    """Battery power flow splits production/grid charge and grid/load discharge."""
    (
        production_to_battery,
        grid_to_battery,
        battery_to_grid,
        battery_to_consumption,
    ) = metrics.battery_power_flows(
        grid=pd.Series([-3.0, 7.0, 1.0, -7.0]),
        production=pd.Series([10.0, 2.0, 1.0, 8.0]),
        consumption=pd.Series([4.0, 5.0, 4.0, 3.0]),
        battery=pd.Series([3.0, 4.0, -2.0, -2.0]),
    )

    assert_series_equal(
        production_to_battery,
        pd.Series([3.0, 0.0, 0.0, 0.0], name="production_to_battery"),
    )
    assert_series_equal(
        grid_to_battery, pd.Series([0.0, 4.0, 0.0, 0.0], name="grid_to_battery")
    )
    assert_series_equal(
        battery_to_grid, pd.Series([0.0, 0.0, 0.0, 2.0], name="battery_to_grid")
    )
    assert_series_equal(
        battery_to_consumption,
        pd.Series([0.0, 0.0, 2.0, 0.0], name="battery_to_consumption"),
    )


def test_battery_power_flows_balances_gross_consumption() -> None:
    """Self-use, grid import, and battery-to-load should sum to consumption."""
    grid = pd.Series([2.0, 5.0])
    production = pd.Series([4.0, 3.0])
    consumption = pd.Series([10.0, 8.0])
    _, _, _, battery_to_consumption = metrics.battery_power_flows(
        grid=grid,
        production=production,
        consumption=consumption,
        battery=pd.Series([-4.0, -1.0]),
    )

    expected_battery_to_consumption = pd.Series(
        [4.0, 0.0], name="battery_to_consumption"
    )
    assert_series_equal(battery_to_consumption, expected_battery_to_consumption)
    assert_series_equal(
        production + grid + battery_to_consumption,
        consumption,
        check_names=False,
    )


def test_battery_power_flows_conserves_charge_and_discharge() -> None:
    """Battery flow splits should sum back to measured charge and discharge."""
    battery = pd.Series([4.0, -3.0, -2.0, 5.0])
    (
        production_to_battery,
        grid_to_battery,
        battery_to_grid,
        battery_to_consumption,
    ) = metrics.battery_power_flows(
        grid=pd.Series([1.0, 3.0, 0.0, 0.0]),
        production=pd.Series([13.0, 2.0, 1.0, 5.0]),
        consumption=pd.Series([8.0, 10.0, 3.0, 3.0]),
        battery=battery,
    )

    battery_charge = battery.clip(lower=0)
    battery_discharge = -battery.clip(upper=0)

    assert_series_equal(
        production_to_battery + grid_to_battery,
        battery_charge,
        check_names=False,
    )
    assert_series_equal(
        battery_to_consumption + battery_to_grid,
        battery_discharge,
        check_names=False,
    )


def test_battery_power_flows_caps_production_to_battery_at_actual_charge() -> None:
    """Production-to-battery power is bounded by measured battery charge power."""
    (
        production_to_battery,
        grid_to_battery,
        battery_to_grid,
        battery_to_consumption,
    ) = metrics.battery_power_flows(
        grid=pd.Series([0.0]),
        production=pd.Series([10.0]),
        consumption=pd.Series([4.0]),
        battery=pd.Series([1.0]),
    )

    assert_series_equal(
        production_to_battery, pd.Series([1.0], name="production_to_battery")
    )
    assert_series_equal(grid_to_battery, pd.Series([0.0], name="grid_to_battery"))
    assert_series_equal(battery_to_grid, pd.Series([0.0], name="battery_to_grid"))
    assert_series_equal(
        battery_to_consumption, pd.Series([0.0], name="battery_to_consumption")
    )
