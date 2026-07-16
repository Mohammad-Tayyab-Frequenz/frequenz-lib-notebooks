# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for reporting notebook utility functions."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from frequenz.lib.notebooks.reporting.utils.column_mapper import ColumnMapper
from frequenz.lib.notebooks.reporting.utils.reporting_nb_functions import (
    aggregate_metrics,
    assemble_component_analysis,
    build_component_analysis,
    build_overview_df,
    compute_energy_summary,
)


def test_build_component_analysis_selects_all_components_and_melts() -> None:
    """All matching component columns are reshaped into a long-format table."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "Battery #1": [2.0, -1.0],
            "Battery #2": [-3.0, 4.0],
            "PV #1": [5.0, 6.0],
        }
    )

    result = build_component_analysis(
        energy_report_df,
        selection_filter=["All"],
        component_label="Battery",
        value_col_name="battery_power_flow",
    )

    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-09 06:30:00",
                    "2026-01-09 07:00:00",
                    "2026-01-09 06:30:00",
                    "2026-01-09 07:00:00",
                ]
            ),
            "Battery": ["#1", "#1", "#2", "#2"],
            "battery_power_flow": [2.0, -1.0, -3.0, 4.0],
        }
    )

    assert_frame_equal(result, expected)


def test_build_component_analysis_returns_empty_when_columns_missing() -> None:
    """A missing component selection should return a typed empty frame."""
    energy_report_df = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-01-09 06:30:00"]), "Battery #1": [2.0]}
    )

    result = build_component_analysis(
        energy_report_df,
        selection_filter=["#9"],
        component_label="Battery",
        value_col_name="battery_power_flow",
    )

    assert_frame_equal(
        result,
        pd.DataFrame(columns=["timestamp", "Battery", "battery_power_flow"]),
    )


def test_assemble_component_analysis_scales_and_truncates_component_sum() -> None:
    """Scaled values are returned and the optional truncated sum is respected."""
    mapper = ColumnMapper.from_default(locale="en")
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "Battery #1": [2.0, -1.0],
            "Battery #2": [-3.0, 4.0],
        }
    )

    result_df, component_sum, filter_text = assemble_component_analysis(
        component_filter=["Alle"],
        component_key="battery",
        component_types=["battery", "pv"],
        energy_report_df=energy_report_df,
        timestep_hours=0.5,
        mapper=mapper,
        component_label="Battery",
        value_col_name="battery_power_flow",
        invert_sign=True,
        trunc_values=True,
    )

    expected_df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(
                [
                    "2026-01-09 06:30:00",
                    "2026-01-09 07:00:00",
                    "2026-01-09 06:30:00",
                    "2026-01-09 07:00:00",
                ]
            ),
            "Battery": ["#1", "#1", "#2", "#2"],
            "Battery Power Flow": [-1.0, 0.5, 1.5, -2.0],
        }
    )

    assert_frame_equal(result_df, expected_df)
    assert component_sum == 2.0
    assert filter_text == "All"


def test_build_overview_df_keeps_expected_optional_columns() -> None:
    """Overview output should preserve order and ignore unavailable columns."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"]),
            "grid_consumption": [3.0],
            "mid_consumption": [7.0],
            "grid_feed_in": [1.0],
            "pv_asset_production": [5.0],
            "battery_power_flow": [-2.0],
            "extra_column": [99.0],
        }
    )

    result = build_overview_df(energy_report_df, component_types=["pv", "battery"])

    expected = energy_report_df[
        [
            "timestamp",
            "grid_consumption",
            "mid_consumption",
            "grid_feed_in",
            "pv_asset_production",
            "battery_power_flow",
        ]
    ]

    assert_frame_equal(result, expected)


def test_compute_energy_summary_includes_rollups_and_percentages() -> None:
    """Energy summaries should include rollups and stable ordering."""
    df = pd.DataFrame(
        {
            "pv_asset_production": [4.0, 4.0],
            "chp_asset_production": [1.0, 1.0],
            "grid_consumption": [2.0, 0.0],
        }
    )

    result = compute_energy_summary(
        df,
        resolution=timedelta(minutes=30),
        include_rollups=True,
    )

    expected = pd.DataFrame(
        {
            "Energy Source": [
                "PV",
                "CHP",
                "Production (PV+Wind+CHP)",
                "Grid Consumption",
            ],
            "Energy [kWh]": [4.0, 1.0, 5.0, 1.0],
            "Power [kW]": [8.0, 2.0, 10.0, 2.0],
            "Mean [kW]": [4.0, 1.0, 5.0, 1.0],
            "Energy %": [36.364, 9.091, 45.455, 9.091],
        }
    )

    assert_frame_equal(result, expected)


def test_compute_energy_summary_rejects_non_positive_resolution() -> None:
    """A non-positive aggregation step must fail clearly."""
    with pytest.raises(ValueError, match="resolution must be positive"):
        compute_energy_summary(pd.DataFrame({"grid_consumption": [1.0]}), timedelta(0))


def test_aggregate_metrics_computes_energy_peak_date_and_pricing() -> None:
    """Aggregate metrics should clip import, localize peak dates, and price energy."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-14 22:00:00", "2026-07-15 00:00:00"]),
            "pv_asset_production": [1.0, 2.0],
            "chp_asset_production": [0.0, 1.0],
            "production_self_use": [0.5, 1.0],
            "production_excess_in_bat": [0.2, 0.3],
            "grid_feed_in": [0.0, 4.0],
            "grid_consumption": [-1.0, 5.0],
            "mid_consumption": [2.0, 6.0],
            "day_ahead_price": [100.0, 200.0],
        }
    )

    result = aggregate_metrics(
        energy_report_df,
        resolution=timedelta(hours=1),
        price_column="day_ahead_price",
    )

    assert result == {
        "pv_production_sum": 3.0,
        "chp_production_sum": 1.0,
        "wind_production_sum": 0.0,
        "prod_self_consumption_sum": 1.5,
        "prod_bat_sum": 0.5,
        "grid_feed_in_sum": 4.0,
        "grid_consumption_sum": 5.0,
        "mid_consumption_sum": 8.0,
        "total_production_sum": 4.0,
        "prod_self_consumption_share": 0.1875,
        "prod_self_production_share": 0.375,
        "peak": 5.0,
        "peak_date": "15.07.2026",
        "grid_import_cost_sum": 1.0,
        "grid_feed_in_revenue_sum": 0.8,
    }
