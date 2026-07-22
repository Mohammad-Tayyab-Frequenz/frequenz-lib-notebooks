# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for reporting data preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd
import pytest
from frequenz.gridpool import MicrogridConfig
from pandas.testing import assert_frame_equal

from frequenz.lib.notebooks.reporting.data_processing import (
    create_battery_usecase_df,
    create_energy_report_df,
)
from frequenz.lib.notebooks.reporting.utils.column_mapper import ColumnMapper


@dataclass
class _DummyMeta:
    """Minimal config metadata stub with only the microgrid id."""

    microgrid_id: int


class _DummyMicrogridConfig:
    """Minimal config stub exposing component type helpers and metadata."""

    def __init__(self, mapping: dict[str, list[str]], microgrid_id: int = 241) -> None:
        self.mapping = mapping
        self.meta = _DummyMeta(microgrid_id)

    def component_types(self) -> list[str]:
        return list(self.mapping.keys())

    def component_type_ids(self, component_type: str) -> list[str]:
        return self.mapping.get(component_type, [])


def test_create_battery_usecase_df_builds_expected_columns() -> None:
    """Battery usecase helper derives the expected canonical columns."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "mid_consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
        }
    )
    result = create_battery_usecase_df(energy_report_df)

    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "peak_before_optimization": [35.0, 35.0],
            "peak_after_optimization": [30.0, 30.0],
            "battery_charge": [5.0, 0.0],
            "battery_discharge": [0.0, -4.0],
        }
    )

    assert_frame_equal(result, expected)


def test_create_battery_usecase_df_preserves_pv_when_available() -> None:
    """Optional PV production is retained in the standardized output."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "mid_consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "pv_asset_production": [12.0, 10.0],
        }
    )
    result = create_battery_usecase_df(energy_report_df)

    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "pv": [12.0, 10.0],
            "peak_before_optimization": [35.0, 35.0],
            "peak_after_optimization": [30.0, 30.0],
            "battery_charge": [5.0, 0.0],
            "battery_discharge": [0.0, -4.0],
        }
    )

    assert_frame_equal(result, expected)


def test_create_battery_usecase_df_accepts_custom_input_column_names() -> None:
    """Custom source column names are normalized to the canonical output schema."""
    energy_report_df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "load": [35.0, 21.0],
            "grid_load": [30.0, 25.0],
            "battery_flow": [5.0, -4.0],
            "pv_power": [12.0, 10.0],
        }
    )
    result = create_battery_usecase_df(
        energy_report_df,
        timestamp_col="time",
        consumption_col="load",
        grid_consumption_col="grid_load",
        battery_col="battery_flow",
        pv_col="pv_power",
    )

    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00", "2026-01-09 07:00:00"]),
            "consumption": [35.0, 21.0],
            "grid_consumption": [30.0, 25.0],
            "battery_power_flow": [5.0, -4.0],
            "pv": [12.0, 10.0],
            "peak_before_optimization": [35.0, 35.0],
            "peak_after_optimization": [30.0, 30.0],
            "battery_charge": [5.0, 0.0],
            "battery_discharge": [0.0, -4.0],
        }
    )

    assert_frame_equal(result, expected)


def test_create_battery_usecase_df_requires_configured_input_columns() -> None:
    """Missing required columns should fail clearly."""
    energy_report_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"]),
            "consumption": [35.0],
            "grid_consumption": [30.0],
        }
    )

    with pytest.raises(KeyError, match="battery_power_flow"):
        create_battery_usecase_df(energy_report_df)


def test_create_energy_report_df_appends_meter_display_names() -> None:
    """Explicit Assets API display names are appended to component labels."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "grid": [30.0],
            "pv": [-12.0],
            "1179": [-12.0],
        }
    )

    result = create_energy_report_df(
        raw_df,
        component_types=["pv"],
        mcfg=cast(MicrogridConfig, _DummyMicrogridConfig({"pv": ["1179"]})),
        mapper=ColumnMapper.from_default(locale="en"),
        component_display_names={"1179": "PV Roof Meter"},
    )

    assert "PV #1179 - PV Roof Meter" in result.columns


def test_create_energy_report_df_uses_explicit_component_display_names() -> None:
    """Explicit display-name mappings override the Assets API lookup."""
    raw_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-09 06:30:00"], utc=True),
            "grid": [30.0],
            "pv": [-12.0],
            "1179": [-12.0],
        }
    )

    result = create_energy_report_df(
        raw_df,
        component_types=["pv"],
        mcfg=cast(MicrogridConfig, _DummyMicrogridConfig({"pv": ["1179"]})),
        mapper=ColumnMapper.from_default(locale="en"),
        component_display_names={"1179": "Explicit PV Meter"},
    )

    assert "PV #1179 - Explicit PV Meter" in result.columns
