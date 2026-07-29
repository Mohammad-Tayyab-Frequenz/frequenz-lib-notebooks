# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for reporting data preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd
from frequenz.gridpool import MicrogridConfig

from frequenz.lib.notebooks.reporting.data_processing import create_energy_report_df
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

    def component_type_ids(
        self, component_type: str, component_category: str | None = None
    ) -> list[str]:
        del component_category
        return self.mapping.get(component_type, [])


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
