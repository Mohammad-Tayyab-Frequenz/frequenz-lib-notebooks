# License: MIT
# Copyright © 2026 Frequenz Energy-as-a-Service GmbH

"""Tests for the microgrid component data helpers."""

import pandas as pd

from frequenz.data.microgrid.component_data import MicrogridData


def test_clamp_consumption_clips_negative_values_to_zero() -> None:
    """Negative consumption is clamped at zero, other columns are untouched."""
    df = pd.DataFrame(
        {"consumption": [-5.0, 0.0, 3.0], "grid": [-5.0, -1.0, 2.0]},
    )

    result = MicrogridData._clamp_consumption(df)  # pylint: disable=protected-access

    assert result["consumption"].tolist() == [0.0, 0.0, 3.0]
    assert result["grid"].tolist() == [-5.0, -1.0, 2.0]


def test_clamp_consumption_without_consumption_column() -> None:
    """A dataframe without a consumption column is returned unchanged."""
    df = pd.DataFrame({"grid": [-5.0, 1.0]})

    result = MicrogridData._clamp_consumption(df)  # pylint: disable=protected-access

    assert result["grid"].tolist() == [-5.0, 1.0]
