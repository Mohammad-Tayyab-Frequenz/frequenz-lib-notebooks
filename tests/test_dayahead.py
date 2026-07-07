# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Tests for day-ahead helpers."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from frequenz.lib.notebooks.dayahead import (
    fetch_day_ahead_prices,
)


def test_fetch_day_ahead_prices_returns_normalized_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that day-ahead prices are returned as a normalized series."""
    prices = pd.Series(
        [42.0, 43.5],
        index=pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"]),
    )
    client = MagicMock()
    client.query_day_ahead_prices.return_value = prices
    client_factory = MagicMock(return_value=client)
    monkeypatch.setattr(
        "frequenz.lib.notebooks.dayahead.EntsoePandasClient", client_factory
    )

    result = fetch_day_ahead_prices(
        entsoe_key="test-key",
        start=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        end=datetime(2025, 1, 2, 0, 0, 0, tzinfo=UTC),
        country_code="DE_LU",
    )

    assert result.name == "price"
    assert result.index.name == "timestamp"
    assert result.tolist() == [42.0, 43.5]


def test_fetch_day_ahead_prices_uses_environment_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the ENTSO-E API key can be read from the environment."""
    prices = pd.Series(
        [42.0, 43.5],
        index=pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"]),
    )
    client = MagicMock()
    client.query_day_ahead_prices.return_value = prices
    client_factory = MagicMock(return_value=client)
    monkeypatch.setattr(
        "frequenz.lib.notebooks.dayahead.EntsoePandasClient", client_factory
    )
    monkeypatch.setenv("ENTSOE_API_KEY", "env-test-key")

    fetch_day_ahead_prices(
        entsoe_key=None,
        start=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        end=datetime(2025, 1, 2, 0, 0, 0, tzinfo=UTC),
        country_code="DE_LU",
    )

    client_factory.assert_called_once_with(api_key="env-test-key")


def test_fetch_day_ahead_prices_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a clear error is raised when no API key is available."""
    client_factory = MagicMock()
    monkeypatch.setattr(
        "frequenz.lib.notebooks.dayahead.EntsoePandasClient", client_factory
    )
    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ENTSOE_API_KEY"):
        fetch_day_ahead_prices(
            entsoe_key=None,
            start=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
            end=datetime(2025, 1, 2, 0, 0, 0, tzinfo=UTC),
            country_code="DE_LU",
        )

    client_factory.assert_not_called()
