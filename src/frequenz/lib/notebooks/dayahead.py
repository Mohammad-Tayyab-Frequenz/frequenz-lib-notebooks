# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Functions for CLI tool to interact with the day-ahead pricing API."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from entsoe import EntsoePandasClient  # type: ignore[attr-defined]


def _normalize_prices(
    prices: pd.Series | pd.DataFrame,
) -> pd.Series:
    """Normalize ENTSO-E day-ahead prices into a single named series.

    Args:
        prices: The raw response from `query_day_ahead_prices()`.

    Returns:
        A series named ``price`` with index name ``timestamp``.

    Raises:
        ValueError: If the response is a dataframe with more than one column.
        TypeError: If the response type is unsupported.
    """
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError(
                "Expected a single price series from ENTSO-E, "
                f"got {prices.shape[1]} columns."
            )
        prices = prices.iloc[:, 0]
    elif not isinstance(prices, pd.Series):
        raise TypeError(
            "Expected ENTSO-E day-ahead prices as a pandas Series or "
            f"single-column DataFrame, got {type(prices).__name__}."
        )

    prices = prices.copy()
    prices.name = "price"
    prices.index.name = "timestamp"
    return prices


def fetch_day_ahead_prices(
    entsoe_key: str | None,
    start: datetime,
    end: datetime,
    country_code: str,
) -> pd.Series:
    """Fetch day-ahead prices for a given country code as a normalized series.

    Args:
        entsoe_key: The API key for the Entsoe API. If not provided, the value
            is read from the ``ENTSOE_API_KEY`` environment variable.
        start: The start date of the query.
        end: The end date of the query.
        country_code: The country code for which to query the prices.

    Returns:
        A pandas series named ``price`` indexed by timestamp.

    Raises:
        ModuleNotFoundError: If the optional entsoe dependency is not installed.
        ValueError: If the time window is invalid or no API key is available.
    """
    if EntsoePandasClient is None:
        raise ModuleNotFoundError(
            "The optional dependency 'entsoe-py' is required to fetch day-ahead "
            "prices. Install it before calling fetch_day_ahead_prices()."
        )

    api_key = entsoe_key or os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing ENTSO-E API key. Pass entsoe_key explicitly or set the "
            "ENTSOE_API_KEY environment variable."
        )

    if start >= end:
        raise ValueError(
            f"Expected start to be earlier than end, got start={start!r}, end={end!r}."
        )

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    if start_ts.tzinfo is None or end_ts.tzinfo is None:
        raise ValueError("Please use timezone-aware datetimes for start and end.")

    client = EntsoePandasClient(api_key=api_key)
    return _normalize_prices(
        client.query_day_ahead_prices(country_code, start=start_ts, end=end_ts)
    )
