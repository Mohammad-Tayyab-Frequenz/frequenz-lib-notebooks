# License: MIT
# Copyright © 2026 Frequenz Energy-as-a-Service GmbH

"""Tests for API credential resolution."""

import pytest

from frequenz.lib.notebooks._credentials import resolve_credentials


def test_resolve_credentials_uses_generic_pair() -> None:
    """Generic credentials are used when no API-specific variables are set."""
    env = {
        "FREQUENZ_API_KEY": "generic-key",
        "FREQUENZ_API_SECRET": "generic-secret",
    }

    assert resolve_credentials(env, "REPORTING_API") == (
        "generic-key",
        "generic-secret",
    )


def test_resolve_credentials_specific_pair_overrides_generic_pair() -> None:
    """API-specific variables override the generic pair as a unit."""
    env = {
        "FREQUENZ_API_KEY": "generic-key",
        "FREQUENZ_API_SECRET": "generic-secret",
        "REPORTING_API_KEY": "reporting-key",
        "REPORTING_API_SECRET": "reporting-secret",
    }

    assert resolve_credentials(env, "REPORTING_API") == (
        "reporting-key",
        "reporting-secret",
    )


def test_resolve_credentials_specific_key_does_not_mix_with_generic_secret() -> None:
    """A partial API-specific pair suppresses the generic credentials."""
    env = {
        "FREQUENZ_API_KEY": "generic-key",
        "FREQUENZ_API_SECRET": "generic-secret",
        "REPORTING_API_KEY": "reporting-key",
    }

    assert resolve_credentials(env, "REPORTING_API") == ("reporting-key", None)


def test_resolve_credentials_rejects_empty_consulted_variable() -> None:
    """Set-but-empty credentials are rejected."""
    env = {
        "FREQUENZ_API_KEY": "generic-key",
        "FREQUENZ_API_SECRET": "",
    }

    with pytest.raises(ValueError, match="FREQUENZ_API_SECRET is set but empty"):
        resolve_credentials(env, "REPORTING_API")
