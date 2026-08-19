# License: MIT
# Copyright © 2026 Frequenz Energy-as-a-Service GmbH

"""Resolution of API credentials from environment variables."""

import logging
from collections.abc import Mapping

_logger = logging.getLogger(__name__)

GENERIC_KEY_VAR = "FREQUENZ_API_KEY"
"""Variable holding the key accepted by every Frequenz API."""

GENERIC_SECRET_VAR = "FREQUENZ_API_SECRET"
"""Variable holding the signing secret belonging to `GENERIC_KEY_VAR`."""


def resolve_credentials(
    env: Mapping[str, str], prefix: str
) -> tuple[str | None, str | None]:
    """Return the auth key and signing secret for the API named by `prefix`.

    The generic credentials are the expected way to authenticate; the API-specific
    pair overrides them for a single API. It applies as a unit, since a specific
    key signed with the generic secret would authenticate against nothing.

    Args:
        env: Environment variables to resolve against.
        prefix: Prefix of the API-specific variables, e.g. `REPORTING_API`.

    Returns:
        The auth key and the signing secret, either of which may be `None`.

    Raises:
        ValueError: If one of the consulted variables is set but empty.
    """
    key_var = f"{prefix}_KEY"
    secret_var = f"{prefix}_SECRET"

    for var in (key_var, secret_var, GENERIC_KEY_VAR, GENERIC_SECRET_VAR):
        if var in env and not env[var]:
            raise ValueError(f"{var} is set but empty.")

    if key_var in env or secret_var in env:
        if GENERIC_KEY_VAR in env:
            _logger.info("%s overrides %s.", key_var, GENERIC_KEY_VAR)
        return env.get(key_var), env.get(secret_var)

    return env.get(GENERIC_KEY_VAR), env.get(GENERIC_SECRET_VAR)
