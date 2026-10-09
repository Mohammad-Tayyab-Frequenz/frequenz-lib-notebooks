# Tooling Library for Notebooks Release Notes

## Summary


## Upgrading

Active-energy access is consolidated in `ac_active_energy()`. Use
`direction="net"` (the default), `"consumed"`, or `"delivered"` instead of the
separate active-energy fetchers.

## New Features
- Update reporting notebook with the latest changes.
- `ac_active_power()` can now derive power from net cumulative AC active energy
with `from_energy=True`. This supports a gradual migration of power-based
reporting; callers that need to avoid interpreting large energy changes as
instantaneous power spikes should consume `ac_active_energy()` data directly.

## Bug Fixes
