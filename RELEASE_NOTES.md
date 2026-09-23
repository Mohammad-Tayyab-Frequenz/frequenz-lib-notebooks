# Tooling Library for Notebooks Release Notes

## Summary


## Upgrading
- When constructing `MicrogridData` directly, pass an `AssetsApiClient` through
  the optional `assets_client` argument to include SOC bounds. Aggregate SOC is
  calculated from configured components' `BATTERY_CAPACITY` values in Reporting.
- A `BATTERY_SOC_PCT` formula is no longer used for SOC reporting and can be
  removed. SOC values are now fetched from configured battery components.


## New Features
- Update reporting notebook with the latest changes.
- Add lower and upper battery SOC rated bounds to energy reports and SOC plots.
- Calculate aggregate battery SOC as a capacity-weighted average of the SOC
  values fetched directly for configured battery components.

## Bug Fixes
