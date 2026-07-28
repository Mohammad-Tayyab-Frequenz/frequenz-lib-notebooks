# Tooling Library for Notebooks Release Notes

## Summary

<!-- Here goes a general summary of what this release is about -->

## Upgrading

<!-- Here goes notes on how to upgrade from previous versions, including deprecations and what they should be replaced with -->

- The dedicated `create_battery_usecase_df()` helper has been removed. Use
  `build_overview_df(..., component_types=[..., "battery"])` to build overview
  data with battery plotting helper columns.

## New Features

- Reporting utilities now support appending Assets API display names to single-component labels such as `PV #1179 - PV Roof Meter`.
- Component analysis can now select either meter or inverter IDs from the microgrid config via `component_id_source`.
- Time-series reporting plots now expose date range selector, legend position, and top margin options.

## Bug Fixes

- Energy summaries now treat `grid_consumption` as import-only by clipping negative
  values before aggregation, so grid feed-in no longer cancels out grid import.
- Battery-usecase plots now render day-ahead prices with the configured default
  color and show negative grid consumption as a separate grid feed-in trace.
