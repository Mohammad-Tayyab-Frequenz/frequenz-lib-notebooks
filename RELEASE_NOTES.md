# Tooling Library for Notebooks Release Notes

## Summary

- The solar maintenance workflow now renders its plots with Plotly instead of Matplotlib, providing interactive plots.
- The combined battery SOC and usecase reporting plot now uses Plotly Resampler to keep large time-series plots responsive while still loading detailed data when users zoom in.

## Upgrading

- Solar maintenance plots are now Plotly figures. Code that directly accessed Matplotlib figure, axes, legend, colormap, or PNG-specific APIs from the solar maintenance plotting internals may need to be updated to use the Plotly-based wrappers or Plotly figure APIs.
- Plotly Resampler is now a runtime dependency. `plot_time_series_battery_soc_and_usecase()` returns a resampler-backed Plotly figure by default; pass `enable_resampler=False` to keep the previous plain `go.Figure` behavior.

## New Features

- Added interactive Plotly plots to the solar maintenance workflow, including per-subplot legends, unified hover boxes, compact axis tick labels, and full date values in hover labels.
- Added dynamic resampling to `plot_time_series_battery_soc_and_usecase()` so the initial figure payload is downsampled and zoom interactions resample from the high-frequency data without adding aggregation-size suffixes to legend labels.


## Bug Fixes
