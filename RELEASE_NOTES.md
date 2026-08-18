# Tooling Library for Notebooks Release Notes

## Summary

- The solar maintenance workflow now renders its plots with Plotly instead of Matplotlib, providing interactive plots.

## Upgrading

- Solar maintenance plots are now Plotly figures. Code that directly accessed Matplotlib figure, axes, legend, colormap, or PNG-specific APIs from the solar maintenance plotting internals may need to be updated to use the Plotly-based wrappers or Plotly figure APIs.

## New Features

- Added interactive Plotly plots to the solar maintenance workflow, including per-subplot legends, unified hover boxes, compact axis tick labels, and full date values in hover labels.


## Bug Fixes
