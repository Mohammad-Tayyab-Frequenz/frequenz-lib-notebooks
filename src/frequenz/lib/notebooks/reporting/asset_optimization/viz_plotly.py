# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Visualization for asset optimization reporting using Plotly."""

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .viz_colors import (
    AVAILABLE,
    BUY,
    CHARGE,
    CHP,
    CONSUMPTION,
    DISCHARGE,
    GRID,
    PV,
    SELL,
    SOC,
    STACK_CHARGE,
    STACK_CHP,
    STACK_CONSUMPTION,
    STACK_DISCHARGE,
    STACK_GRID,
    STACK_LINE_WIDTH,
    STACK_PV,
    STACK_WIND,
    ZERO_LINE,
)
from .viz_data import (
    prepare_battery_power_data,
    prepare_energy_trade_data,
    prepare_monthly_data,
    prepare_power_data,
    prepare_power_flow_data,
)

_logger = logging.getLogger(__name__)


FIGURE_SIZE = (1000, 700)
LINE_WIDTH = 1
FONT_SIZE = 14
FONT_FAMILY = "Golos Text, sans-serif"
GRID_LINE_WIDTH = 1
ZERO_LINE_WIDTH = 2
HOVER_BG = "rgba(255,255,255,0.95)"
HOVER_BORDER = "rgba(0,0,0,0.2)"
HOVER_FONT_COLOR = "#111"
LEGEND_COLS = 3


def _legend_grid(num_items: int, *, cols: int = LEGEND_COLS) -> dict[str, int]:
    items = max(num_items, 1)
    cols_eff = min(cols, items)
    rows = (items + cols_eff - 1) // cols_eff
    return {"rows": rows, "cols": cols_eff}


def _legend_config(
    *,
    num_items: int,
    y: float,
    x: float = 0.0,
    cols: int = LEGEND_COLS,
) -> dict[str, object]:
    grid = _legend_grid(num_items, cols=cols)
    config: dict[str, object] = {
        "orientation": "h",
        "yanchor": "top",
        "y": y,
        "xanchor": "left",
        "x": x,
        "bgcolor": "rgba(255,255,255,0)",
    }
    config["entrywidth"] = 1.0 / grid["cols"]
    config["entrywidthmode"] = "fraction"
    return config


def _apply_common_layout(
    fig: go.Figure,
    *,
    y_title: str | None = None,
    legend_items: int | None = None,
) -> None:
    fig.update_layout(
        width=FIGURE_SIZE[0],
        height=FIGURE_SIZE[1],
        template="plotly_white",
        font={"size": FONT_SIZE, "family": FONT_FAMILY},
        hovermode="x unified",
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel={
            "bgcolor": HOVER_BG,
            "bordercolor": HOVER_BORDER,
            "font": {
                "family": FONT_FAMILY,
                "size": FONT_SIZE,
                "color": HOVER_FONT_COLOR,
            },
            "align": "left",
            "namelength": -1,
        },
        separators=",.",
    )
    if legend_items is not None:
        fig.update_layout(legend=_legend_config(num_items=legend_items, y=1.05))
    if y_title:
        fig.update_yaxes(title_text=y_title)
    fig.update_xaxes(
        showgrid=True,
        showline=False,
        mirror=False,
        gridwidth=GRID_LINE_WIDTH,
        ticks="outside",
        ticklen=6,
        tickcolor="rgba(0,0,0,0.35)",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        showline=False,
        mirror=False,
        gridwidth=GRID_LINE_WIDTH,
        ticks="outside",
        ticklen=6,
        tickcolor="rgba(0,0,0,0.35)",
        zeroline=False,
    )
    fig.update_traces(
        line={"width": LINE_WIDTH, "simplify": False}, selector={"type": "scatter"}
    )


# pylint: disable=too-many-arguments
def _apply_axis_padding(
    fig: go.Figure,
    *,
    x_index: pd.Index,
    y_min: float,
    y_max: float,
    row: int | None = None,
    col: int | None = None,
    secondary_y: bool | None = None,
) -> None:
    if len(x_index) > 1:
        span = x_index.max() - x_index.min()
        pad = span * 0.1
        fig.update_xaxes(range=[x_index.min(), x_index.max() + pad], row=row, col=col)

    padded_max = y_max * 1.1 if y_max >= 0 else y_max * 0.9
    fig.update_yaxes(
        range=[y_min, padded_max], row=row, col=col, secondary_y=secondary_y
    )


def _apply_range_slider(
    fig: go.Figure, *, row: int | None = None, col: int | None = None
) -> None:
    fig.update_xaxes(
        rangeslider={"visible": True, "thickness": 0.08},
        row=row,
        col=col,
    )


def _finalize_time_series_plot(
    fig: go.Figure,
    *,
    title: str,
    y_title: str,
    legend_items: int,
    x_index: pd.Index,
    y_series: list[pd.Series | None],
    row: int | None = None,
) -> None:
    if row is None:
        _apply_common_layout(fig, y_title=y_title, legend_items=legend_items)
        fig.update_layout(title={"text": title, "x": 0.0, "xanchor": "left"})
        _apply_range_slider(fig)
    y_all = pd.concat([series for series in y_series if series is not None])
    _apply_axis_padding(
        fig,
        x_index=x_index,
        y_min=float(y_all.min()),
        y_max=float(y_all.max()),
        row=row,
        col=1 if row is not None else None,
    )


def _add_battery_fill_line_traces(
    fig: go.Figure,
    *,
    x_index: pd.Index,
    y: pd.Series,
    name: str,
    color: str,
    opacity: float | None = None,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=x_index,
            y=y,
            mode="none",
            fill="tozeroy",
            fillcolor=color,
            connectgaps=False,
            showlegend=False,
            hoverinfo="skip",
            line={"shape": "linear"},
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=x_index,
            y=y,
            name=name,
            line={"color": color, "shape": "linear"},
            connectgaps=False,
            opacity=opacity,
            hovertemplate="<b>%{fullData.name}</b>: %{y} kW<extra></extra>",
        ),
    )


def _add_zero_boundaries(series: pd.Series) -> pd.Series:
    """Replace values at NaN boundaries with 0.0 to prevent visual bridging across gaps.

    For each transition between valid values and NaNs, the function sets:
    - the last valid value before a NaN sequence to 0.0
    - the first valid value after a NaN sequence to 0.0

    This is useful for plotting (e.g., with fill='tozeroy'), where gaps (NaNs)
    would otherwise be visually connected.
    """
    s = series.copy().astype(float)
    is_null = s.isna()

    # Index positions just before a gap (last valid before NaN run)
    before_gap = (~is_null) & is_null.shift(-1, fill_value=False)
    # Index positions just after a gap (first valid after NaN run)
    after_gap = (~is_null) & is_null.shift(1, fill_value=False)

    s[before_gap] = 0.0
    s[after_gap] = 0.0
    return s


def plot_power_flow(
    df: pd.DataFrame,
    fig: go.Figure | None = None,
    row: int | None = None,
) -> go.Figure:
    """Plot the microgrid power flow as a stacked time-series chart.

    Builds a Plotly figure showing the evolution of key power-flow components
    over time, including on-site production (CHP and/or PV), battery charging and
    discharging (when available), grid exchange, and total consumption. The input
    data is normalized via ``prepare_power_flow_data`` before traces are added.

    If no figure is provided, a new one is created. When ``row`` is not provided,
    the function also applies the common layout and a range slider. Axis padding is
    applied in all cases based on the combined y-range of the plotted series.

    Args:
        df: Input DataFrame containing the columns required by
            ``prepare_power_flow_data``.
        fig: Optional existing figure to add traces to. If not provided, a new
            ``go.Figure`` is created.
        row: Optional subplot row index. When provided, common layout and range
            slider configuration are skipped and axis padding is applied to the
            specified subplot row.

    Returns:
        A Plotly figure containing the power-flow traces.
    """
    data = prepare_power_flow_data(df)

    if fig is None:
        fig = go.Figure()

    legend_items = 0

    if data.has_chp:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.chp,
                name="CHP",
                stackgroup="production",
                line={"color": CHP, "shape": "linear"},
                opacity=0.8,
                hovertemplate="<b>CHP</b>: %{y} kW<extra></extra>",
            )
        )
        legend_items += 1

    if data.has_pv:
        pv_label = "PV (on CHP)" if data.has_chp else "PV"
        pv_series = data.production - data.chp
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=pv_series,
                name=pv_label,
                stackgroup="production",
                line={"color": PV, "shape": "linear"},
                opacity=0.9,
                hovertemplate="<b>%{fullData.name}</b>: %{y} kW<extra></extra>",
            )
        )
        legend_items += 1

    if data.charge is not None and data.discharge is not None:
        # 1. Consumption Reference (The 'Floor' for Charge, 'Ceiling' for Discharge)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.consumption,
                stackgroup="charge_stack",
                fill="none",
                line={"color": "rgba(0,0,0,0)", "width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        charge_delta = (data.charge - data.consumption).clip(lower=0)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=charge_delta,
                name="Charge",
                stackgroup="charge_stack",
                line={"color": CHARGE, "shape": "linear", "width": 1},
                customdata=data.charge,
                hovertemplate="<b>Charge</b>: %{customdata} kW<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.consumption,
                stackgroup="discharge_stack",
                fill="none",
                line={"color": "rgba(0,0,0,0)", "width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        discharge_delta = (data.discharge - data.consumption).clip(upper=0)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=discharge_delta,
                name="Discharge",
                stackgroup="discharge_stack",
                line={"color": DISCHARGE, "shape": "linear", "width": 1},
                customdata=data.discharge,
                hovertemplate="<b>Discharge</b>: %{customdata} kW<extra></extra>",
            )
        )
        legend_items += 2

    if data.grid is not None:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.grid,
                name="Grid",
                line={"color": GRID, "shape": "hv"},
                hovertemplate="<b>Grid</b>: %{y} kW<extra></extra>",
            )
        )
        legend_items += 1

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.consumption,
            name="Consumption",
            line={"color": CONSUMPTION, "shape": "hv"},
            hovertemplate="<b>Consumption</b>: %{y} kW<extra></extra>",
        )
    )
    legend_items += 1

    _finalize_time_series_plot(
        fig,
        title="Power Flow",
        y_title="Power (kW)",
        legend_items=legend_items,
        x_index=data.index,
        y_series=[
            data.consumption,
            data.chp,
            data.production,
            data.grid,
            data.charge,
            data.discharge,
        ],
        row=row,
    )
    return fig


def _split_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end), …] for each contiguous True-run in *mask*."""
    if not mask.any():
        return []
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size - 1]
    return list(zip(starts, ends))


def _step_x(x: np.ndarray, start: int, end: int) -> np.ndarray:
    """Return the doubled x coordinates spanning the hv steps of samples start..end.

    Sample i holds from x[i] to x[i+1], so the run reaches one sample past its last
    index. A run ending on the final sample is extended by the sampling interval,
    which is the only width it can be given. Without this a single-sample run would
    collapse to zero width and disappear.
    """
    if end + 1 < x.size:
        right = x[end + 1]
    else:
        right = x[end] + (x[end] - x[end - 1]) if x.size > 1 else x[end]
    edges = np.concatenate([x[start : end + 1], [right]])
    return np.repeat(edges, 2)[1:-1]


# pylint: disable=too-many-arguments
def _add_step_fill(
    fig: go.Figure,
    *,
    x_index: pd.Index,
    upper: np.ndarray,
    lower: np.ndarray,
    segments: list[tuple[int, int]],
    color: str,
    name: str,
) -> None:
    """Fill between two step curves over each segment, with one legend entry."""
    x = x_index.to_numpy()
    for start, end in segments:
        x_step = _step_x(x, start, end)
        # slice incl. endpoint
        y_up = np.repeat(upper[start : end + 1], 2)
        y_lo = np.repeat(lower[start : end + 1], 2)

        # closed polygon
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_step, x_step[::-1]]),
                y=np.concatenate([y_up, y_lo[::-1]]),
                mode="lines",
                line={"width": 0},
                fill="toself",
                fillcolor=color,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # legend proxy
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 10, "color": color},
            name=name,
            hoverinfo="skip",
        )
    )


def _restore_stack_line_widths(traces: tuple[go.Scatter, ...]) -> None:
    """Re-assert the line widths the stack relies on.

    _apply_common_layout stamps LINE_WIDTH on every scatter, which would outline
    the fills and reveal the invisible baseline.
    """
    for trace in traces:
        if trace.fill in ("tonexty", "toself"):
            trace.line.width = 0
        elif trace.name in ("Grid", "Consumption"):
            trace.line.width = STACK_LINE_WIDTH
    traces[0].line.width = 0


def plot_power(
    df: pd.DataFrame,
    fig: go.Figure | None = None,
    row: int | None = None,
) -> go.Figure:
    """Plot the microgrid power as a cumulative PSC stack.

    Each component is added to the running total in its raw PSC sign, starting
    from the consumption line, so the top of the stack coincides with the grid
    line by energy balance. A stack top that drifts away from the grid line means
    the components do not account for the measured exchange.

    Unlike plot_power_flow, production is not clipped, so PV drawing power is
    visible rather than silently flattened to zero.

    Args:
        df: Input DataFrame containing the columns required by prepare_power_data.
        fig: Optional existing figure to add traces to. If not provided, a new
            figure is created.
        row: Optional subplot row index. When provided, common layout and range
            slider configuration are skipped and axis padding is applied to the
            specified subplot row.

    Returns:
        A Plotly figure containing the stacked power traces.
    """
    data = prepare_power_data(df)

    if fig is None:
        fig = go.Figure()

    legend_items = 0
    first_trace = len(fig.data)

    # Invisible consumption baseline so the first band fills from the cons line.
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.consumption,
            mode="lines",
            line={"width": 0, "shape": "hv"},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    bands = (
        (data.has_chp, data.after_chp, "CHP", STACK_CHP),
        (data.has_pv, data.after_pv, "PV", STACK_PV),
        (data.has_wind, data.after_wind, "Wind", STACK_WIND),
    )
    for present, level, name, color in bands:
        if not present:
            continue
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=level,
                name=name,
                mode="lines",
                line={"width": 0, "shape": "hv"},
                fill="tonexty",
                fillcolor=color,
                hovertemplate=f"<b>{name}</b>: %{{y}} kW<extra></extra>",
            )
        )
        legend_items += 1

    if data.has_battery:
        top = data.after_battery.to_numpy()
        base = data.after_wind.to_numpy()
        _add_step_fill(
            fig,
            x_index=data.index,
            upper=top,
            lower=base,
            segments=_split_segments(top > base),
            color=STACK_CHARGE,
            name="Charge",
        )
        _add_step_fill(
            fig,
            x_index=data.index,
            upper=base,
            lower=top,
            segments=_split_segments(top < base),
            color=STACK_DISCHARGE,
            name="Discharge",
        )
        legend_items += 2

    # Grid line on top, should visually coincide with the stack top.
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.grid,
            name="Grid",
            line={"color": STACK_GRID, "shape": "hv", "width": STACK_LINE_WIDTH},
            hovertemplate="<b>Grid</b>: %{y} kW<extra></extra>",
        )
    )
    legend_items += 1

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.consumption,
            name="Consumption",
            line={"color": STACK_CONSUMPTION, "shape": "hv", "width": STACK_LINE_WIDTH},
            hovertemplate="<b>Consumption</b>: %{y} kW<extra></extra>",
        )
    )
    legend_items += 1

    _finalize_time_series_plot(
        fig,
        title="Power",
        y_title="Power (kW)",
        legend_items=legend_items,
        x_index=data.index,
        y_series=[
            data.consumption,
            data.after_chp,
            data.after_pv,
            data.after_wind,
            data.after_battery,
            data.grid,
        ],
        row=row,
    )

    # Zero
    fig.update_yaxes(
        zeroline=True,
        zerolinecolor=ZERO_LINE,
        zerolinewidth=ZERO_LINE_WIDTH,
        row=row,
        col=1 if row is not None else None,
    )
    _restore_stack_line_widths(fig.data[first_trace:])
    return fig


def plot_energy_trade(
    df: pd.DataFrame,
    fig: go.Figure | None = None,
    row: int | None = None,
) -> go.Figure:
    """Plot the microgrid energy trade as a time-series chart.

    Creates a Plotly figure showing bought and sold energy over time based on
    the processed output of ``prepare_energy_trade_data``. The function adds
    separate traces for energy buy and sell values and optionally applies the
    standard layout and range slider when used as a standalone figure.

    Args:
        df: Input DataFrame containing the columns required by
            ``prepare_energy_trade_data``.
        fig: Optional existing figure to which traces are added. If not provided,
            a new ``go.Figure`` is created.
        row: Optional subplot row index. When provided, the common layout and
            range slider are not applied, and axis padding is scoped to the
            specified subplot row.

    Returns:
        A Plotly figure containing the energy trade traces.
    """
    data = prepare_energy_trade_data(df)

    if fig is None:
        fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.buy,
            name="Buy",
            line={"color": BUY, "shape": "hv"},
            fill="tozeroy",
            hovertemplate="<b>Buy</b>: %{y} kWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.sell,
            name="Sell",
            line={"color": SELL, "shape": "hv"},
            fill="tozeroy",
            hovertemplate="<b>Sell</b>: %{y} kWh<extra></extra>",
        )
    )

    _finalize_time_series_plot(
        fig,
        title="Energy Trade",
        y_title="Energy (kWh)",
        legend_items=2,
        x_index=data.index,
        y_series=[data.buy, data.sell],
        row=row,
    )
    return fig


def plot_power_flow_trade(df: pd.DataFrame) -> go.Figure:
    """Create a combined subplot showing power flow and energy trade.

    Builds a two-row Plotly figure with shared x-axis:
    - The top subplot renders the power-flow visualization.
    - The bottom subplot renders the corresponding energy-trade visualization.

    Traces from the underlying figures are merged into a single subplot layout and
    assigned to two separate legends (one per subplot). The figure applies a common
    layout configuration, axis styling, subplot-specific y-axis titles, and adds a
    range slider on the lower subplot.

    Args:
        df: Input DataFrame containing the columns required by ``plot_power_flow``
            and ``plot_energy_trade``.

    Returns:
        A Plotly figure containing the stacked power-flow and energy-trade
        subplots with independent legends and shared time navigation.
    """
    fig_final = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.2,
        row_heights=[0.75, 0.25],
    )

    # Generate standalone figures
    fig_power = plot_power_flow(df)
    fig_trade = plot_energy_trade(df)

    for trace in fig_power.data:
        trace.legend = "legend"
        fig_final.add_trace(trace, row=1, col=1)

    for trace in fig_trade.data:
        trace.legend = "legend2"
        fig_final.add_trace(trace, row=2, col=1)

    _apply_common_layout(fig_final)
    fig_final.update_layout(margin={"l": 30, "r": 70, "t": 30, "b": 40})

    # Configure the layout for both legends
    power_items = len([t for t in fig_power.data if t.showlegend is not False])
    trade_items = len([t for t in fig_trade.data if t.showlegend is not False])
    fig_final.update_layout(
        legend=_legend_config(num_items=power_items, y=1.1),
        legend2=_legend_config(num_items=trade_items, y=0.30),
    )

    # Set Specific Titles
    fig_final.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig_final.update_yaxes(title_text="Energy (kWh)", row=2, col=1)
    _apply_range_slider(fig_final, row=2, col=1)

    return fig_final


def plot_battery_power(df: pd.DataFrame) -> go.Figure:
    """Plot battery power and state of charge (SOC) over time.

    Creates a Plotly figure visualizing battery behavior, including available
    power, charging, discharging, and SOC. The SOC is displayed on a secondary
    y-axis, while power-related traces share the primary axis.

    Args:
        df: Input DataFrame containing the columns required by
            ``prepare_battery_power_data``.

    Returns:
        A Plotly figure showing battery power flows and SOC.
    """
    data = prepare_battery_power_data(df)

    fig = go.Figure()

    charge_pos = data.charge.clip(lower=0)
    available_pos = data.available.clip(lower=0)

    charge_plot = charge_pos.where(charge_pos > 0)
    charge_plot = charge_plot.mask(charge_plot.isna(), np.nan)

    mask = charge_plot.notna() & available_pos.notna()
    charge_plot = charge_plot.where(
        ~mask, charge_plot.where(charge_plot <= available_pos, available_pos)
    )

    discharge_plot = data.battery.where(data.discharge.notna())

    charge_plot = _add_zero_boundaries(charge_plot)
    discharge_plot = _add_zero_boundaries(discharge_plot)

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.soc,
            name="SOC",
            line={"color": SOC, "shape": "linear"},
            fill="tozeroy",
            opacity=0.4,
            yaxis="y2",
            hovertemplate="<b>%{fullData.name}</b>: %{y}%<extra></extra>",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data.available,
            name="Available power",
            line={"color": AVAILABLE, "shape": "linear"},
            hovertemplate="<b>%{fullData.name}</b>: %{y} kW<extra></extra>",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=[0] * len(data.index),
            name="Zero",
            line={"color": ZERO_LINE, "dash": "dash", "shape": "linear"},
            showlegend=False,
            hoverinfo="skip",
        ),
    )

    _add_battery_fill_line_traces(
        fig,
        x_index=data.index,
        y=charge_plot,
        name="Charge",
        color=CHARGE,
    )
    _add_battery_fill_line_traces(
        fig,
        x_index=data.index,
        y=discharge_plot,
        name="Discharge",
        color=DISCHARGE,
        opacity=0.9,
    )

    fig.update_layout(
        yaxis={
            "title": "Battery Power",
            "range": [-data.max_abs_battery * 1.1, data.max_abs_battery * 1.1],
        },
        yaxis2={
            "title": "Battery SOC",
            "range": [0, 100],
            "overlaying": "y",
            "side": "right",
            "tickfont": {"color": SOC},
        },
    )

    _apply_common_layout(fig, legend_items=4)
    fig.update_layout(legend=_legend_config(num_items=4, y=1.1, cols=4))
    _apply_range_slider(fig)
    return fig


def plot_monthly(df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    """Plot monthly aggregated energy data as grouped bar charts.

    Builds a Plotly bar chart showing positive and negative monthly energy
    values returned by ``prepare_monthly_data``. Positive values are plotted
    as standard bars, while negative values are rendered with reduced opacity
    for visual distinction.

    Args:
        df:
            Input DataFrame containing time-series data required by
            ``prepare_monthly_data``.

    Returns:
        The monthly aggregated DataFrame used for plotting.
    """
    data = prepare_monthly_data(df)

    fig = go.Figure()
    x_labels = pd.to_datetime(data.months.index).strftime("%d-%m-%Y")
    for column in data.positive.columns:
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=data.positive[column],
                name=column,
                text=data.positive[column].round(3),
                textposition="outside",
                offsetgroup=column,
            )
        )
    for column in data.negative.columns:
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=data.negative[column],
                name=column,
                opacity=0.7,
                text=data.negative[column].round(3),
                textposition="outside",
                offsetgroup=column,
            )
        )

    _apply_common_layout(fig, y_title="Energy (MWh)")
    fig.update_layout(
        barmode="group",
        title={
            "text": "Monthly Energy",
            "x": 0.05,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
            "pad": {"t": 8},
        },
    )
    fig.update_xaxes(
        tickangle=45, showdividers=False, dividercolor="rgba(0,0,0,0)", automargin=True
    )
    return fig, data.months
