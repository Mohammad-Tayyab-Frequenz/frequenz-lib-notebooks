# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Plotly figure management for solar maintenance plots.

This module provides a small Plotly-based plotting manager used by the solar
maintenance workflow. It owns Plotly figures, creates subplot layouts, applies
the project plot theme, updates legends, displays figures in notebooks, and
saves generated plots as standalone HTML files.

The :class:`PlotlyAxis` wrapper intentionally exposes the small subset of the
Matplotlib ``Axes`` API that the solar plotters already use. This keeps the
plotter code focused on domain plotting while allowing the rendering backend to
be Plotly.
"""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, cast

import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots

_logger = logging.getLogger(__name__)
_display_figure = cast(Callable[[go.Figure], None], display)

TextModificationType = str | None
ReplaceLabelType = dict[str, str] | None
AdditionalItemsType = list[tuple[Any, str]] | None
AnyModificationType = AdditionalItemsType | TextModificationType | ReplaceLabelType
ModificationType = dict[str, AnyModificationType]

FREQUENZ_COLOURS = [
    "#000000",
    "#00FFCE",
    "#600DFF",
    "#F42784",
    "#00AEEF",
    "#FFC200",
    "#1A1A1A",
    "#393939",
    "#585858",
    "#777777",
    "#979797",
    "#B6B6B6",
    "#D5D5D5",
    "#F4F4F4",
]


@dataclass
class PlotlyTickLabel:
    """Small compatibility object for legacy tick-label reads."""

    text: str

    def get_text(self) -> str:
        """Return the stored tick label text.

        Returns:
            The tick label text.
        """
        return self.text


class PlotlyLegend:
    """Compatibility object for legacy legend visibility calls."""

    def set_visible(self, visible: bool) -> None:
        """Ignore Matplotlib-style legend visibility changes.

        Args:
            visible: Requested visibility state. The value is accepted for API
                compatibility and ignored because Plotly legend visibility is
                handled on traces and layout.
        """
        del visible


# pylint: disable=too-many-instance-attributes, too-many-locals, too-many-arguments
@dataclass
class PlotlyAxis:
    """A single subplot reference in a Plotly figure."""

    figure: go.Figure
    row: int = 1
    col: int = 1
    visible: bool = True
    labels: list[str] = field(default_factory=list)
    x_label: str = ""
    y_label: str = ""
    xticklabels: list[str] = field(default_factory=list)

    def add_trace(self, trace: Any) -> None:
        """Add a Plotly trace to this subplot.

        Args:
            trace: A Plotly trace object to attach to this axis' row and column.
        """
        if getattr(trace, "name", None) and getattr(trace, "showlegend", None) is None:
            trace.showlegend = True
        self.figure.add_trace(trace, row=self.row, col=self.col)
        name = getattr(trace, "name", None)
        if name:
            self.labels.append(str(name))

    def plot(self, x: Any, y: Any, *args: Any, **kwargs: Any) -> None:
        """Add line traces using the Matplotlib-style arguments used here.

        Args:
            x: X-axis values shared by all plotted series.
            y: One or more y-axis series. Two-dimensional inputs are split into
                one Plotly trace per series.
            *args: Optional Matplotlib-style format strings, such as ``"o-"``.
            **kwargs: Supported plotting options. The solar plotters currently
                use ``label``, ``color``, ``style``, and ``alpha``.
        """
        style = str(kwargs.get("style", ""))
        if args:
            style += "".join(str(arg) for arg in args)
        name = kwargs.get("label")
        color = kwargs.get("color")
        line_dash = "dot" if "--" in style else None
        marker = (
            "markers" if any(marker in style for marker in ["o", "s", "D"]) else None
        )
        mode = "lines+markers" if marker else "lines"
        opacity = kwargs.get("alpha")
        y_values = getattr(y, "T", [y])
        if getattr(y, "ndim", 1) == 1:
            y_values = [y]
        for idx, series in enumerate(y_values):
            trace_name = name[idx] if isinstance(name, list) else name
            trace_color = color[idx] if isinstance(color, list) else color
            self.add_trace(
                go.Scatter(
                    x=x,
                    y=series,
                    mode=mode,
                    name=trace_name,
                    line={"color": trace_color, "dash": line_dash, "width": 1.3},
                    opacity=opacity,
                )
            )

    def fill_between(
        self,
        x: Any,
        y_lower: Any,
        y_upper: Any,
        *,
        color: str | None = None,
        alpha: float | None = None,
        label: str | None = None,
        **_: Any,
    ) -> None:
        """Add a filled band between two curves.

        Args:
            x: X-axis values for both boundary curves.
            y_lower: Lower boundary values.
            y_upper: Upper boundary values.
            color: Fill and boundary color.
            alpha: Fill opacity.
            label: Legend label for the filled band.
            **_: Additional Matplotlib-style keyword arguments accepted for
                compatibility and ignored.
        """
        self.add_trace(
            go.Scatter(
                x=x,
                y=y_upper,
                mode="lines",
                line={"width": 0, "color": color},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        self.add_trace(
            go.Scatter(
                x=x,
                y=y_lower,
                mode="lines",
                fill="tonexty",
                fillcolor=color,
                line={"width": 0, "color": color},
                name=label,
                opacity=alpha,
            )
        )

    def annotate(self, text: str, **_: Any) -> None:
        """Add an annotation near the top-right of this subplot.

        Args:
            text: Annotation text.
            **_: Additional Matplotlib-style annotation arguments accepted for
                compatibility and ignored.
        """
        self.figure.add_annotation(
            text=text,
            xref="paper",
            yref="paper",
            x=0.95,
            y=0.95,
            showarrow=False,
            bgcolor="lightgray",
            row=self.row,
            col=self.col,
        )

    def set_visible(self, visible: bool) -> None:
        """Set whether this subplot is considered visible.

        Args:
            visible: Whether downstream layout and legend handling should treat
                this subplot as visible.
        """
        self.visible = visible

    def get_visible(self) -> bool:
        """Return whether this subplot should be considered visible.

        Returns:
            ``True`` if this subplot is marked visible, otherwise ``False``.
        """
        return self.visible

    def set_title(self, title: str) -> None:
        """Set the subplot title.

        Args:
            title: Text to place above this subplot.
        """
        self.figure.update_xaxes(title_text=self.x_label, row=self.row, col=self.col)
        self.figure.add_annotation(
            text=title,
            xref=f"x{self.axis_suffix()} domain",
            yref=f"y{self.axis_suffix()} domain",
            x=0.5,
            y=1.08,
            showarrow=False,
            font={"size": 16},
        )

    def set_xlabel(self, label: str) -> None:
        """Set the x-axis label.

        Args:
            label: X-axis label text.
        """
        self.x_label = label
        self.figure.update_xaxes(title_text=label, row=self.row, col=self.col)

    def get_xlabel(self) -> str:
        """Return the x-axis label.

        Returns:
            The x-axis label text.
        """
        return self.x_label

    def set_ylabel(self, label: str) -> None:
        """Set the y-axis label.

        Args:
            label: Y-axis label text.
        """
        self.y_label = label
        self.figure.update_yaxes(title_text=label, row=self.row, col=self.col)

    def get_ylabel(self) -> str:
        """Return the y-axis label.

        Returns:
            The y-axis label text.
        """
        return self.y_label

    def set_xlim(self, value: tuple[Any, Any]) -> None:
        """Set the x-axis range.

        Args:
            value: Lower and upper range bounds.
        """
        self.figure.update_xaxes(range=list(value), row=self.row, col=self.col)

    def set_ylim(self, value: tuple[Any, Any]) -> None:
        """Set the y-axis range.

        Args:
            value: Lower and upper range bounds.
        """
        self.figure.update_yaxes(range=list(value), row=self.row, col=self.col)

    def set_xticks(self, ticks: list[int]) -> None:
        """Set x-axis tick positions.

        Args:
            ticks: Tick values to display. For category axes these values must
                match the trace x values, not only the compact tick labels.
        """
        self.figure.update_xaxes(
            tickmode="array",
            tickvals=ticks,
            type="category",
            unifiedhovertitle={"text": "%{x}"},
            row=self.row,
            col=self.col,
        )

    def set_xticklabels(self, labels: list[str]) -> None:
        """Set compact x-axis tick labels.

        Args:
            labels: Display labels for the configured tick values.
        """
        self.xticklabels = labels
        self.figure.update_xaxes(
            ticktext=labels,
            type="category",
            unifiedhovertitle={"text": "%{x}"},
            row=self.row,
            col=self.col,
        )

    def get_xticklabels(self) -> list[PlotlyTickLabel]:
        """Return current x tick labels as compatibility objects.

        Returns:
            Tick labels wrapped in objects exposing ``get_text()``.
        """
        return [PlotlyTickLabel(text) for text in self.xticklabels]

    def get_legend_handles_labels(self) -> tuple[list[Any], list[str]]:
        """Return compatibility legend handles and labels.

        Returns:
            A tuple of placeholder handles and collected trace labels.
        """
        return [None] * len(self.labels), self.labels

    def legend(self, *_: Any, **__: Any) -> PlotlyLegend:
        """Return a compatibility legend object.

        Args:
            *_: Positional Matplotlib-style legend arguments accepted for
                compatibility and ignored.
            **__: Keyword Matplotlib-style legend arguments accepted for
                compatibility and ignored.

        Returns:
            A ``PlotlyLegend`` object that accepts Matplotlib-style visibility
            calls.
        """
        return PlotlyLegend()

    def axis_suffix(self) -> str:
        """Return the Plotly axis suffix for this subplot.

        Returns:
            An empty string for the first subplot and the numeric Plotly axis
            suffix for subsequent subplots.
        """
        index = (self.row - 1) + self.col
        return "" if index == 1 else str(index)


class PlotManager:
    """Manage Plotly figures and subplot references."""

    def __init__(self, theme: str = "frequenz-neustrom"):
        """Initialize a PlotManager instance and apply a plot theme.

        Args:
            theme: Name of the plot theme to use for newly created figures.
        """
        self.figures: dict[str, go.Figure] = {}
        self.axes: dict[str, list[PlotlyAxis]] = {}
        self.current_style_params: dict[str, Any] = {}
        self.apply_plot_theme(theme)
        _logger.info("PlotManager initialised with theme: %s", theme)

    def apply_plot_theme(self, theme: str) -> None:
        """Apply a predefined Plotly theme.

        Args:
            theme: Name of the theme to apply.

        Raises:
            ValueError: If the theme name is not supported.
        """
        if theme == "frequenz-neustrom":
            self.current_style_params = {
                "figure.figsize": (1000, 460),
                "image.cmap": FREQUENZ_COLOURS,
                "lines.color": FREQUENZ_COLOURS[0],
            }
        elif theme in {"elegant-minimalist", "vibrant"}:
            self.current_style_params = {
                "figure.figsize": (1000, 460),
                "image.cmap": FREQUENZ_COLOURS,
                "lines.color": FREQUENZ_COLOURS[0],
            }
        else:
            raise ValueError(f"Style '{theme}' is not recognized.")

    def create_figure(
        self,
        fig_id: str,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[int, int] = (1000, 460),
    ) -> tuple[go.Figure, list[PlotlyAxis]]:
        """Create a new Plotly figure with subplots.

        Args:
            fig_id: Unique identifier for the managed figure.
            nrows: Number of subplot rows.
            ncols: Number of subplot columns.
            figsize: Figure width and height in pixels.

        Returns:
            The created Plotly figure and its subplot wrappers.

        Raises:
            ValueError: If the figure ID already exists or subplot dimensions are
                invalid.
        """
        if fig_id in self.figures:
            raise ValueError(f"Figure with id '{fig_id}' already exists.")
        if nrows < 1 or ncols < 1:
            raise ValueError("Number of rows and columns must be at least 1.")
        fig = make_subplots(rows=nrows, cols=ncols, vertical_spacing=0.12)
        fig.update_layout(
            width=figsize[0],
            height=figsize[1],
            template="plotly_white",
            hovermode="x unified",
            spikedistance=-1,
            legend={"orientation": "h", "yanchor": "top", "y": -0.08},
        )
        axes = [
            PlotlyAxis(fig, row=row, col=col)
            for row in range(1, nrows + 1)
            for col in range(1, ncols + 1)
        ]
        self.figures[fig_id] = fig
        self.axes[fig_id] = axes
        return fig, axes

    def create_multiple_figures(self, fig_params: list[dict[str, Any]]) -> None:
        """Create multiple Plotly figures.

        Args:
            fig_params: List of keyword-argument dictionaries passed to
                :meth:`create_figure`.
        """
        for params in fig_params:
            self.create_figure(**params)

    def create_gridspec_figure(self, **kwargs: Any) -> None:
        """Create a figure while accepting legacy GridSpec parameters.

        Args:
            **kwargs: Figure creation parameters. ``gridspec_kwargs`` is
                accepted for compatibility and ignored because Plotly subplots
                are created without Matplotlib GridSpec.
        """
        kwargs.pop("gridspec_kwargs", None)
        self.create_figure(**kwargs)

    def create_multiple_gridspec_figures(
        self, fig_params: list[dict[str, Any]]
    ) -> None:
        """Create multiple GridSpec-like figures.

        Args:
            fig_params: List of keyword-argument dictionaries passed to
                :meth:`create_gridspec_figure`.
        """
        for params in fig_params:
            self.create_gridspec_figure(**params)

    def adjust_axes_spacing(self, fig_id: str, pixels: float = 100.0) -> None:
        """Adjust subplot spacing through figure margins.

        Args:
            fig_id: Identifier of the managed figure to update.
            pixels: Top and bottom margin size in pixels.

        Raises:
            ValueError: If the figure ID is unknown.
        """
        if fig_id not in self.figures:
            raise ValueError(f"Figure '{fig_id}' does not exist.")
        self.figures[fig_id].update_layout(margin={"t": pixels, "b": pixels})

    def update_legend(
        self,
        fig_id: str,
        axs: list[PlotlyAxis],
        on: str = "axes",
        modifications: ModificationType | None = None,
        **legend_kwargs: Any,
    ) -> None:
        """Update Plotly legend labels and placement.

        Args:
            fig_id: Identifier of the managed figure to update.
            axs: Subplot wrappers whose traces should be included in the legend.
            on: Whether to create one legend for the whole figure or separate
                legends per subplot. Supported values are ``"figure"`` and
                ``"axes"``.
            modifications: Optional legend modifications. Supported keys are
                ``"additional_items"``, ``"remove_label"``, and
                ``"replace_label"``.
            **legend_kwargs: Matplotlib-style legend placement values. The
                current Plotly implementation uses ``loc`` for orientation and
                approximate placement.

        Raises:
            ValueError: If the figure ID is unknown or ``on`` is invalid.
        """
        if fig_id not in self.figures:
            raise ValueError(f"Figure '{fig_id}' does not exist.")
        replace_label = (modifications or {}).get("replace_label")
        remove_label = (modifications or {}).get("remove_label")
        additional_items = (modifications or {}).get("additional_items")
        fig = self.figures[fig_id]
        for trace in fig.data:
            if remove_label and trace.name == remove_label:
                trace.showlegend = False
            if isinstance(replace_label, dict) and trace.name in replace_label:
                trace.name = replace_label[trace.name]
                trace.showlegend = True

        if on == "axes":
            self._update_axes_legends(fig, axs, additional_items, **legend_kwargs)
            return
        if on != "figure":
            raise ValueError(
                "Invalid value for 'on' parameter. Must be 'figure' or 'axes'."
            )

        self._add_legend_items(fig, additional_items)
        fig.update_layout(
            legend={
                "orientation": (
                    "h" if legend_kwargs.get("loc") == "lower center" else "v"
                ),
                "x": 0.5 if legend_kwargs.get("loc") == "lower center" else 1.02,
                "y": -0.08 if legend_kwargs.get("loc") == "lower center" else 0.5,
            }
        )

    def _update_axes_legends(
        self,
        fig: go.Figure,
        axs: list[PlotlyAxis],
        additional_items: AnyModificationType,
        **legend_kwargs: Any,
    ) -> None:
        """Assign traces on each subplot to separate Plotly legends.

        Args:
            fig: Plotly figure containing the subplot traces.
            axs: Subplot wrappers that should each receive their own legend.
            additional_items: Optional marker-only legend entries to add.
            **legend_kwargs: Matplotlib-style legend placement values.
        """
        for idx, ax in enumerate(axs):
            legend_name = "legend" if idx == 0 else f"legend{idx + 1}"
            for trace in fig.data:
                if self._trace_belongs_to_axis(trace, ax):
                    trace.legend = legend_name
                    if trace.name:
                        trace.showlegend = True

            if isinstance(additional_items, list) and idx < len(additional_items):
                self._add_legend_items(
                    fig,
                    [additional_items[idx]],
                    axis=ax,
                    legend_name=legend_name,
                )

            legend_layout_key = "legend" if idx == 0 else f"legend{idx + 1}"
            fig.update_layout(
                {
                    legend_layout_key: {
                        "orientation": (
                            "h" if legend_kwargs.get("loc") == "lower center" else "v"
                        ),
                        "x": 1.02,
                        "xanchor": "left",
                        "y": self._axis_legend_y(fig, ax),
                        "yanchor": "middle",
                    }
                }
            )

    @staticmethod
    def _add_legend_items(
        fig: go.Figure,
        additional_items: AnyModificationType,
        *,
        axis: PlotlyAxis | None = None,
        legend_name: str = "legend",
    ) -> None:
        """Add marker-only legend items to a figure or subplot legend.

        Args:
            fig: Plotly figure to update.
            additional_items: Optional ``(color, label)`` pairs to add.
            axis: Subplot to attach the legend traces to. If omitted, traces are
                added at the figure level.
            legend_name: Plotly legend layout key to associate with the items.
        """
        if not isinstance(additional_items, list):
            return
        for item in additional_items:
            if not item or item[0] is None:
                continue
            color, label = item
            trace = go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"color": color},
                name=label,
                showlegend=True,
                legend=legend_name,
            )
            if axis is None:
                fig.add_trace(trace)
            else:
                fig.add_trace(trace, row=axis.row, col=axis.col)

    @staticmethod
    def _trace_belongs_to_axis(trace: Any, ax: PlotlyAxis) -> bool:
        """Return whether a trace belongs to a subplot reference.

        Args:
            trace: Plotly trace to inspect.
            ax: Subplot wrapper to compare against.

        Returns:
            ``True`` if the trace uses the subplot's x and y axes.
        """
        suffix = ax.axis_suffix()
        xaxis = f"x{suffix}" if suffix else "x"
        yaxis = f"y{suffix}" if suffix else "y"
        return (
            getattr(trace, "xaxis", None) == xaxis
            and getattr(trace, "yaxis", None) == yaxis
        )

    @staticmethod
    def _axis_legend_y(fig: go.Figure, ax: PlotlyAxis) -> float:
        """Return the vertical center of a subplot domain.

        Args:
            fig: Plotly figure containing the subplot.
            ax: Subplot wrapper whose y-domain should be inspected.

        Returns:
            The vertical center of the subplot in figure coordinates.
        """
        suffix = ax.axis_suffix()
        yaxis_name = f"yaxis{suffix}"
        yaxis = getattr(fig.layout, yaxis_name)
        domain = yaxis.domain
        return float((domain[0] + domain[1]) / 2)

    def get_style_attribute(self, attribute: str) -> Any:
        """Retrieve a specific style attribute.

        Args:
            attribute: Name of the style attribute.

        Returns:
            The style value, or ``None`` if the attribute is not configured.
        """
        return self.current_style_params.get(attribute)

    def get_all_style_attributes(self) -> dict[str, Any]:
        """Retrieve all current style attributes.

        Returns:
            The active style parameter mapping.
        """
        return self.current_style_params

    def get_axes(self, fig_id: str, ax_idx: int | None = None) -> list[PlotlyAxis]:
        """Retrieve subplot references for a figure.

        Args:
            fig_id: Identifier of the managed figure.
            ax_idx: Optional zero-based subplot index. If omitted, all subplots
                are returned.

        Returns:
            A list containing the requested subplot wrappers.

        Raises:
            ValueError: If the figure ID is unknown.
            IndexError: If ``ax_idx`` is outside the figure's subplot list.
        """
        if fig_id not in self.axes:
            raise ValueError(f"Figure '{fig_id}' does not exist.")
        axes = self.axes[fig_id]
        if ax_idx is None:
            return axes
        if ax_idx >= len(axes):
            raise IndexError(
                f"Axis index '{ax_idx}' is out of bounds for figure '{fig_id}'."
            )
        return [axes[ax_idx]]

    def get_figure(self, fig_id: str) -> go.Figure:
        """Retrieve a managed Plotly figure.

        Args:
            fig_id: Identifier of the managed figure.

        Returns:
            The requested Plotly figure.

        Raises:
            ValueError: If the figure ID is unknown.
        """
        if fig_id in self.figures:
            return self.figures[fig_id]
        raise ValueError(f"Figure '{fig_id}' does not exist.")

    def show_all(self) -> None:
        """Display all managed figures that contain data.

        Raises:
            RuntimeError: If there are no managed figures to display.
        """
        if not self.figures:
            raise RuntimeError("No figures to display.")
        for fig in self.figures.values():
            if fig.data:
                _display_figure(fig)

    def save_all(self, directory: str) -> None:
        """Save all managed figures as standalone HTML files.

        Args:
            directory: Destination directory for generated ``.html`` files.

        Raises:
            ValueError: If ``directory`` is empty.
        """
        if not directory:
            raise ValueError("Directory not specified.")
        os.makedirs(directory, exist_ok=True)
        for fig_id, fig in self.figures.items():
            fig.write_html(f"{directory}/{fig_id}.html", include_plotlyjs="cdn")

    @contextmanager
    def manage_figure(
        self, fig_id: str, save: bool = False, directory: str | None = None
    ) -> Generator[None, None, None]:
        """Show a managed figure after a plotting block and optionally save it.

        Args:
            fig_id: Identifier of the managed figure to show after the context.
            save: Whether to save all managed figures after showing ``fig_id``.
            directory: Destination directory used when ``save`` is ``True``.

        Yields:
            Control to the caller's plotting block.
        """
        yield
        self.get_figure(fig_id).show()
        if save and directory:
            self.save_all(directory)
