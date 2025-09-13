"""High-level plotting functions for oceanvis-py."""

from .sections import plot_section, plot_dual_section
from .maps import plot_map
from .widgets import (
    ColorMapWidget,
    InteractiveColormapWidget,
    create_colormap_widget,
    calculate_histogram_levels,
    round_to_human_readable,
)

__all__ = [
    "plot_section",
    "plot_dual_section",
    "plot_map",
    "ColorMapWidget",
    "InteractiveColormapWidget",
    "create_colormap_widget",
    "calculate_histogram_levels",
    "round_to_human_readable",
]
