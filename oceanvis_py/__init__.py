"""
oceanvis-py: A Python package for visualizing physical oceanographic data.

This package provides tools for creating publication-quality plots from
netCDF oceanographic data, including CTD profiles, LADCP velocity data,
and bathymetry maps.
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

# Core functionality
from .core import (
    load_netcdf,
    validate_oceanographic_data,
    calculate_distance,
    distance_from_coordinates,
    cumulative_distance_from_coordinates,
    add_distance_coordinate,
    get_oceanographic_colormap,
    save_colormap_preferences,
)

# High-level plotting functions
from .plots import plot_section, plot_map, ColorMapWidget

# Figure size constants for publication
FIGURE_SIZES = {
    "full_width": (13, 8),  # 13 cm wide (space for colorbar)
    "full_width_no_cbar": (15, 8),  # 15 cm wide
    "half_width": (7, 6),  # 7 cm wide
    "third_width": (4.5, 5),  # 4.5 cm wide
}

__all__ = [
    "__version__",
    "FIGURE_SIZES",
    # Core functions
    "load_netcdf",
    "validate_oceanographic_data",
    "calculate_distance",
    "distance_from_coordinates",
    "cumulative_distance_from_coordinates",
    "add_distance_coordinate",
    "get_oceanographic_colormap",
    "save_colormap_preferences",
    # Plotting functions
    "plot_section",
    "plot_map",
    "ColorMapWidget",
]
