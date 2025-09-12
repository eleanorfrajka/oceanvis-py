"""
Section plotting functionality for oceanographic data.

Implements section plots with sigma2 contouring and dual subplot functionality.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import Tuple, Optional, List
from ..core.colormaps import get_oceanographic_colormap


def plot_section(
    data: xr.Dataset,
    variable: str = "temperature",
    x_coord: str = "distance",
    y_coord: str = "pressure",
    colormap: Optional[str] = None,
    sigma2_contours: bool = False,
    sigma2_levels: Optional[List[float]] = None,
    figsize: Tuple[float, float] = (10, 6),
    ylims: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create section plot of oceanographic variable vs distance and depth/pressure.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing oceanographic data
    variable : str, default 'temperature'
        Variable to plot
    x_coord : str, default 'distance'
        X-coordinate variable name (usually distance or time)
    y_coord : str, default 'pressure'
        Y-coordinate variable name (pressure or depth)
    colormap : str, optional
        Matplotlib colormap name. If None, uses default for variable
    sigma2_contours : bool, default False
        Whether to overlay sigma2 density contours
    sigma2_levels : list of float, optional
        Contour levels for sigma2. If None, uses automatic levels
    figsize : tuple, default (10, 6)
        Figure size (width, height) in inches
    ylims : tuple, optional
        Y-axis limits (min, max). If None, uses sensible defaults for pressure
    **kwargs
        Additional arguments passed to pcolormesh

    Returns
    -------
    tuple
        (figure, axes) objects
    """
    # Get colormap
    if colormap is None:
        colormap = get_oceanographic_colormap(variable)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set font size for axes labels and ticks
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)

    # Get data arrays
    x_data = data[x_coord]
    y_data = data[y_coord]
    var_data = data[variable]

    # Create section plot
    pcm = ax.pcolormesh(
        x_data, y_data, var_data, cmap=colormap, shading="nearest", **kwargs
    )

    # Customize axes
    ax.set_xlabel(f"{x_coord.title()} (km)")
    ax.set_ylabel(f"{y_coord.title()} (dbar)")
    ax.invert_yaxis()  # Surface at top, depth increases downward

    # Set sensible y-axis limits
    if ylims is not None:
        ax.set_ylim(ylims)
    elif y_coord == "pressure":
        ax.set_ylim(2600, 800)  # Default pressure range, inverted

    # Add colorbar
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label(f"{_get_long_name(data, variable)} ({_get_units(data, variable)})")

    # Add sigma2 contours if requested
    if sigma2_contours and "sigma2" in data:
        _add_sigma2_contours(ax, data, x_coord, y_coord, sigma2_levels)

    plt.tight_layout()
    return fig, ax


def plot_dual_section(
    data: xr.Dataset,
    left_var: str = "temperature",
    right_var: str = "salinity",
    x_coord: str = "distance",
    y_coord: str = "pressure",
    sigma2_contours: bool = True,
    sigma2_levels: Optional[List[float]] = None,
    figsize: Tuple[float, float] = (14, 6),
    ylims: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Create dual section plot (temperature left, salinity right).

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing oceanographic data
    left_var : str, default 'temperature'
        Variable for left subplot
    right_var : str, default 'salinity'
        Variable for right subplot
    x_coord : str, default 'distance'
        X-coordinate variable name
    y_coord : str, default 'pressure'
        Y-coordinate variable name
    sigma2_contours : bool, default True
        Whether to overlay sigma2 density contours
    sigma2_levels : list of float, optional
        Contour levels for sigma2
    figsize : tuple, default (14, 6)
        Figure size (width, height) in inches
    ylims : tuple, optional
        Y-axis limits (min, max). If None, uses sensible defaults for pressure
    **kwargs
        Additional arguments passed to pcolormesh

    Returns
    -------
    tuple
        (figure, (left_axes, right_axes)) objects
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Set font size for both axes
    for ax in [ax1, ax2]:
        ax.tick_params(labelsize=11)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)

    # Get data arrays
    x_data = data[x_coord]
    y_data = data[y_coord]

    # Left subplot (temperature)
    left_cmap = get_oceanographic_colormap(left_var)
    pcm1 = ax1.pcolormesh(
        x_data, y_data, data[left_var], cmap=left_cmap, shading="nearest", **kwargs
    )

    # Right subplot (salinity)
    right_cmap = get_oceanographic_colormap(right_var)
    pcm2 = ax2.pcolormesh(
        x_data, y_data, data[right_var], cmap=right_cmap, shading="nearest", **kwargs
    )

    # Customize axes
    for ax in [ax1, ax2]:
        ax.set_xlabel(f"{x_coord.title()} (km)")
        ax.invert_yaxis()

        # Set sensible y-axis limits
        if ylims is not None:
            ax.set_ylim(ylims)
        elif y_coord == "pressure":
            ax.set_ylim(2600, 800)  # Default pressure range, inverted

    ax1.set_ylabel(f"{y_coord.title()} (dbar)")

    # Add colorbars
    cbar1 = plt.colorbar(pcm1, ax=ax1)
    cbar1.set_label(f"{_get_long_name(data, left_var)} ({_get_units(data, left_var)})")

    cbar2 = plt.colorbar(pcm2, ax=ax2)
    cbar2.set_label(
        f"{_get_long_name(data, right_var)} ({_get_units(data, right_var)})"
    )

    # Add sigma2 contours if requested
    if sigma2_contours and "sigma2" in data:
        for ax in [ax1, ax2]:
            _add_sigma2_contours(ax, data, x_coord, y_coord, sigma2_levels)

    # Adjust spacing to prevent suptitle overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Leave space for suptitle
    return fig, (ax1, ax2)


def _add_sigma2_contours(
    ax: plt.Axes,
    data: xr.Dataset,
    x_coord: str,
    y_coord: str,
    levels: Optional[List[float]] = None,
) -> None:
    """
    Add sigma2 density contours to existing plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add contours to
    data : xarray.Dataset
        Dataset containing sigma2 data
    x_coord : str
        X-coordinate variable name
    y_coord : str
        Y-coordinate variable name
    levels : list of float, optional
        Contour levels. If None, uses automatic levels
    """
    if levels is None:
        # More dense sigma2 contour levels for better resolution
        levels = np.arange(36.0, 37.5, 0.02)  # Every 0.02 kg/m³

    x_data = data[x_coord]
    y_data = data[y_coord]
    sigma2_data = data["sigma2"]

    # Add contour lines
    ax.contour(
        x_data,
        y_data,
        sigma2_data,
        levels=levels,
        colors="black",
        alpha=0.7,
        linewidths=0.6,
    )

    # Add contour labels for every 5th contour to avoid crowding
    label_levels = levels[::5]  # Every 5th level
    cs_labels = ax.contour(
        x_data,
        y_data,
        sigma2_data,
        levels=label_levels,
        colors="black",
        alpha=0,
        linewidths=0,
    )
    ax.clabel(cs_labels, inline=True, fontsize=9, fmt="%.2f")


def _get_units(data: xr.Dataset, var_name: str) -> str:
    """
    Get units string for a variable.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing the variable
    var_name : str
        Variable name

    Returns
    -------
    str
        Units string
    """
    if var_name in data:
        return data[var_name].attrs.get("units", "")
    elif var_name in data.coords:
        return data.coords[var_name].attrs.get("units", "")
    return ""


def _get_long_name(data: xr.Dataset, var_name: str) -> str:
    """
    Get long name for a variable.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing the variable
    var_name : str
        Variable name

    Returns
    -------
    str
        Long name string
    """
    if var_name in data:
        long_name = data[var_name].attrs.get("long_name", var_name)
    else:
        long_name = var_name

    return long_name.replace("_", " ").title()
