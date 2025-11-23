"""
Colormap utilities for oceanographic data visualization.

Uses cmocean colormaps and custom colormaps for oceanographic variables.
"""

from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import cmocean

    CMOCEAN_AVAILABLE = True
except ImportError:
    CMOCEAN_AVAILABLE = False

# Import custom colormaps
from .custom_colormaps import CUSTOM_COLORMAPS


def get_oceanographic_colormap(variable: str = "temperature"):
    """
    Get appropriate colormap for oceanographic variable.

    Parameters
    ----------
    variable : str, default 'temperature'
        Variable name to get colormap for

    Returns
    -------
    matplotlib colormap or str
        Colormap object if cmocean available, colormap name otherwise
    """
    # Use custom colormaps as preferred by user, with fallbacks
    colormap_mapping = {
        # User preferences: TEMP2 for temperature, SAL for salinity,
        # TOPO for bathymetry, OXY for oxygen, PurGre for density
        "temperature": CUSTOM_COLORMAPS["TEMP2"],
        "CT": CUSTOM_COLORMAPS["TEMP2"],
        "conservative_temperature": CUSTOM_COLORMAPS["TEMP2"],
        "salinity": CUSTOM_COLORMAPS["SAL"],
        "SA": CUSTOM_COLORMAPS["SAL"],
        "absolute_salinity": CUSTOM_COLORMAPS["SAL"],
        "bathymetry": CUSTOM_COLORMAPS[
            "TOPO2"
        ],  # Use Flemish Cap bathymetry as default
        "depth": CUSTOM_COLORMAPS["TOPO2"],
        "oxygen": CUSTOM_COLORMAPS["OXY"],
        "density": CUSTOM_COLORMAPS["PurGre"],
        "sigma2": CUSTOM_COLORMAPS["PurGre"],
    }

    # Add cmocean fallbacks for velocity if available
    if CMOCEAN_AVAILABLE:
        colormap_mapping.update(
            {
                "velocity": cmocean.cm.balance,
                "u_velocity": cmocean.cm.balance,
                "v_velocity": cmocean.cm.balance,
                "u_across": cmocean.cm.balance,
                "v_along": cmocean.cm.balance,
                "pressure": CUSTOM_COLORMAPS["TOPO2"],
            }
        )
    else:
        colormap_mapping.update(
            {
                "velocity": plt.cm.RdBu_r,
                "u_velocity": plt.cm.RdBu_r,
                "v_velocity": plt.cm.RdBu_r,
                "u_across": plt.cm.RdBu_r,
                "v_along": plt.cm.RdBu_r,
                "pressure": CUSTOM_COLORMAPS["TOPO2"],
            }
        )

    return colormap_mapping.get(variable, plt.cm.viridis)


def get_saved_colormap_path(colormap_name: str) -> Path:
    """
    Get path to a saved colormap file in the config directory.

    Parameters
    ----------
    colormap_name : str
        Name of the colormap file (with or without .cpt extension)

    Returns
    -------
    Path
        Path to the colormap file

    Examples
    --------
    >>> path = get_saved_colormap_path("topo_negative")
    >>> path = get_saved_colormap_path("topo_negative.cpt")
    """
    if not colormap_name.endswith(".cpt"):
        colormap_name += ".cpt"

    config_dir = Path(__file__).parent.parent / "config" / "saved_colormaps"
    return config_dir / colormap_name


def get_bathymetry_colormap(style: str = "flemish_cap") -> str:
    """
    Get path to bathymetry colormap for PyGMT plotting.

    Parameters
    ----------
    style : str, optional
        Bathymetry colormap style:
        - "flemish_cap": Custom Flemish Cap bathymetry colormap (default)
        - "topo": Standard TOPO colormap
        - "topo2": Alias for "flemish_cap"

    Returns
    -------
    str
        Path to the .cpt colormap file

    Notes
    -----
    The "flemish_cap" colormap is also available as matplotlib colormap
    CUSTOM_COLORMAPS["TOPO2"] for use in matplotlib plots.
    """
    if style in ("flemish_cap", "topo2"):
        return str(get_saved_colormap_path("topo_negative"))
    elif style == "topo":
        return str(get_saved_colormap_path("topo"))
    else:
        raise ValueError(
            f"Unknown bathymetry style: {style}. Available: 'flemish_cap', 'topo2', 'topo'"
        )


def save_colormap_preferences(preferences: Dict) -> None:
    """
    Save user colormap preferences to config.

    Parameters
    ----------
    preferences : dict
        Dictionary of variable: colormap preferences

    Notes
    -----
    Placeholder implementation - will save to YAML config in future.
    """
    # Placeholder - will implement YAML saving later
    pass
