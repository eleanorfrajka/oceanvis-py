"""
Colormap utilities for oceanographic data visualization.

Uses cmocean colormaps and custom colormaps for oceanographic variables.
"""

from typing import Dict
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
        "bathymetry": CUSTOM_COLORMAPS["TOPO"],
        "depth": CUSTOM_COLORMAPS["TOPO"],
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
                "pressure": cmocean.cm.deep,
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
                "pressure": plt.cm.Blues_r,
            }
        )

    return colormap_mapping.get(variable, plt.cm.viridis)


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
