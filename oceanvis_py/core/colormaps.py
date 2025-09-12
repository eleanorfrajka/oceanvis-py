"""
Colormap utilities for oceanographic data visualization.

Uses cmocean colormaps for oceanographic variables.
"""

from typing import Dict
import matplotlib.pyplot as plt

try:
    import cmocean

    CMOCEAN_AVAILABLE = True
except ImportError:
    CMOCEAN_AVAILABLE = False


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
    if CMOCEAN_AVAILABLE:
        # Use cmocean colormaps when available
        # Note: Using RdYlBu_r for temperature per user preference over thermal
        colormap_mapping = {
            "temperature": plt.cm.RdYlBu_r,
            "CT": plt.cm.RdYlBu_r,
            "conservative_temperature": plt.cm.RdYlBu_r,
            "salinity": cmocean.cm.haline,
            "SA": cmocean.cm.haline,
            "absolute_salinity": cmocean.cm.haline,
            "velocity": cmocean.cm.balance,
            "u_velocity": cmocean.cm.balance,
            "v_velocity": cmocean.cm.balance,
            "u_across": cmocean.cm.balance,
            "v_along": cmocean.cm.balance,
            "density": cmocean.cm.dense,
            "sigma2": cmocean.cm.dense,
            "pressure": cmocean.cm.deep,
            "depth": cmocean.cm.deep,
        }
        return colormap_mapping.get(variable, plt.cm.viridis)
    else:
        # Fallback to standard matplotlib colormaps
        colormap_mapping = {
            "temperature": "RdYlBu_r",
            "CT": "RdYlBu_r",
            "conservative_temperature": "RdYlBu_r",
            "salinity": "viridis",
            "SA": "viridis",
            "absolute_salinity": "viridis",
            "velocity": "RdBu_r",
            "u_velocity": "RdBu_r",
            "v_velocity": "RdBu_r",
            "u_across": "RdBu_r",
            "v_along": "RdBu_r",
            "density": "plasma",
            "sigma2": "plasma",
            "pressure": "Blues_r",
            "depth": "Blues_r",
        }
        return colormap_mapping.get(variable, "viridis")


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
