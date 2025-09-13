"""
PyGMT backend for oceanvis plotting.

Provides PyGMT-based section plotting with support for CPT files and 
integration with the interactive colormap widgets.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Optional, Union, List

# Optional PyGMT support (lazy import to avoid hard dependency)
HAVE_PYGMT = False
try:
    import importlib
    _pygmt_spec = importlib.util.find_spec("pygmt")
    HAVE_PYGMT = _pygmt_spec is not None
except Exception:
    HAVE_PYGMT = False


def _require_pygmt():
    """Import and return pygmt or raise a clear error if unavailable."""
    if not HAVE_PYGMT:
        raise RuntimeError(
            "PyGMT/GMT not available. Install 'gmt' and 'pygmt' (e.g., via conda-forge) "
            "to use PyGMT plotting."
        )
    import pygmt  # local import to avoid import-time failure
    return pygmt


def make_pygmt_grid(ds, x_var_name, y_var_name, z_data_matrix):
    """
    Create a PyGMT-compatible xarray.DataArray grid from an xarray.Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing coordinate variables.
    x_var_name : str
        Name of the x-coordinate variable (e.g., 'distance').
    y_var_name : str
        Name of the y-coordinate variable (e.g., 'pressure').
    z_data_matrix : str or np.ndarray
        Name of the data variable in ds, or a 2D numpy array of data.

    Returns
    -------
    grid : xarray.DataArray
        DataArray with dims (y, x) and coordinates (y, x) for use in PyGMT.
    """
    x = ds[x_var_name].values
    y = ds[y_var_name].values
    if isinstance(z_data_matrix, str):
        z = ds[z_data_matrix].values
    else:
        z = z_data_matrix
    grid = xr.DataArray(
        z, coords={y_var_name: y, x_var_name: x}, dims=[y_var_name, x_var_name]
    )
    return grid


def create_bathymetry_polygon(ds_towyo, max_depth=3000):
    """
    Create bathymetry polygon for section plotting.
    
    Parameters
    ----------
    ds_towyo : xarray.Dataset
        Dataset containing distance and waterdepth variables
    max_depth : float, default 3000
        Maximum depth for polygon closure
        
    Returns
    -------
    bathymetry_df : pandas.DataFrame
        DataFrame with distance and waterdepth columns for polygon plotting
    """
    distance = np.concatenate([
        [0],
        [0],
        ds_towyo.distance.values,
        [ds_towyo.distance.values[-1]],
        [ds_towyo.distance.values[-1]],
        [0],
    ])
    
    print(f"Max water depth is {np.max(ds_towyo.waterdepth.values)} m")
    waterdepth = np.concatenate([
        [max_depth],
        [ds_towyo.waterdepth.values[0]],
        ds_towyo.waterdepth.values,
        [ds_towyo.waterdepth.values[-1]],
        [max_depth],
        [max_depth],
    ])
    
    bathymetry_df = pd.DataFrame({"distance": distance, "waterdepth": waterdepth})
    return bathymetry_df


def plot_section(
    data: xr.Dataset,
    variable: str = "temperature",
    x_coord: str = "distance",
    y_coord: str = "pressure", 
    cpt_file: Optional[Union[str, Path]] = None,
    boundaries: Optional[List[float]] = None,
    cmap_name: str = "viridis",
    output_file: Optional[str] = None,
    bathymetry: bool = True,
    sigma2_contours: bool = False,
    region: Optional[List[float]] = None,
    projection: str = "X8c/-6c",
    figsize: tuple = (8, 6),
    **kwargs
) -> None:
    """
    Create section plot using PyGMT backend.
    
    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing oceanographic data
    variable : str, default 'temperature'
        Variable to plot
    x_coord : str, default 'distance'
        X-coordinate variable name
    y_coord : str, default 'pressure' 
        Y-coordinate variable name
    cpt_file : str or Path, optional
        Path to PyGMT .cpt file. If provided, overrides boundaries and cmap_name
    boundaries : list of float, optional
        Contour boundaries. If None and no cpt_file, uses automatic levels
    cmap_name : str, default 'viridis'
        Matplotlib colormap name (ignored if cpt_file provided)
    output_file : str, optional
        Output file path. If None, shows plot interactively
    bathymetry : bool, default True
        Whether to show bathymetry polygon
    sigma2_contours : bool, default False
        Whether to overlay sigma2 density contours
    region : list of float, optional
        Plot region [xmin, xmax, ymin, ymax]. If None, uses data bounds
    projection : str, default 'X8c/-6c'
        GMT projection string
    figsize : tuple, default (8, 6)
        Figure size in cm (width, height)
    **kwargs
        Additional arguments for PyGMT functions
        
    Examples
    --------
    >>> # Using a saved CPT file from interactive widget
    >>> plot_section(ds, variable='SA', 
    ...              cpt_file='oceanvis_py/config/saved_colormaps/salinity_SAL_15levels.cpt')
    
    >>> # Using boundaries and colormap
    >>> plot_section(ds, variable='CT', boundaries=[2.7, 2.8, 2.9, 3.0, 3.1, 3.2],
    ...              cmap_name='thermal')
    """
    pygmt = _require_pygmt()
    
    # Handle CPT file vs boundaries
    if cpt_file is not None:
        # Use provided CPT file
        if not Path(cpt_file).exists():
            raise FileNotFoundError(f"CPT file not found: {cpt_file}")
        actual_cpt = cpt_file
        print(f"✅ Using CPT file: {cpt_file}")
    else:
        # Generate CPT file from boundaries and colormap
        if boundaries is None:
            # Calculate automatic boundaries based on data distribution
            from ..plots.widgets import calculate_histogram_levels
            clean_data = data[variable].values[~np.isnan(data[variable].values)]
            boundaries = calculate_histogram_levels(clean_data, n_levels=15, method='percentile')
            print(f"📊 Auto-generated {len(boundaries)} boundaries: {boundaries}")
        
        # Generate temporary CPT file
        from ..plots.plotters import generate_cpt_file_simple
        temp_cpt = f"temp_{variable}_{cmap_name}.cpt"
        generate_cpt_file_simple(boundaries, cmap_name, temp_cpt)
        actual_cpt = temp_cpt
        print(f"🎨 Generated temporary CPT file: {temp_cpt}")
    
    # Create PyGMT grid
    grid = make_pygmt_grid(data, x_coord, y_coord, variable)
    
    # Set up region
    if region is None:
        x_min, x_max = float(data[x_coord].min()), float(data[x_coord].max())
        y_min, y_max = float(data[y_coord].min()), float(data[y_coord].max())
        region = [x_min, x_max, y_max, y_min]  # Note: y_max, y_min for depth
    
    # Create figure
    fig = pygmt.Figure()
    
    # Configure fonts and appearance
    with pygmt.config(
        FONT_ANNOT_PRIMARY="11p",
        FONT_LABEL="12p", 
        FONT_TITLE="12p",
        FONT_ANNOT_SECONDARY="11p",
        MAP_FRAME_PEN="thin,darkgrey",
        MAP_FRAME_AXES="WS",
        MAP_FRAME_TYPE="plain",
    ):
        # Get variable metadata
        var_long_name = data[variable].attrs.get("long_name", variable)
        var_units = data[variable].attrs.get("units", "")
        cruise_label = data.attrs.get("short_cruise_name", "")
        
        # Create basemap
        fig.basemap(
            region=region,
            projection=projection,
            frame=[f"xaf+l{x_coord.title()} (km)", f"yaf+l{y_coord.title()} (dbar)", f"+t{cruise_label}"],
        )
        
        # Plot main data
        fig.grdimage(grid=grid, region=region, cmap=actual_cpt)
        
        # Add sigma2 contours if requested
        if sigma2_contours and "sigma2" in data:
            sigma2_levels = np.arange(36.84, 37.0, 0.02)
            sigma2_grid = make_pygmt_grid(data, x_coord, y_coord, "sigma2")
            fig.grdcontour(
                grid=sigma2_grid.where(~np.isnan(sigma2_grid)),
                region=region,
                levels=0.02,
                annotation="0.04+f11p,Helvetica,black",
                pen="0.5p,black,2_1:1p",
                limit=[36.0, 37.5],
                label_placement="d3c",
            )
        
        # Add bathymetry polygon if requested
        if bathymetry and "waterdepth" in data:
            bathy_df = create_bathymetry_polygon(data, max_depth=region[3])
            fig.plot(
                x=bathy_df["distance"],
                y=bathy_df["waterdepth"] + 200,
                close=True,
                pen="0.7p,black",
                fill="lightgray",
            )
        
        # Add variable name text
        fig.text(
            x=region[0] + 0.02 * (region[1] - region[0]),
            y=region[3] - 0.05 * (region[3] - region[2]),
            text=var_long_name,
            font="12p,Helvetica,black",
            justify="LB",
            offset="0.15c/0.15c",
            no_clip=True,
        )
        
        # Add colorbar
        colorbar_label = f"{var_units}" if var_units else ""
        if boundaries is not None:
            a_int = boundaries[1] - boundaries[0] if len(boundaries) > 1 else "auto"
        else:
            a_int = "auto"
            
        with pygmt.config(
            FONT_ANNOT_PRIMARY="16p", FONT_LABEL="16p", FONT_ANNOT_SECONDARY="16p"
        ):
            fig.colorbar(
                cmap=actual_cpt,
                position="JMR+w5c/.5c+o.4c/.2c",
                frame=[f"a{a_int}", f"y+l{colorbar_label}"],
            )
    
    # Save or show
    if output_file:
        fig.savefig(output_file, dpi=300, transparent=True)
        print(f"💾 Figure saved to: {output_file}")
    else:
        fig.show()
    
    # Clean up temporary CPT file
    if cpt_file is None and Path(temp_cpt).exists():
        Path(temp_cpt).unlink()