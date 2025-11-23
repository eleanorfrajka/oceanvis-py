"""Bathymetry and track mapping functions using PyGMT.

This module provides functions for creating publication-quality bathymetry maps
with optional ship track overlays and contour lines.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union, Tuple, List

def _check_pygmt_availability():
    """Check if PyGMT is available without importing it."""
    try:
        import importlib
        _pygmt_spec = importlib.util.find_spec("pygmt")
        return _pygmt_spec is not None
    except Exception:
        return False


# Check for PyGMT availability at module level for backwards compatibility
HAVE_PYGMT = _check_pygmt_availability()


def _require_pygmt():
    """Import and return pygmt or raise a clear error if unavailable."""
    try:
        import pygmt
        return pygmt
    except Exception:
        raise RuntimeError(
            "PyGMT/GMT not available. Install 'gmt' and 'pygmt' (e.g., via conda-forge) "
            "to use PyGMT mapping functions."
        )


def plot_bathymetry_map(
    bathymetry_file: Union[str, Path],
    region: Tuple[float, float, float, float],
    projection: str = "M5i",
    cpt_file: Optional[str] = None,
    ship_track_file: Optional[str] = None,
    output_file: str = "bathymetry_map.png",
    dpi: int = 300,
    contour_file: Optional[str] = None,
    contour_interval: Optional[int] = 1000,
    title: Optional[str] = None,
    return_figure: bool = False,
    invert_bathymetry: bool = True,
    **kwargs
) -> Union[str, tuple]:
    """Create a bathymetry map with optional ship track and contours.

    This function replicates the functionality of a GMT script for creating
    publication-quality bathymetry maps using PyGMT.

    Parameters
    ----------
    bathymetry_file : str or Path
        Path to the bathymetry grid file (NetCDF format)
    region : tuple of float
        Map region as (west, east, south, north) in degrees
    projection : str, optional
        GMT projection string (default: "M5i" for 5-inch wide Mercator)
    cpt_file : str, optional
        Path to colormap file. If None, uses a default bathymetry colormap
    ship_track_file : str, optional
        Path to ship track file (lon, lat format)
    output_file : str, optional
        Output filename (default: "bathymetry_map.png")
    dpi : int, optional
        Output resolution (default: 300)
    contour_file : str, optional
        Separate grid file for contours (if different from bathymetry_file)
    contour_interval : int, optional
        Contour interval in meters (default: 1000m)
    title : str, optional
        Map title
    invert_bathymetry : bool, optional
        Whether to multiply bathymetry data by -1 for depth sign correction.
        True (default) converts positive depths to negative for standard topo colormap.
        False keeps original depth sign.
    **kwargs
        Additional arguments passed to grdimage

    Returns
    -------
    str
        Path to the created map file

    Examples
    --------
    Basic bathymetry map:
    >>> plot_bathymetry_map(
    ...     bathymetry_file="bathy.nc",
    ...     region=(-56, -36, 40, 52),
    ...     cpt_file="topo_negative.cpt"
    ... )

    With ship track:
    >>> plot_bathymetry_map(
    ...     bathymetry_file="bathy.nc",
    ...     region=(-56, -36, 40, 52),
    ...     ship_track_file="ship_track.xy",
    ...     title="M212 Cruise Track"
    ... )

    """
    pygmt = _require_pygmt()

    # Convert paths to strings
    bathymetry_file = str(bathymetry_file)
    if ship_track_file:
        ship_track_file = str(ship_track_file)
    if contour_file:
        contour_file = str(contour_file)

    # Set GMT parameters
    pygmt.config(MAP_ANNOT_MIN_SPACING="0.1p")

    # Create the figure
    fig = pygmt.Figure()

    # Process bathymetry data with optional depth sign correction
    # This replicates: gmt grdmath input.nc?waterdepth -1 MUL = bathy_negative.nc
    temp_bathy = "temp_bathy_processed.nc"

    # Load bathymetry data and optionally invert depths using xarray
    import xarray as xr
    try:
        ds = xr.open_dataset(bathymetry_file)
        if 'waterdepth' in ds:
            bathy_data = ds.waterdepth * (-1 if invert_bathymetry else 1)
        else:
            # Try other common depth variable names
            depth_vars = ['depth', 'elevation', 'z', 'topo']
            bathy_var = None
            for var in depth_vars:
                if var in ds:
                    bathy_var = var
                    break
            if bathy_var:
                bathy_data = ds[bathy_var] * (-1 if invert_bathymetry else 1)
            else:
                raise ValueError(f"Could not find depth variable in {bathymetry_file}. Available variables: {list(ds.data_vars.keys())}")

        # Save as temporary NetCDF file for PyGMT
        bathy_data.to_netcdf(temp_bathy)
    except Exception as e:
        raise RuntimeError(f"Error processing bathymetry file {bathymetry_file}: {e}")

    # Set default colormap if not provided
    if cpt_file is None:
        from ..core.colormaps import get_bathymetry_colormap
        cpt_file = get_bathymetry_colormap("flemish_cap")

    # Create bathymetry image
    # Replicates: gmt grdimage data/bathy_negative.nc -R$xmin/$xmax/$ymin/$ymax -JM5i -Cantfiles/topo_negative.cpt -Bxa4f2 -Bya2f1 -BWSne
    grdimage_kwargs = {
        'grid': temp_bathy,
        'region': region,
        'projection': projection,
        'cmap': cpt_file,
        'frame': ["xa4f2", "ya2f1", "WSne"]
    }
    grdimage_kwargs.update(kwargs)
    fig.grdimage(**grdimage_kwargs)

    # Add contours if requested
    if contour_file or contour_interval:
        contour_grid = contour_file if contour_file else temp_bathy
        try:
            # Replicates: gmt grdcontour data/coarse_topo.nc -C1000 -A- -Wfaint,100/100/100
            fig.grdcontour(
                grid=contour_grid,
                levels=contour_interval if contour_interval else 1000,
                annotation="n",
                pen="faint,100/100/100"
            )
        except Exception as e:
            print(f"Warning: Could not add contours: {e}")

    # Add ship track if provided
    if ship_track_file and Path(ship_track_file).exists():
        try:
            fig.plot(
                data=ship_track_file,
                pen="1p,220/20/20"
            )
        except Exception as e:
            print(f"Warning: Could not plot ship track: {e}")

    # Add title if provided
    if title:
        west, east, south, north = region
        fig.text(
            x=west, y=north,
            text=title,
            justify="BL",
            offset="0i/0.1i",
            no_clip=True
        )

    # Add colorbar with robust positioning
    # Use anchor-based positioning that works with any projection
    fig.colorbar(
        cmap=cpt_file,
#        position="jMR+w+o-0.5c/0+v",  # Right side, 4cm wide, 0.5cm thick, 0.5cm offset
#        position="n1.05/0.1+jB+w5+v",
        position="jMR+v+o-1c/0+m",
        frame=["x1000", "y+lm"]
    )

    # Save the figure if not returning for further modification
    if not return_figure:
        fig.savefig(output_file, dpi=dpi)

    # Clean up temporary file
    try:
        Path(temp_bathy).unlink()
    except FileNotFoundError:
        pass

    if return_figure:
        return fig, output_file, temp_bathy
    else:
        return output_file


def create_ship_track_map(
    ship_track_file: Union[str, Path],
    bathymetry_file: Union[str, Path],
    region: Optional[Tuple[float, float, float, float]] = None,
    output_file: str = "ship_track_map.png",
    invert_bathymetry: bool = True,
    **kwargs
) -> str:
    """Create a bathymetry map focused on a ship track.

    Convenience function that automatically determines the region from
    the ship track data and creates a bathymetry map.

    Parameters
    ----------
    ship_track_file : str or Path
        Path to ship track file (lon, lat format)
    bathymetry_file : str or Path
        Path to bathymetry grid file
    region : tuple, optional
        Map region. If None, automatically determined from track data
    output_file : str, optional
        Output filename
    invert_bathymetry : bool, optional
        Whether to multiply bathymetry data by -1 for depth sign correction
    **kwargs
        Additional arguments passed to plot_bathymetry_map

    Returns
    -------
    str
        Path to the created map file

    """
    # Read ship track to determine region if not provided
    if region is None:
        import pandas as pd
        import numpy as np
        track_data = pd.read_csv(ship_track_file, sep=r'\s+',
                                names=['lon', 'lat'])

        # Check that track data is valid
        if len(track_data) == 0:
            raise ValueError("Ship track file is empty")

        # Add buffer around track - use numpy for safer min/max
        lon_values = track_data['lon'].values
        lat_values = track_data['lat'].values

        lon_min, lon_max = float(np.min(lon_values)), float(np.max(lon_values))
        lat_min, lat_max = float(np.min(lat_values)), float(np.max(lat_values))

        lon_buffer = (lon_max - lon_min) * 0.1
        lat_buffer = (lat_max - lat_min) * 0.1

        region = (
            lon_min - lon_buffer,
            lon_max + lon_buffer,
            lat_min - lat_buffer,
            lat_max + lat_buffer
        )

    return plot_bathymetry_map(
        bathymetry_file=bathymetry_file,
        region=region,
        ship_track_file=ship_track_file,
        output_file=output_file,
        invert_bathymetry=invert_bathymetry,
        **kwargs
    )


def plot_multi_track_map(
    track_files: List[Union[str, Path]],
    bathymetry_file: Union[str, Path],
    region: Tuple[float, float, float, float],
    track_colors: Optional[List[str]] = None,
    track_labels: Optional[List[str]] = None,
    output_file: str = "multi_track_map.png",
    invert_bathymetry: bool = True,
    **kwargs
) -> str:
    """Create a bathymetry map with multiple ship tracks.

    Parameters
    ----------
    track_files : list of str/Path
        List of paths to ship track files
    bathymetry_file : str or Path
        Path to bathymetry grid file
    region : tuple
        Map region as (west, east, south, north)
    track_colors : list of str, optional
        Colors for each track (GMT color format)
    track_labels : list of str, optional
        Labels for each track
    output_file : str, optional
        Output filename
    invert_bathymetry : bool, optional
        Whether to multiply bathymetry data by -1 for depth sign correction
    **kwargs
        Additional arguments passed to plot_bathymetry_map

    Returns
    -------
    str
        Path to the created map file

    """
    # Create base bathymetry map without ship track, get figure for further modification
    fig, output_file, temp_bathy = plot_bathymetry_map(
        bathymetry_file=bathymetry_file,
        region=region,
        ship_track_file=None,  # We'll add tracks separately
        output_file=output_file,
        return_figure=True,
        invert_bathymetry=invert_bathymetry,
        **kwargs
    )

    # Set default colors if not provided
    if track_colors is None:
        track_colors = ["220/20/20", "20/220/20", "20/20/220", "220/220/20"]

    # Plot each track
    for i, track_file in enumerate(track_files):
        if Path(track_file).exists():
            color = track_colors[i % len(track_colors)]
            fig.plot(
                data=str(track_file),
                pen=f"1p,{color}",
                region=region,
                projection="M5i"
            )

            # Add label if provided
            if track_labels and i < len(track_labels):
                # Position labels at start of each track
                track_data = pd.read_csv(track_file, sep=r'\s+',
                                       names=['lon', 'lat'])
                fig.text(
                    x=track_data['lon'].iloc[0],
                    y=track_data['lat'].iloc[0],
                    text=track_labels[i],
                    justify="BL",
                    offset="0.1i/0.1i"
                )

    fig.savefig(output_file, dpi=kwargs.get('dpi', 300))

    # Clean up temporary file
    try:
        Path(temp_bathy).unlink()
    except FileNotFoundError:
        pass

    return output_file


# Legacy function for backward compatibility
def plot_map(bathymetry_file: str, **kwargs) -> str:
    """Legacy function - use plot_bathymetry_map instead.

    Parameters
    ----------
    bathymetry_file : str
        Path to bathymetry netCDF file
    **kwargs
        Additional plotting arguments

    Returns
    -------
    str
        Path to the created map file

    """
    # Default region if not provided
    if 'region' not in kwargs:
        kwargs['region'] = (-180, 180, -90, 90)

    return plot_bathymetry_map(bathymetry_file, **kwargs)
