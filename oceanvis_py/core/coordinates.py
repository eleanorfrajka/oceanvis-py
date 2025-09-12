"""
Coordinate conversion utilities for oceanographic data.

Handles conversion from latitude/longitude to distance coordinates
for section plots, preserving the uneven spacing of profiles.
"""

import numpy as np
import xarray as xr
import gsw
from typing import Union, Tuple, Optional, List
import warnings


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points using GSW.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of first point (degrees)
    lat2, lon2 : float
        Latitude and longitude of second point (degrees)

    Returns
    -------
    float
        Distance in kilometers
    """
    try:
        # GSW distance function returns distance in meters
        distance_m = gsw.distance([lon1, lon2], [lat1, lat2])
        # Convert to kilometers and return the single distance value
        return distance_m[0] / 1000.0
    except Exception as e:
        warnings.warn(f"Error calculating distance: {e}", stacklevel=2)
        return np.nan


def distance_from_coordinates(
    latitudes: Union[np.ndarray, List[float]],
    longitudes: Union[np.ndarray, List[float]],
    reference_point: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Convert latitude/longitude coordinates to distances from a reference point.

    Creates an uneven distance grid based on actual profile locations.

    Parameters
    ----------
    latitudes : array-like
        Array of latitude values (degrees)
    longitudes : array-like
        Array of longitude values (degrees)
    reference_point : tuple of float, optional
        (lat, lon) of reference point. If None, uses first point.

    Returns
    -------
    numpy.ndarray
        Array of distances in kilometers (uneven spacing preserved)
    """
    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)

    if len(latitudes) != len(longitudes):
        raise ValueError("Latitude and longitude arrays must have same length")

    if reference_point is None:
        # Use first point as reference
        ref_lat, ref_lon = latitudes[0], longitudes[0]
    else:
        ref_lat, ref_lon = reference_point

    distances = np.zeros_like(latitudes, dtype=float)

    for i, (lat, lon) in enumerate(zip(latitudes, longitudes)):
        distances[i] = calculate_distance(ref_lat, ref_lon, lat, lon)

    return distances


def cumulative_distance_from_coordinates(
    latitudes: Union[np.ndarray, List[float]],
    longitudes: Union[np.ndarray, List[float]],
) -> np.ndarray:
    """
    Calculate cumulative distance along a track from lat/lon coordinates.

    Useful for towyo data where you want distance along the ship track.

    Parameters
    ----------
    latitudes : array-like
        Array of latitude values (degrees)
    longitudes : array-like
        Array of longitude values (degrees)

    Returns
    -------
    numpy.ndarray
        Array of cumulative distances in kilometers
    """
    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)

    if len(latitudes) != len(longitudes):
        raise ValueError("Latitude and longitude arrays must have same length")

    distances = np.zeros_like(latitudes, dtype=float)

    for i in range(1, len(latitudes)):
        segment_dist = calculate_distance(
            latitudes[i - 1], longitudes[i - 1], latitudes[i], longitudes[i]
        )
        distances[i] = distances[i - 1] + segment_dist

    return distances


def add_distance_coordinate(
    dataset: xr.Dataset,
    lat_var: str = "latitude",
    lon_var: str = "longitude",
    profile_dim: str = "N_PROF",
    cumulative: bool = False,
) -> xr.Dataset:
    """
    Add distance coordinate to dataset based on lat/lon.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset with latitude/longitude coordinates
    lat_var : str, default 'latitude'
        Name of latitude variable
    lon_var : str, default 'longitude'
        Name of longitude variable
    profile_dim : str, default 'N_PROF'
        Name of profile dimension
    cumulative : bool, default False
        If True, calculate cumulative distance along track.
        If False, calculate distance from first point.

    Returns
    -------
    xarray.Dataset
        Dataset with added 'distance' coordinate
    """
    if lat_var not in dataset.variables:
        raise ValueError(f"Latitude variable '{lat_var}' not found in dataset")
    if lon_var not in dataset.variables:
        raise ValueError(f"Longitude variable '{lon_var}' not found in dataset")

    lats = dataset[lat_var].values
    lons = dataset[lon_var].values

    # Handle different array shapes
    if lats.ndim == 1:
        # 1D coordinate arrays
        if cumulative:
            distances = cumulative_distance_from_coordinates(lats, lons)
        else:
            distances = distance_from_coordinates(lats, lons)
    elif lats.ndim == 2:
        # 2D arrays - calculate distance for each profile
        if profile_dim in dataset[lat_var].dims:
            prof_axis = dataset[lat_var].dims.index(profile_dim)
            if prof_axis == 0:
                # Profiles along first dimension
                profile_lats, profile_lons = lats[:, 0], lons[:, 0]
            else:
                # Profiles along second dimension
                profile_lats, profile_lons = lats[0, :], lons[0, :]

            if cumulative:
                distances = cumulative_distance_from_coordinates(
                    profile_lats, profile_lons
                )
            else:
                distances = distance_from_coordinates(profile_lats, profile_lons)
        else:
            warnings.warn("Cannot determine profile axis for distance calculation")
            return dataset
    else:
        warnings.warn("Cannot handle >2D coordinate arrays")
        return dataset

    # Add distance coordinate
    distance_type = "cumulative" if cumulative else "from first profile"
    if profile_dim in dataset.dims:
        dataset = dataset.assign_coords(
            distance=(
                profile_dim,
                distances,
                {
                    "long_name": f"Distance {distance_type}",
                    "units": "km",
                    "description": f"Great circle distance {distance_type} calculated from latitude/longitude",
                },
            )
        )
    else:
        # Add as data variable if dimension not found
        dataset = dataset.assign(
            distance=(
                ["profile"],
                distances,
                {"long_name": f"Distance {distance_type}", "units": "km"},
            )
        )

    return dataset
