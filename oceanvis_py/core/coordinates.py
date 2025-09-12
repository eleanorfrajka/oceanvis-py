"""
Coordinate conversion utilities for oceanographic data.

Handles conversion from latitude/longitude to distance coordinates
for section plots, preserving the uneven spacing of profiles.
"""

import numpy as np
import xarray as xr
from geopy.distance import geodesic
from typing import Union, Tuple, Optional, List, Dict
import warnings


def calculate_distance(lat1: float, lon1: float, 
                      lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points.
    
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
        distance = geodesic((lat1, lon1), (lat2, lon2)).kilometers
        return distance
    except Exception as e:
        warnings.warn(f"Error calculating distance: {e}")
        return np.nan


def distance_from_coordinates(latitudes: Union[np.ndarray, List[float]], 
                            longitudes: Union[np.ndarray, List[float]],
                            reference_point: Optional[Tuple[float, float]] = None) -> np.ndarray:
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


def add_distance_coordinate(dataset: xr.Dataset, 
                          lat_var: str = 'latitude',
                          lon_var: str = 'longitude',
                          profile_dim: str = 'N_PROF') -> xr.Dataset:
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
        distances = distance_from_coordinates(lats, lons)
    elif lats.ndim == 2:
        # 2D arrays - calculate distance for each profile
        if profile_dim in dataset[lat_var].dims:
            prof_axis = dataset[lat_var].dims.index(profile_dim)
            if prof_axis == 0:
                # Profiles along first dimension
                distances = distance_from_coordinates(lats[:, 0], lons[:, 0])
            else:
                # Profiles along second dimension  
                distances = distance_from_coordinates(lats[0, :], lons[0, :])
        else:
            warnings.warn("Cannot determine profile axis for distance calculation")
            return dataset
    else:
        warnings.warn("Cannot handle >2D coordinate arrays")
        return dataset
    
    # Add distance coordinate
    if profile_dim in dataset.dims:
        dataset = dataset.assign_coords(
            distance=(profile_dim, distances, {
                'long_name': 'Distance from first profile',
                'units': 'km',
                'description': 'Great circle distance calculated from latitude/longitude'
            })
        )
    else:
        # Add as data variable if dimension not found
        dataset = dataset.assign(
            distance=(['profile'], distances, {
                'long_name': 'Distance from first profile', 
                'units': 'km'
            })
        )
    
    return dataset


def great_circle_bearing(lat1: float, lon1: float,
                        lat2: float, lon2: float) -> float:
    """
    Calculate initial bearing along great circle path.
    
    Uses geographic/navigation convention:
    - 0° = North
    - 90° = East
    - 180° = South
    - 270° = West
    
    Parameters
    ----------
    lat1, lon1 : float
        Start point latitude and longitude (degrees)
    lat2, lon2 : float
        End point latitude and longitude (degrees)
        
    Returns
    -------
    float
        Initial bearing in degrees (0-360), where 0° is North, 90° is East
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    
    # Calculate bearing using spherical trigonometry
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    # This gives mathematical angle (0° = East, 90° = North)
    math_angle = np.arctan2(y, x)
    math_angle = np.degrees(math_angle)
    
    # Convert to geographic bearing (0° = North, 90° = East)
    # Mathematical angle: 0° = East, 90° = North, 180° = West, 270° = South
    # Geographic bearing: 0° = North, 90° = East, 180° = South, 270° = West
    # Conversion: bearing = 90° - math_angle
    bearing = 90.0 - math_angle
    
    # Ensure bearing is in [0, 360) range
    bearing = bearing % 360.0
    
    return bearing


def section_orientation(dataset: xr.Dataset, 
                       lat_var: str = 'latitude',
                       lon_var: str = 'longitude') -> Dict[str, float]:
    """
    Calculate overall orientation of a section from lat/lon coordinates.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset with latitude/longitude coordinates
    lat_var : str, default 'latitude'
        Name of latitude variable
    lon_var : str, default 'longitude'
        Name of longitude variable
        
    Returns
    -------
    dict
        Dictionary with 'bearing' (degrees) and 'length' (km) of section
    """
    if lat_var not in dataset.variables or lon_var not in dataset.variables:
        raise ValueError(f"Dataset must contain {lat_var} and {lon_var} variables")
        
    lats = dataset[lat_var].values
    lons = dataset[lon_var].values
    
    # Use first and last points for overall bearing
    start_lat, start_lon = lats.flat[0], lons.flat[0]
    end_lat, end_lon = lats.flat[-1], lons.flat[-1]
    
    bearing = great_circle_bearing(start_lat, start_lon, end_lat, end_lon)
    total_length = calculate_distance(start_lat, start_lon, end_lat, end_lon)
    
    return {
        'bearing': bearing,
        'length': total_length,
        'start_point': (start_lat, start_lon),
        'end_point': (end_lat, end_lon)
    }