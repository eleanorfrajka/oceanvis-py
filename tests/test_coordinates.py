"""
Tests for oceanvis_py.core.coordinates module.
"""

import numpy as np
import xarray as xr

from oceanvis_py.core.coordinates import (
    calculate_distance,
    distance_from_coordinates,
    cumulative_distance_from_coordinates,
    add_distance_coordinate,
)


def test_calculate_distance():
    """Test great circle distance calculation using GSW."""
    # Distance from equator to North Pole should be ~10,007 km (GSW result)
    dist = calculate_distance(0, 0, 90, 0)
    assert abs(dist - 10007.543398) < 1  # Allow small floating point error

    # Distance between same point should be 0
    dist = calculate_distance(45, -30, 45, -30)
    assert dist == 0


def test_distance_from_coordinates():
    """Test converting lat/lon arrays to distance."""
    # Simple north-south transect
    lats = np.array([45.0, 46.0, 47.0])
    lons = np.array([-30.0, -30.0, -30.0])

    distances = distance_from_coordinates(lats, lons)

    assert len(distances) == 3
    assert distances[0] == 0  # First point is reference
    assert distances[1] > 0  # Second point is north of first
    assert distances[2] > distances[1]  # Third point is furthest

    # Test with custom reference point
    distances_custom = distance_from_coordinates(
        lats, lons, reference_point=(46.0, -30.0)
    )
    assert distances_custom[1] == 0  # Second point is now reference


def test_cumulative_distance_from_coordinates():
    """Test cumulative distance calculation along track."""
    # Simple north-south transect - each degree ~111 km
    lats = np.array([45.0, 46.0, 47.0])
    lons = np.array([-30.0, -30.0, -30.0])

    distances = cumulative_distance_from_coordinates(lats, lons)

    assert len(distances) == 3
    assert distances[0] == 0  # First point
    assert distances[1] > 100  # ~111 km north
    assert distances[2] > distances[1]  # Even further north
    assert distances[2] > 200  # Should be ~222 km total


def test_add_distance_coordinate():
    """Test adding distance coordinate to xarray dataset."""
    # Create test dataset
    data = xr.Dataset(
        {
            "latitude": (["N_PROF"], [45.0, 46.0, 47.0]),
            "longitude": (["N_PROF"], [-30.0, -30.0, -30.0]),
            "temperature": (["N_PROF", "depth"], [[15, 10, 5], [14, 9, 4], [13, 8, 3]]),
        }
    )

    # Test distance from first point
    data_with_dist = add_distance_coordinate(data)
    assert "distance" in data_with_dist.coords
    assert data_with_dist.distance.values[0] == 0
    assert data_with_dist.distance.values[1] > 0

    # Test cumulative distance
    data_cumulative = add_distance_coordinate(data, cumulative=True)
    assert "distance" in data_cumulative.coords
    assert data_cumulative.distance.values[0] == 0
