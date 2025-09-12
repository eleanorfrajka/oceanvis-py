"""
Tests for oceanvis_py.core.data_loader module.
"""

import xarray as xr

from oceanvis_py.core.data_loader import (
    load_variable_mappings,
    find_variable,
    validate_oceanographic_data,
    get_data_info,
)


def test_load_variable_mappings():
    """Test loading variable mappings from YAML config."""
    mappings = load_variable_mappings()

    assert isinstance(mappings, dict)
    assert "temperature" in mappings
    assert "salinity" in mappings
    assert "pressure" in mappings

    # Check structure
    temp_config = mappings["temperature"]
    assert "primary" in temp_config
    assert "alternative" in temp_config
    assert "units" in temp_config


def test_find_variable():
    """Test finding variables in dataset."""
    # Create test dataset
    data = xr.Dataset(
        {
            "temp": (["x"], [1, 2, 3]),
            "SALT": (["x"], [35, 35.1, 35.2]),
            "pressure": (["x"], [10, 20, 30]),
        }
    )

    mappings = load_variable_mappings()

    # Should find temperature variable
    assert find_variable(data, "temperature", mappings) == "temp"

    # Should find salinity variable
    assert find_variable(data, "salinity", mappings) == "SALT"

    # Should find pressure variable
    assert find_variable(data, "pressure", mappings) == "pressure"

    # Should return None for missing variable
    assert find_variable(data, "nonexistent", mappings) is None


def test_validate_oceanographic_data():
    """Test validation of oceanographic datasets."""
    # Create minimal valid dataset
    data = xr.Dataset(
        {
            "temperature": (["profile", "depth"], [[15, 10, 5], [14, 9, 4]]),
            "salinity": (["profile", "depth"], [[35, 35.1, 35.2], [35, 35.1, 35.2]]),
            "pressure": (["depth"], [10, 50, 100]),
            "latitude": (["profile"], [45.0, 45.1]),
            "longitude": (["profile"], [-30.0, -30.1]),
        }
    )

    result = validate_oceanographic_data(data)

    assert isinstance(result, dict)
    assert "is_valid" in result
    assert "warnings" in result
    assert "found_variables" in result

    # Should find basic variables
    assert "temperature" in result["found_variables"]
    assert "salinity" in result["found_variables"]
    assert "pressure" in result["found_variables"]


def test_get_data_info():
    """Test extracting dataset information."""
    data = xr.Dataset(
        {
            "temperature": (["x"], [1, 2, 3], {"units": "celsius"}),
            "salinity": (["x"], [35, 35.1, 35.2], {"units": "psu"}),
        }
    )

    info = get_data_info(data)

    assert isinstance(info, dict)
    assert "dimensions" in info
    assert "data_variables" in info
    assert "variable_info" in info

    # Check variable info structure
    temp_info = info["variable_info"]["temperature"]
    assert temp_info["units"] == "celsius"
    assert "dimensions" in temp_info
    assert "shape" in temp_info
