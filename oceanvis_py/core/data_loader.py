"""
Data loading and validation utilities for oceanographic netCDF files.
"""

import xarray as xr
import yaml
import warnings
from pathlib import Path
from typing import Dict, Optional, Union


def load_variable_mappings() -> Dict:
    """Load variable name mappings from YAML configuration."""
    config_path = Path(__file__).parent.parent / "config" / "variable_mappings.yaml"
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Default mappings if config file doesn't exist
        return {
            "temperature": {
                "primary": ["temperature", "temp", "T"],
                "alternative": ["CT", "conservative_temperature"],
                "units": ["celsius", "degC", "degree_C", "K"],
            },
            "salinity": {
                "primary": ["salinity", "sal", "S"],
                "alternative": ["SA", "absolute_salinity"],
                "units": ["psu", "1", "g/kg"],
            },
            "pressure": {
                "primary": ["pressure", "pres", "P"],
                "alternative": ["PRES"],
                "units": ["dbar", "db", "Pa"],
            },
            "depth": {
                "primary": ["depth", "DEPTH"],
                "alternative": ["Z"],
                "units": ["m", "meters"],
            },
        }


def find_variable(
    dataset: xr.Dataset, variable_type: str, mappings: Optional[Dict] = None
) -> Optional[str]:
    """
    Find a variable in the dataset based on type and naming conventions.

    Parameters
    ----------
    dataset : xarray.Dataset
        The netCDF dataset to search
    variable_type : str
        Type of variable to find (e.g., 'temperature', 'salinity')
    mappings : dict, optional
        Variable mapping configuration

    Returns
    -------
    str or None
        Name of found variable, or None if not found
    """
    if mappings is None:
        mappings = load_variable_mappings()

    if variable_type not in mappings:
        return None

    var_config = mappings[variable_type]
    search_names = var_config.get("primary", []) + var_config.get("alternative", [])

    for var_name in search_names:
        if var_name in dataset.variables:
            return var_name

    return None


def load_netcdf(file_path: Union[str, Path], validate: bool = True) -> xr.Dataset:
    """
    Load netCDF file with oceanographic data using xarray.

    Parameters
    ----------
    file_path : str or Path
        Path to the netCDF file
    validate : bool, default True
        Whether to validate the dataset for oceanographic conventions

    Returns
    -------
    xarray.Dataset
        Loaded dataset

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist
    ValueError
        If validation fails and critical variables are missing
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {file_path}")

    try:
        dataset = xr.open_dataset(file_path)
    except Exception as e:
        raise ValueError(f"Error loading netCDF file {file_path}: {e}")

    if validate:
        validation_result = validate_oceanographic_data(dataset)
        if not validation_result["is_valid"]:
            warnings.warn(f"Dataset validation issues: {validation_result['warnings']}")

    return dataset


def validate_oceanographic_data(dataset: xr.Dataset) -> Dict:
    """
    Validate dataset for oceanographic data conventions.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to validate

    Returns
    -------
    dict
        Validation results with 'is_valid', 'warnings', and 'found_variables'
    """
    validation_result = {
        "is_valid": True,
        "warnings": [],
        "found_variables": {},
        "required_dimensions": [],
        "optional_variables": {},
    }

    mappings = load_variable_mappings()

    # Check for coordinate variables
    coordinate_vars = ["latitude", "longitude", "time"]
    for coord_var in coordinate_vars:
        found_var = find_variable(dataset, coord_var, mappings)
        if found_var:
            validation_result["found_variables"][coord_var] = found_var
        else:
            # Try common alternatives
            alternatives = {
                "latitude": ["lat", "LATITUDE", "LAT"],
                "longitude": ["lon", "LONGITUDE", "LON"],
                "time": ["TIME", "time_counter"],
            }

            for alt_name in alternatives.get(coord_var, []):
                if alt_name in dataset.variables:
                    validation_result["found_variables"][coord_var] = alt_name
                    break
            else:
                validation_result["warnings"].append(
                    f"Missing coordinate variable: {coord_var}"
                )

    # Check for pressure or depth
    depth_found = find_variable(dataset, "depth", mappings)
    pressure_found = find_variable(dataset, "pressure", mappings)

    if depth_found:
        validation_result["found_variables"]["depth"] = depth_found
    elif pressure_found:
        validation_result["found_variables"]["pressure"] = pressure_found
    else:
        validation_result["warnings"].append(
            "Missing vertical coordinate (depth or pressure)"
        )
        validation_result["is_valid"] = False

    # Check for common oceanographic variables
    ocean_vars = ["temperature", "salinity"]
    for var_type in ocean_vars:
        found_var = find_variable(dataset, var_type, mappings)
        if found_var:
            validation_result["found_variables"][var_type] = found_var
        else:
            validation_result["warnings"].append(f"Missing {var_type} variable")

    # Check for velocity components (optional)
    velocity_vars = [
        "u_velocity",
        "v_velocity",
        "eastward_velocity",
        "northward_velocity",
    ]
    velocity_found = []
    for var_name in velocity_vars:
        if var_name in dataset.variables:
            velocity_found.append(var_name)

    if velocity_found:
        validation_result["optional_variables"]["velocity"] = velocity_found

    # Check dimensions
    common_dims = ["N_PROF", "N_LEVELS", "N_MEASUREMENTS"]
    for dim_name in common_dims:
        if dim_name in dataset.dims:
            validation_result["required_dimensions"].append(dim_name)

    # Check for distance coordinate (for towyo data)
    if "distance" in dataset.variables:
        validation_result["optional_variables"]["distance"] = "distance"

    # Check for water depth (bathymetry)
    water_depth_vars = ["water_depth", "WATER_DEPTH", "bathy", "bathymetry"]
    for depth_var in water_depth_vars:
        if depth_var in dataset.variables:
            validation_result["optional_variables"]["water_depth"] = depth_var
            break

    return validation_result


def get_data_info(dataset: xr.Dataset) -> Dict:
    """
    Extract useful information about the oceanographic dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset to analyze

    Returns
    -------
    dict
        Information about the dataset structure and content
    """
    info = {
        "dimensions": dict(dataset.sizes),
        "coordinates": list(dataset.coords.keys()),
        "data_variables": list(dataset.data_vars.keys()),
        "global_attributes": dict(dataset.attrs),
        "variable_info": {},
    }

    # Get information about each variable
    for var_name, var_data in dataset.data_vars.items():
        info["variable_info"][var_name] = {
            "dimensions": var_data.dims,
            "shape": var_data.shape,
            "dtype": str(var_data.dtype),
            "attributes": dict(var_data.attrs),
            "units": var_data.attrs.get("units", "unknown"),
            "long_name": var_data.attrs.get("long_name", var_name),
        }

    # Detect data type (profile vs timeseries vs section)
    feature_type = dataset.attrs.get("featureType", "unknown")
    if feature_type == "unknown":
        # Try to infer from dimensions
        if "N_PROF" in dataset.dims:
            if dataset.dims["N_PROF"] > 1:
                if "distance" in dataset.variables:
                    feature_type = "towyo_section"
                else:
                    feature_type = "profile_collection"
            else:
                feature_type = "single_profile"
        elif "time" in dataset.dims and dataset.dims["time"] > 1:
            feature_type = "timeseries"

    info["inferred_type"] = feature_type

    return info
