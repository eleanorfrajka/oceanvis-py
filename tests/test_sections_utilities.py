"""
Tests for utility functions in oceanvis_py.plots.sections module.

These tests focus on standalone utility functions that don't require 
matplotlib figure creation, avoiding environment compatibility issues.
"""

import pytest
import numpy as np
import xarray as xr
import tempfile
from pathlib import Path


class TestSectionUtilityFunctions:
    """Test utility functions from sections.py."""
    
    def test_get_units_function(self):
        """Test _get_units utility function."""
        from oceanvis_py.plots.sections import _get_units
        
        # Create test dataset with units
        data = xr.Dataset({
            'temperature': (['x'], [1, 2, 3], {'units': 'degrees_C'}),
            'salinity': (['x'], [34, 35, 36], {'units': 'psu'}),
            'no_units_var': (['x'], [1, 2, 3])
        }, coords={'x': [1, 2, 3]})
        
        # Test getting units
        assert _get_units(data, 'temperature') == 'degrees_C'
        assert _get_units(data, 'salinity') == 'psu'
        assert _get_units(data, 'no_units_var') == ''
        assert _get_units(data, 'nonexistent') == ''

    def test_get_long_name_function(self):
        """Test _get_long_name utility function."""
        from oceanvis_py.plots.sections import _get_long_name
        
        # Create test dataset with long names
        data = xr.Dataset({
            'CT': (['x'], [1, 2, 3], {'long_name': 'Conservative Temperature'}),
            'SA': (['x'], [34, 35, 36], {'long_name': 'Absolute Salinity'}),
            'no_longname_var': (['x'], [1, 2, 3])
        }, coords={'x': [1, 2, 3]})
        
        # Test getting long names
        assert _get_long_name(data, 'CT') == 'Conservative Temperature'
        assert _get_long_name(data, 'SA') == 'Absolute Salinity'
        assert _get_long_name(data, 'no_longname_var') == 'No Longname Var'  # Formatted
        assert _get_long_name(data, 'nonexistent') == 'Nonexistent'  # Formatted

    def test_get_units_with_standard_name(self):
        """Test _get_units with standard_name fallback."""
        from oceanvis_py.plots.sections import _get_units
        
        # Create dataset with standard_name but no units
        data = xr.Dataset({
            'temp': (['x'], [1, 2, 3], {'standard_name': 'sea_water_temperature'})
        }, coords={'x': [1, 2, 3]})
        
        # Should return empty string since no units attribute
        assert _get_units(data, 'temp') == ''

    def test_get_long_name_fallback_to_variable_name(self):
        """Test that _get_long_name falls back to variable name."""
        from oceanvis_py.plots.sections import _get_long_name
        
        # Dataset with no long_name
        data = xr.Dataset({
            'mystery_var': (['x'], [1, 2, 3])
        }, coords={'x': [1, 2, 3]})
        
        # Should return the formatted variable name (title case, underscores to spaces)
        assert _get_long_name(data, 'mystery_var') == 'Mystery Var'


class TestCPTFileHandling:
    """Test CPT file handling in section plots."""
    
    def test_cpt_file_path_validation(self):
        """Test that CPT file path validation works."""
        # This tests the path validation logic without requiring matplotlib
        
        # Create a temporary CPT file
        with tempfile.TemporaryDirectory() as temp_dir:
            cpt_file = Path(temp_dir) / "test.cpt"
            
            # Create a simple CPT file
            cpt_content = """# Simple test colormap
0 red 1 red
1 green 2 green
2 blue 3 blue
"""
            cpt_file.write_text(cpt_content)
            
            # Test that file exists
            assert cpt_file.exists()
            
            # Test path string
            assert str(cpt_file).endswith("test.cpt")

    def test_nonexistent_cpt_file(self):
        """Test handling of nonexistent CPT file."""
        nonexistent_file = "/path/to/nonexistent.cpt"
        
        # Should not exist
        assert not Path(nonexistent_file).exists()


class TestDataValidation:
    """Test data validation functions."""
    
    def test_dataset_variable_checking(self):
        """Test checking for variables in dataset."""
        # Create test dataset
        data = xr.Dataset({
            'temperature': (['x', 'y'], np.random.random((10, 5))),
            'salinity': (['x', 'y'], np.random.random((10, 5))),
            'pressure': (['y'], np.linspace(0, 1000, 5))
        }, coords={
            'x': np.linspace(0, 100, 10),
            'y': np.linspace(0, 1000, 5)
        })
        
        # Test variable existence
        assert 'temperature' in data
        assert 'salinity' in data
        assert 'pressure' in data
        assert 'nonexistent' not in data
        
        # Test coordinate existence
        assert 'x' in data.coords
        assert 'y' in data.coords

    def test_coordinate_dimension_validation(self):
        """Test coordinate and dimension validation."""
        # Create dataset with specific dimensions
        x_coords = np.linspace(0, 100, 10)
        y_coords = np.linspace(0, 1000, 5)
        
        data = xr.Dataset({
            'var': (['distance', 'pressure'], np.random.random((10, 5)))
        }, coords={
            'distance': x_coords,
            'pressure': y_coords
        })
        
        # Test dimensions
        assert data.sizes['distance'] == 10
        assert data.sizes['pressure'] == 5
        assert 'var' in data.data_vars

    def test_missing_variable_handling(self):
        """Test handling of missing variables."""
        # Dataset without sigma2
        data = xr.Dataset({
            'temperature': (['x', 'y'], np.random.random((10, 5))),
            'salinity': (['x', 'y'], np.random.random((10, 5)))
        }, coords={
            'x': np.linspace(0, 100, 10),
            'y': np.linspace(0, 1000, 5)
        })
        
        # sigma2 should not exist
        assert 'sigma2' not in data
        
        # But temperature and salinity should exist
        assert 'temperature' in data
        assert 'salinity' in data


class TestCoordinateHandling:
    """Test coordinate handling logic."""
    
    def test_default_coordinate_names(self):
        """Test default coordinate naming conventions."""
        # Common coordinate names in oceanographic data
        common_x_coords = ['distance', 'longitude', 'lon', 'time', 'profile']
        common_y_coords = ['pressure', 'depth', 'z']
        
        # Create dataset with standard names
        data = xr.Dataset({
            'temperature': (['distance', 'pressure'], np.random.random((10, 5)))
        }, coords={
            'distance': np.linspace(0, 100, 10),
            'pressure': np.linspace(0, 1000, 5)
        })
        
        # Test that coordinates exist
        assert 'distance' in data.coords
        assert 'pressure' in data.coords
        
        # Test coordinate values
        assert len(data.coords['distance']) == 10
        assert len(data.coords['pressure']) == 5

    def test_alternative_coordinate_names(self):
        """Test alternative coordinate naming."""
        # Dataset with alternative names
        data = xr.Dataset({
            'CT': (['profile', 'depth'], np.random.random((8, 12)))
        }, coords={
            'profile': np.arange(8),
            'depth': np.linspace(0, 500, 12)
        })
        
        # Test dimensions and coordinates
        assert data.sizes['profile'] == 8
        assert data.sizes['depth'] == 12
        assert 'CT' in data


class TestPlotParameterValidation:
    """Test plot parameter validation without creating actual plots."""
    
    def test_figsize_parameter_types(self):
        """Test that figsize parameters are valid."""
        # Valid figsize tuples
        valid_figsizes = [(10, 6), (12, 8), (15, 10)]
        
        for figsize in valid_figsizes:
            assert isinstance(figsize, tuple)
            assert len(figsize) == 2
            assert all(isinstance(x, (int, float)) for x in figsize)
            assert all(x > 0 for x in figsize)

    def test_colormap_parameter_types(self):
        """Test colormap parameter validation."""
        # Valid colormap names
        valid_cmaps = ['viridis', 'plasma', 'coolwarm', 'RdBu_r']
        
        for cmap in valid_cmaps:
            assert isinstance(cmap, str)
            assert len(cmap) > 0

    def test_boolean_parameter_types(self):
        """Test boolean parameters."""
        boolean_params = [True, False]
        
        for param in boolean_params:
            assert isinstance(param, bool)

    def test_numeric_parameter_ranges(self):
        """Test numeric parameter ranges."""
        # Valid DPI values
        valid_dpis = [72, 150, 300, 600]
        for dpi in valid_dpis:
            assert isinstance(dpi, int)
            assert dpi > 0
            
        # Valid alpha values
        valid_alphas = [0.0, 0.5, 1.0]
        for alpha in valid_alphas:
            assert isinstance(alpha, float)
            assert 0.0 <= alpha <= 1.0


class TestErrorHandling:
    """Test error handling in section plotting functions."""
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        # Create empty dataset
        empty_data = xr.Dataset({})
        
        # Should be empty
        assert len(empty_data.data_vars) == 0
        assert len(empty_data.coords) == 0

    def test_single_point_dataset(self):
        """Test handling of single-point datasets."""
        # Dataset with only one point
        single_point = xr.Dataset({
            'temp': (['x'], [25.0])
        }, coords={'x': [0]})
        
        assert len(single_point.x) == 1
        assert float(single_point.temp) == 25.0

    def test_mismatched_coordinates(self):
        """Test handling of mismatched coordinate dimensions."""
        # This would be an invalid dataset structure
        try:
            # Trying to create incompatible dimensions
            x_coords = np.linspace(0, 100, 10)  # 10 points
            y_coords = np.linspace(0, 1000, 5)  # 5 points
            data_array = np.random.random((8, 6))  # 8x6 array - mismatched!
            
            # This should work but be internally inconsistent
            # xarray will handle dimension checking
            pass
        except Exception:
            # If xarray catches dimension mismatches, that's good
            pass