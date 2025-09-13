"""
Tests for oceanvis_py.core.colormaps module.
"""

import pytest
import matplotlib.pyplot as plt

from oceanvis_py.core.colormaps import (
    get_oceanographic_colormap,
    save_colormap_preferences,
    CMOCEAN_AVAILABLE,
)

# Try importing cmocean for conditional tests
try:
    import cmocean

    CMOCEAN_INSTALLED = True
except ImportError:
    CMOCEAN_INSTALLED = False


class TestGetOceanographicColormap:
    """Test the get_oceanographic_colormap function."""

    def test_temperature_variables(self):
        """Test colormap selection for temperature variables."""
        # All temperature variants should get RdYlBu_r
        temp_vars = ["temperature", "CT", "conservative_temperature"]

        for var in temp_vars:
                cmap = get_oceanographic_colormap(var)
                if CMOCEAN_AVAILABLE:
                    assert hasattr(cmap, "name")
                    assert cmap.name == "TEMP2"
                else:
                    assert cmap == "RdYlBu_r"

    def test_salinity_variables(self):
        """Test colormap selection for salinity variables."""
        sal_vars = ["salinity", "SA", "absolute_salinity"]

        for var in sal_vars:
            cmap = get_oceanographic_colormap(var)
            if CMOCEAN_AVAILABLE:
                # Should get cmocean haline colormap
                assert hasattr(cmap, "name")  # Colormap object
                assert cmap.name == "SAL"
            else:
                assert cmap == "viridis"

    def test_velocity_variables(self):
        """Test colormap selection for velocity variables."""
        vel_vars = ["velocity", "u_velocity", "v_velocity", "u_across", "v_along"]

        for var in vel_vars:
            cmap = get_oceanographic_colormap(var)
            if CMOCEAN_AVAILABLE:
                assert hasattr(cmap, "name")
                assert cmap.name == "balance"
            else:
                assert cmap == "RdBu_r"

    def test_density_variables(self):
        """Test colormap selection for density variables."""
        density_vars = ["density", "sigma2"]

        for var in density_vars:
            cmap = get_oceanographic_colormap(var)
            if CMOCEAN_AVAILABLE:
                assert hasattr(cmap, "name")
                assert cmap.name == "PurGre"
            else:
                assert cmap == "plasma"

    def test_pressure_depth_variables(self):
        """Test colormap selection for pressure/depth variables."""
        depth_vars = ["pressure", "depth"]

        for var in depth_vars:
            cmap = get_oceanographic_colormap(var)
            if CMOCEAN_AVAILABLE:
                assert hasattr(cmap, "name")
                assert cmap.name == "TOPO"
            else:
                assert cmap == "Blues_r"

    def test_unknown_variable(self):
        """Test colormap selection for unknown variables."""
        cmap = get_oceanographic_colormap("unknown_variable")
        if CMOCEAN_AVAILABLE:
            assert hasattr(cmap, "name")
            assert cmap.name == "viridis"
        else:
            assert cmap == "viridis"

    def test_default_parameter(self):
        """Test default parameter value."""
        cmap = get_oceanographic_colormap()  # Should default to 'temperature'
        if CMOCEAN_AVAILABLE:
            assert hasattr(cmap, "name")
            assert cmap.name == "TEMP2"
        else:
            assert cmap == "RdYlBu_r"

    @pytest.mark.skipif(not CMOCEAN_INSTALLED, reason="cmocean not available")
    def test_cmocean_colormap_properties(self):
        """Test properties of returned cmocean colormaps."""
        if CMOCEAN_AVAILABLE:
            # Test that returned colormaps are valid matplotlib colormaps
            cmap = get_oceanographic_colormap("salinity")

            # Should be callable to get colors
            colors = cmap(0.5)  # Get color at midpoint
            assert len(colors) == 4  # RGBA values

            # Should have proper colormap attributes
            assert hasattr(cmap, "name")
            assert hasattr(cmap, "N")


class TestColormapUtilities:
    """Test colormap utility functions."""

    def test_save_colormap_preferences_placeholder(self):
        """Test save_colormap_preferences placeholder function."""
        # This is currently a placeholder, should not raise errors
        preferences = {"temperature": "RdYlBu_r", "salinity": "haline"}

        # Should not raise any errors
        save_colormap_preferences(preferences)

    def test_cmocean_availability_flag(self):
        """Test CMOCEAN_AVAILABLE flag matches actual import."""
        assert CMOCEAN_AVAILABLE == CMOCEAN_INSTALLED


class TestColormapIntegration:
    """Test colormap integration with matplotlib."""

    def test_colormap_with_pcolormesh(self):
        """Test that returned colormaps work with matplotlib pcolormesh."""
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        plt.ioff()

        # Create simple test data
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        X, Y = np.meshgrid(x, y)
        Z = X + Y

        # Test different variable colormaps
        variables = ["temperature", "salinity", "u_velocity"]

        for var in variables:
            fig, ax = plt.subplots()
            cmap = get_oceanographic_colormap(var)

            # Should not raise errors
            pcm = ax.pcolormesh(X, Y, Z, cmap=cmap)

            # Colorbar should work
            plt.colorbar(pcm, ax=ax)

            plt.close(fig)
        plt.close('all')

    def test_colormap_color_range(self):
        """Test that colormaps return valid color values."""
        variables = ["temperature", "salinity", "velocity", "density"]

        for var in variables:
            cmap = get_oceanographic_colormap(var)

            # Test different positions in colormap
            test_values = [0.0, 0.25, 0.5, 0.75, 1.0]

            for val in test_values:
                if CMOCEAN_AVAILABLE and callable(cmap):
                    # Colormap object - call it
                    color = cmap(val)
                else:
                    # String name - get from matplotlib
                    mpl_cmap = plt.cm.get_cmap(cmap)
                    color = mpl_cmap(val)

                # Should return RGBA values
                assert len(color) == 4
                # Values should be between 0 and 1
                assert all(0 <= c <= 1 for c in color)


class TestColormapConsistency:
    """Test consistency of colormap selections."""

    def test_consistent_temperature_colormaps(self):
        """Test that all temperature variants get same colormap."""
        temp_vars = ["temperature", "CT", "conservative_temperature"]
        colormaps = [get_oceanographic_colormap(var) for var in temp_vars]

        # All should be the same
        assert all(cmap == colormaps[0] for cmap in colormaps)

    def test_consistent_salinity_colormaps(self):
        """Test that all salinity variants get same colormap."""
        sal_vars = ["salinity", "SA", "absolute_salinity"]
        colormaps = [get_oceanographic_colormap(var) for var in sal_vars]

        # All should be the same
        assert all(cmap == colormaps[0] for cmap in colormaps)

    def test_consistent_velocity_colormaps(self):
        """Test that all velocity variants get same colormap."""
        vel_vars = ["velocity", "u_velocity", "v_velocity", "u_across", "v_along"]
        colormaps = [get_oceanographic_colormap(var) for var in vel_vars]

        # All should be the same
        assert all(cmap == colormaps[0] for cmap in colormaps)
