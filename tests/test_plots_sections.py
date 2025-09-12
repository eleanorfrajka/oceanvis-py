"""
Tests for oceanvis_py.plots.sections module.
"""

import numpy as np
import pytest
import xarray as xr
import matplotlib.pyplot as plt

from oceanvis_py.plots.sections import (
    plot_section,
    plot_dual_section,
    _add_sigma2_contours,
    _get_units,
    _get_long_name,
)


@pytest.fixture
def sample_towyo_dataset():
    """Create a sample towyo-like dataset for testing."""
    # Create coordinate arrays
    pressure = np.arange(800, 2700, 100)  # 19 pressure levels
    distance = np.linspace(0, 7, 20)  # 20 distance points

    # Create realistic-looking oceanographic data
    P, D = np.meshgrid(pressure, distance, indexing="ij")

    # Conservative Temperature (decreases with depth)
    CT = 4.0 - (P - 800) / 1000 + 0.2 * np.sin(D * 2 * np.pi / 7)

    # Absolute Salinity (increases slightly with depth)
    SA = 34.85 + (P - 800) / 5000 + 0.05 * np.sin(D * 2 * np.pi / 7)

    # Sigma2 density (realistic values)
    sigma2 = 36.0 + (P - 800) / 1000 + 0.1 * np.sin(D * 2 * np.pi / 7)

    # Velocity components
    u_vel = 0.1 * np.sin(D * 2 * np.pi / 7) * np.exp(-((P - 1500) ** 2) / 500000)
    v_vel = 0.05 * np.cos(D * 2 * np.pi / 7) * np.exp(-((P - 1500) ** 2) / 500000)

    # Create xarray dataset
    ds = xr.Dataset(
        {
            "CT": (["pressure", "distance"], CT),
            "SA": (["pressure", "distance"], SA),
            "temperature": (
                ["pressure", "distance"],
                CT - 0.1,
            ),  # Slightly different from CT
            "salinity": (
                ["pressure", "distance"],
                SA - 0.01,
            ),  # Slightly different from SA
            "sigma2": (["pressure", "distance"], sigma2),
            "u_velocity": (["pressure", "distance"], u_vel),
            "v_velocity": (["pressure", "distance"], v_vel),
        },
        coords={
            "pressure": pressure,
            "distance": distance,
        },
        attrs={
            "featureType": "profile",
            "ship": "RV Test",
            "cruise": "TEST001",
        },
    )

    # Add realistic attributes
    ds.CT.attrs = {"long_name": "Conservative Temperature", "units": "degrees_C"}
    ds.SA.attrs = {"long_name": "Absolute Salinity", "units": "g/kg"}
    ds.sigma2.attrs = {"long_name": "Potential density anomaly", "units": "kg/m3"}
    ds.u_velocity.attrs = {"long_name": "Eastward velocity", "units": "m/s"}
    ds.v_velocity.attrs = {"long_name": "Northward velocity", "units": "m/s"}
    ds.pressure.attrs = {"units": "dbar"}
    ds.distance.attrs = {"units": "km"}

    return ds


class TestPlotSection:
    """Test the plot_section function."""

    def test_basic_section_plot(self, sample_towyo_dataset):
        """Test basic section plot creation."""
        fig, ax = plot_section(sample_towyo_dataset, variable="CT")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Check axes labels
        assert "Distance (km)" in ax.get_xlabel()
        assert "Pressure (dbar)" in ax.get_ylabel()

        # Check y-axis is inverted (surface at top)
        assert ax.yaxis_inverted()

        # Check default y-limits
        ylim = ax.get_ylim()
        assert ylim[0] == 2600  # Max depth at bottom (inverted)
        assert ylim[1] == 800  # Min depth at top

        plt.close(fig)

    def test_section_with_sigma2_contours(self, sample_towyo_dataset):
        """Test section plot with sigma2 contours."""
        fig, ax = plot_section(
            sample_towyo_dataset, variable="SA", sigma2_contours=True
        )

        # Check that contours were added (contour collections should exist)
        contour_collections = [
            child for child in ax.get_children() if hasattr(child, "get_paths")
        ]
        assert len(contour_collections) > 1  # Should have contours beyond pcolormesh

        plt.close(fig)

    def test_section_custom_ylims(self, sample_towyo_dataset):
        """Test section plot with custom y-limits."""
        custom_ylims = (2000, 1000)
        fig, ax = plot_section(sample_towyo_dataset, variable="CT", ylims=custom_ylims)

        ylim = ax.get_ylim()
        assert ylim == custom_ylims

        plt.close(fig)

    def test_section_custom_sigma2_levels(self, sample_towyo_dataset):
        """Test section plot with custom sigma2 contour levels."""
        custom_levels = np.arange(36.0, 37.0, 0.1)
        fig, ax = plot_section(
            sample_towyo_dataset,
            variable="CT",
            sigma2_contours=True,
            sigma2_levels=custom_levels,
        )

        plt.close(fig)

    def test_section_different_variables(self, sample_towyo_dataset):
        """Test section plots with different variables."""
        variables = ["CT", "SA", "u_velocity", "v_velocity", "temperature", "salinity"]

        for var in variables:
            fig, ax = plot_section(sample_towyo_dataset, variable=var)
            assert isinstance(fig, plt.Figure)
            assert isinstance(ax, plt.Axes)
            plt.close(fig)


class TestPlotDualSection:
    """Test the plot_dual_section function."""

    def test_basic_dual_section(self, sample_towyo_dataset):
        """Test basic dual section plot creation."""
        fig, (ax1, ax2) = plot_dual_section(
            sample_towyo_dataset, left_var="CT", right_var="SA"
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax1, plt.Axes)
        assert isinstance(ax2, plt.Axes)

        # Check that both axes have proper labels
        assert "Distance (km)" in ax1.get_xlabel()
        assert "Distance (km)" in ax2.get_xlabel()
        assert "Pressure (dbar)" in ax1.get_ylabel()

        # Check y-axes are shared and inverted
        assert ax1.yaxis_inverted()
        assert ax2.yaxis_inverted()

        # Check default spacing for suptitle
        assert fig.subplotpars.top == 0.9

        plt.close(fig)

    def test_dual_section_with_sigma2_contours(self, sample_towyo_dataset):
        """Test dual section plot with sigma2 contours."""
        fig, (ax1, ax2) = plot_dual_section(sample_towyo_dataset, sigma2_contours=True)

        # Both axes should have contours
        for ax in [ax1, ax2]:
            contour_collections = [
                child for child in ax.get_children() if hasattr(child, "get_paths")
            ]
            assert len(contour_collections) > 1

        plt.close(fig)

    def test_dual_section_velocity(self, sample_towyo_dataset):
        """Test dual section plot with velocity components."""
        fig, (ax1, ax2) = plot_dual_section(
            sample_towyo_dataset,
            left_var="u_velocity",
            right_var="v_velocity",
            sigma2_contours=False,
        )

        plt.close(fig)


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_units(self, sample_towyo_dataset):
        """Test _get_units function."""
        # Test variable with units
        units = _get_units(sample_towyo_dataset, "CT")
        assert units == "degrees_C"

        # Test coordinate with units
        units = _get_units(sample_towyo_dataset, "pressure")
        assert units == "dbar"

        # Test variable without units
        units = _get_units(sample_towyo_dataset, "nonexistent")
        assert units == ""

    def test_get_long_name(self, sample_towyo_dataset):
        """Test _get_long_name function."""
        # Test variable with long_name
        name = _get_long_name(sample_towyo_dataset, "CT")
        assert name == "Conservative Temperature"

        # Test variable without long_name (should use variable name)
        name = _get_long_name(sample_towyo_dataset, "nonexistent")
        assert name == "Nonexistent"

    def test_add_sigma2_contours(self, sample_towyo_dataset):
        """Test _add_sigma2_contours function."""
        fig, ax = plt.subplots()

        # Add contours
        _add_sigma2_contours(ax, sample_towyo_dataset, "distance", "pressure")

        # Check that contours were added
        contour_collections = [
            child for child in ax.get_children() if hasattr(child, "get_paths")
        ]
        assert len(contour_collections) > 0

        plt.close(fig)

    def test_add_sigma2_contours_custom_levels(self, sample_towyo_dataset):
        """Test _add_sigma2_contours with custom levels."""
        fig, ax = plt.subplots()
        custom_levels = [36.0, 36.2, 36.4, 36.6]

        _add_sigma2_contours(
            ax, sample_towyo_dataset, "distance", "pressure", levels=custom_levels
        )

        plt.close(fig)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_missing_sigma2_variable(self, sample_towyo_dataset):
        """Test handling when sigma2 variable is missing."""
        # Remove sigma2 from dataset
        ds_no_sigma2 = sample_towyo_dataset.drop_vars("sigma2")

        # Should not raise error, just skip contours
        fig, ax = plot_section(ds_no_sigma2, variable="CT", sigma2_contours=True)

        plt.close(fig)

    def test_different_coordinate_names(self, sample_towyo_dataset):
        """Test with different coordinate names."""
        # Rename coordinates
        ds_renamed = sample_towyo_dataset.rename(
            {"pressure": "depth", "distance": "time"}
        )

        fig, ax = plot_section(
            ds_renamed, variable="CT", x_coord="time", y_coord="depth"
        )

        assert "Time (km)" in ax.get_xlabel()  # Units still hardcoded
        assert "Depth (dbar)" in ax.get_ylabel()

        plt.close(fig)
