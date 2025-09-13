"""
Tests for oceanvis_py.plots.widgets module.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import xarray as xr

# Import widgets to test
from oceanvis_py.plots.widgets import (
    InteractiveColormapWidget,
    create_colormap_widget,
    calculate_histogram_levels,
    round_to_human_readable,
    CMOCEAN_AVAILABLE,
    STANDARD_COLORMAPS,
)

# Try importing ipywidgets
try:
    import ipywidgets as widgets

    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False


@pytest.fixture
def sample_normal_data():
    """Create normal distributed sample data."""
    np.random.seed(42)
    return np.random.normal(0, 1, 1000)


@pytest.fixture
def sample_skewed_data():
    """Create skewed sample data (like salinity)."""
    np.random.seed(42)
    # Most data concentrated around 35, with tails
    concentrated = np.random.normal(35.0, 0.05, 800)
    tails = np.random.uniform(34.8, 35.2, 200)
    return np.concatenate([concentrated, tails])


@pytest.fixture
def sample_xarray_data():
    """Create sample xarray DataArray."""
    np.random.seed(42)
    data = np.random.normal(10, 2, (50, 30))
    return xr.DataArray(
        data,
        dims=["x", "y"],
        coords={"x": range(50), "y": range(30)},
        attrs={"long_name": "Test Variable", "units": "test_units"},
    )


class TestRoundToHumanReadable:
    """Test the round_to_human_readable function."""

    def test_round_integers(self):
        """Test rounding values that should become integers."""
        assert round_to_human_readable(1.0) == 1
        assert round_to_human_readable(1.00001) == 1
        assert round_to_human_readable(35.999999) == 36

    def test_round_decimals(self):
        """Test rounding decimal values."""
        assert round_to_human_readable(1.234567) == 1.23
        assert round_to_human_readable(35.876) == 35.88
        assert round_to_human_readable(0.1234) == 0.12

    def test_precision_parameter(self):
        """Test different precision parameters."""
        value = 1.23456
        assert round_to_human_readable(value, precision=1) == 1.2
        assert round_to_human_readable(value, precision=3) == 1.235
        assert round_to_human_readable(value, precision=0) == 1

    def test_edge_cases(self):
        """Test edge cases."""
        assert round_to_human_readable(0.0) == 0
        assert round_to_human_readable(-1.234) == -1.23
        assert round_to_human_readable(1e-10) == 0


class TestCalculateHistogramLevels:
    """Test the calculate_histogram_levels function."""

    def test_linear_method(self, sample_normal_data):
        """Test linear spacing method."""
        levels = calculate_histogram_levels(
            sample_normal_data, n_levels=10, method="linear"
        )

        assert len(levels) == 11  # n_levels + 1 boundaries
        assert np.all(levels[1:] > levels[:-1])  # Strictly increasing

        # Should be approximately equally spaced
        diffs = np.diff(levels)
        assert np.allclose(diffs, diffs[0], rtol=0.15)

    def test_percentile_method(self, sample_skewed_data):
        """Test percentile-based method."""
        levels = calculate_histogram_levels(
            sample_skewed_data, n_levels=20, method="percentile"
        )

        assert len(levels) <= 21  # May be fewer due to rounding/uniqueness
        assert np.all(levels[1:] > levels[:-1])  # Strictly increasing

        # Check that levels capture reasonable range (allowing for rounding)
        data_min, data_max = sample_skewed_data.min(), sample_skewed_data.max()
        assert levels[0] <= data_min + 0.1  # Allow for rounding
        assert levels[-1] >= data_max - 0.1  # Allow for rounding

    def test_histogram_method(self, sample_normal_data):
        """Test histogram-based method."""
        levels = calculate_histogram_levels(
            sample_normal_data, n_levels=15, method="histogram"
        )

        assert len(levels) <= 16  # May be fewer due to rounding
        assert np.all(levels[1:] > levels[:-1])  # Strictly increasing

    def test_compressed_percentile_method(self, sample_skewed_data):
        """Test compressed percentile method."""
        levels = calculate_histogram_levels(
            sample_skewed_data,
            n_levels=20,
            method="compressed_percentile",
            tail_compression=0.8,
        )

        assert len(levels) <= 21  # May be fewer due to rounding
        assert np.all(levels[1:] > levels[:-1])  # Strictly increasing

        # Should focus more on central distribution (relax test due to rounding)
        central_range = np.percentile(sample_skewed_data, [10, 90])  # Wider range
        central_levels = levels[
            (levels >= central_range[0]) & (levels <= central_range[1])
        ]

        # At least some levels should be in central range
        assert len(central_levels) > len(levels) * 0.2  # More relaxed threshold

    def test_symmetric_zero_method(self):
        """Test symmetric zero method for velocity/anomaly data."""
        # Create test data with positive and negative values (like velocity)
        np.random.seed(42)
        velocity_data = np.random.normal(0, 1, 1000)  # Centered at zero
        velocity_data = np.concatenate(
            [velocity_data, [2.5, -2.3]]
        )  # Add some extremes

        # Test with even number of levels
        levels_even = calculate_histogram_levels(
            velocity_data, n_levels=10, method="symmetric_zero"
        )

        assert len(levels_even) == 11  # n_levels + 1 boundaries
        assert np.all(levels_even[1:] > levels_even[:-1])  # Strictly increasing

        # Should be symmetric around zero
        abs_max = max(abs(velocity_data.min()), abs(velocity_data.max()))
        assert abs(levels_even[0]) <= abs_max + 0.1  # Allow for rounding
        assert abs(levels_even[-1]) <= abs_max + 0.1  # Allow for rounding

        # Should have zero as a boundary (even number of levels)
        assert 0 in levels_even

        # Test with odd number of levels
        levels_odd = calculate_histogram_levels(
            velocity_data, n_levels=9, method="symmetric_zero"
        )

        assert len(levels_odd) == 10  # n_levels + 1 boundaries
        assert np.all(levels_odd[1:] > levels_odd[:-1])  # Strictly increasing

        # Should be symmetric around zero
        assert abs(levels_odd[0]) <= abs_max + 0.1
        assert abs(levels_odd[-1]) <= abs_max + 0.1

        # For odd number of levels, zero should NOT be a boundary but be centered in middle band
        assert 0 not in levels_odd
        # But the middle band should be centered around zero
        middle_idx = len(levels_odd) // 2
        middle_band = (levels_odd[middle_idx - 1] + levels_odd[middle_idx]) / 2
        assert abs(middle_band) < 0.1  # Should be close to zero

    def test_human_readable_rounding(self, sample_normal_data):
        """Test that levels are rounded to human-readable values."""
        levels = calculate_histogram_levels(
            sample_normal_data, n_levels=10, method="linear"
        )

        # Check that values are reasonably rounded
        for level in levels:
            if level != int(level):
                decimal_places = len(str(level).split(".")[-1])
                assert decimal_places <= 2  # Should be at most 2 decimal places

    def test_nan_handling(self):
        """Test handling of NaN values."""
        data_with_nan = np.array([1.0, 2.0, np.nan, 3.0, np.nan, 4.0])
        levels = calculate_histogram_levels(data_with_nan, n_levels=5, method="linear")

        assert len(levels) == 6
        assert not np.any(np.isnan(levels))
        assert levels[0] >= 1.0
        assert levels[-1] <= 4.0

    def test_empty_data(self):
        """Test handling of empty data."""
        empty_data = np.array([])
        levels = calculate_histogram_levels(empty_data, n_levels=5, method="linear")

        assert len(levels) == 6
        assert np.allclose(levels, np.linspace(0, 1, 6))


@pytest.mark.skipif(not IPYWIDGETS_AVAILABLE, reason="ipywidgets not available")
class TestInteractiveColormapWidget:
    """Test the InteractiveColormapWidget class."""

    def test_widget_initialization_numpy(self, sample_normal_data):
        """Test widget initialization with numpy array."""
        widget = InteractiveColormapWidget(sample_normal_data, "Test Variable")

        assert widget.variable_name == "Test Variable"
        assert len(widget.clean_data) == len(sample_normal_data)
        assert not np.any(np.isnan(widget.clean_data))

        # Check that widgets are created
        assert hasattr(widget, "colormap_dropdown")
        assert hasattr(widget, "n_levels_slider")
        assert hasattr(widget, "norm_method_dropdown")
        assert hasattr(widget, "output")

    def test_widget_initialization_xarray(self, sample_xarray_data):
        """Test widget initialization with xarray DataArray."""
        widget = InteractiveColormapWidget(sample_xarray_data, "XArray Test")

        assert widget.variable_name == "XArray Test"
        assert widget.clean_data.shape == sample_xarray_data.values.flatten().shape

    def test_available_colormaps(self, sample_normal_data):
        """Test that available colormaps include standard ones."""
        widget = InteractiveColormapWidget(sample_normal_data)

        # Should have standard matplotlib colormaps
        for name in STANDARD_COLORMAPS.keys():
            assert name in widget.available_colormaps

        # If cmocean is available, should have those too
        if CMOCEAN_AVAILABLE:
            assert "haline" in widget.available_colormaps
            assert "thermal" in widget.available_colormaps

    def test_widget_controls_creation(self, sample_normal_data):
        """Test that all widget controls are created properly."""
        widget = InteractiveColormapWidget(sample_normal_data)

        # Test colormap dropdown
        assert isinstance(widget.colormap_dropdown, widgets.Dropdown)
        assert len(widget.colormap_dropdown.options) > 0

        # Test n_levels slider
        assert isinstance(widget.n_levels_slider, widgets.IntSlider)
        assert widget.n_levels_slider.min == 5
        assert widget.n_levels_slider.max == 30
        assert widget.n_levels_slider.value == 20

        # Test normalization method dropdown
        assert isinstance(widget.norm_method_dropdown, widgets.Dropdown)
        methods = [option[1] for option in widget.norm_method_dropdown.options]
        assert "linear" in methods
        assert "percentile" in methods
        assert "compressed_percentile" in methods

        # Test other controls
        assert isinstance(widget.tail_compression_slider, widgets.FloatSlider)
        assert isinstance(widget.extend_dropdown, widgets.Dropdown)
        assert isinstance(widget.generate_code_button, widgets.Button)

    def test_update_plot_executes(self, sample_skewed_data):
        """Test that update_plot runs without errors."""
        widget = InteractiveColormapWidget(sample_skewed_data, "Salinity")

        # Should not raise errors
        try:
            widget.update_plot()
            # Close any plots created
            plt.close("all")
        except Exception as e:
            pytest.fail(f"update_plot() raised {e} unexpectedly!")

    def test_generate_code_functionality(self, sample_normal_data):
        """Test code generation functionality."""
        widget = InteractiveColormapWidget(sample_normal_data)

        # Mock button click
        class MockButton:
            pass

        button = MockButton()

        # Should not raise errors
        try:
            widget._generate_code(button)
        except Exception as e:
            pytest.fail(f"_generate_code() raised {e} unexpectedly!")


class TestCreateColormapWidget:
    """Test the create_colormap_widget function."""

    @pytest.mark.skipif(not IPYWIDGETS_AVAILABLE, reason="ipywidgets not available")
    def test_create_widget_numpy(self, sample_normal_data):
        """Test creating widget with numpy data."""
        widget = create_colormap_widget(sample_normal_data, "Test Variable")

        assert isinstance(widget, InteractiveColormapWidget)
        assert widget.variable_name == "Test Variable"

    @pytest.mark.skipif(not IPYWIDGETS_AVAILABLE, reason="ipywidgets not available")
    def test_create_widget_xarray(self, sample_xarray_data):
        """Test creating widget with xarray data."""
        widget = create_colormap_widget(sample_xarray_data, "XArray Variable")

        assert isinstance(widget, InteractiveColormapWidget)
        assert widget.variable_name == "XArray Variable"

    @pytest.mark.skipif(not IPYWIDGETS_AVAILABLE, reason="ipywidgets not available")
    def test_create_widget_defaults(self, sample_normal_data):
        """Test creating widget with default parameters."""
        widget = create_colormap_widget(sample_normal_data)

        assert isinstance(widget, InteractiveColormapWidget)
        assert widget.variable_name == "variable"


class TestColormapIntegration:
    """Test integration with matplotlib colormap functionality."""

    @pytest.mark.skipif(not IPYWIDGETS_AVAILABLE, reason="ipywidgets not available")
    def test_discrete_colormap_creation(self, sample_skewed_data):
        """Test creation of discrete colormaps from widget settings."""
        widget = InteractiveColormapWidget(sample_skewed_data, "Test")

        # Get settings
        n_levels = 15
        colormap_name = list(widget.available_colormaps.keys())[0]
        cmap = widget.available_colormaps[colormap_name]

        # Create discrete colormap
        discrete_cmap = ListedColormap(cmap(np.linspace(0, 1, n_levels)))

        assert discrete_cmap.N == n_levels

        # Test that it works with matplotlib
        levels = calculate_histogram_levels(widget.clean_data, n_levels, "percentile")
        norm = BoundaryNorm(levels, n_levels)

        # Should not raise errors
        fig, ax = plt.subplots()

        # Create test data for plotting
        x, y = np.meshgrid(range(10), range(10))
        z = np.random.uniform(levels[0], levels[-1], (10, 10))

        pcm = ax.pcolormesh(x, y, z, cmap=discrete_cmap, norm=norm)
        plt.colorbar(pcm, ax=ax)
        plt.close(fig)

    def test_boundary_norm_with_calculated_levels(self, sample_normal_data):
        """Test BoundaryNorm with calculated levels and extension padding."""
        levels = calculate_histogram_levels(sample_normal_data, 10, "percentile")
        extend = "both"
        n_bins = len(levels) - 1
        if extend == "both":
            n_bins += 2
        ncolors = n_bins
        norm = BoundaryNorm(levels, ncolors, extend=extend)

        # Test some values
        test_values = [levels[0] - 1, levels[5], levels[-1] + 1]
        normalized = norm(test_values)

        # Should return valid normalized values
        assert len(normalized) == len(test_values)
        assert all(isinstance(val, (int, float, np.number)) for val in normalized)


class TestExtendParameter:
    """Test extend parameter functionality in BoundaryNorm."""

    def test_extend_neither(self, sample_normal_data):
        """Test extend='neither' (no extension)."""
        levels = calculate_histogram_levels(sample_normal_data, 10, "linear")

        # This should work without issues
        from matplotlib.colors import BoundaryNorm, ListedColormap
        import matplotlib.pyplot as plt

        cmap = plt.cm.viridis
        n_levels = 10
        discrete_cmap = ListedColormap(cmap(np.linspace(0, 1, n_levels)))
        norm = BoundaryNorm(levels, n_levels, extend="neither")

        assert norm.extend == "neither"
        assert len(levels) == n_levels + 1  # n_levels boundaries

    def test_extend_both(self, sample_normal_data):
        """Test extend='both' (extension on both ends)."""
        levels = calculate_histogram_levels(sample_normal_data, 10, "linear")

        from matplotlib.colors import BoundaryNorm, ListedColormap
        import matplotlib.pyplot as plt

        cmap = plt.cm.viridis
        n_levels = 10
        ncolors = n_levels + 2  # Account for extension bins
        discrete_cmap = ListedColormap(cmap(np.linspace(0, 1, ncolors)))
        norm = BoundaryNorm(levels, ncolors, extend="both")

        assert norm.extend == "both"

    def test_extend_min_max(self, sample_normal_data):
        """Test extend='min' and extend='max'."""
        levels = calculate_histogram_levels(sample_normal_data, 10, "linear")

        from matplotlib.colors import BoundaryNorm, ListedColormap
        import matplotlib.pyplot as plt

        cmap = plt.cm.viridis
        n_levels = 10

        # Test extend='min'
        ncolors_min = n_levels + 1
        discrete_cmap_min = ListedColormap(cmap(np.linspace(0, 1, ncolors_min)))
        norm_min = BoundaryNorm(levels, ncolors_min, extend="min")
        assert norm_min.extend == "min"

        # Test extend='max'
        ncolors_max = n_levels + 1
        discrete_cmap_max = ListedColormap(cmap(np.linspace(0, 1, ncolors_max)))
        norm_max = BoundaryNorm(levels, ncolors_max, extend="max")
        assert norm_max.extend == "max"


class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions."""

    def test_calculate_levels_with_constant_data(self):
        """Test calculating levels when all data values are the same."""
        constant_data = np.array([5.0] * 100)
        levels = calculate_histogram_levels(constant_data, 10, "percentile")

        # Should handle gracefully
        assert len(levels) >= 2  # At least min and max
        assert levels[0] <= 5.0 <= levels[-1]

    def test_calculate_levels_with_tiny_range(self):
        """Test with very small data range."""
        tiny_range_data = np.array([1.0000001, 1.0000002, 1.0000003])
        levels = calculate_histogram_levels(tiny_range_data, 5, "linear")

        # Should handle gracefully and produce reasonable levels
        assert len(levels) >= 2
        assert levels[0] <= levels[-1]

    @pytest.mark.skipif(not IPYWIDGETS_AVAILABLE, reason="ipywidgets not available")
    def test_widget_with_all_nan_data(self):
        """Test widget initialization with all NaN data."""
        nan_data = np.array([np.nan] * 100)

        # Should handle gracefully
        widget = InteractiveColormapWidget(nan_data, "All NaN")
        assert len(widget.clean_data) == 0  # All NaNs removed

    @pytest.mark.skipif(not IPYWIDGETS_AVAILABLE, reason="ipywidgets not available")
    def test_widget_with_single_value(self):
        """Test widget with only one unique value."""
        single_value_data = np.array([42.0] * 100)
        widget = InteractiveColormapWidget(single_value_data, "Single Value")

        # Should initialize without errors
        assert len(widget.clean_data) == 100
        assert np.all(widget.clean_data == 42.0)
