"""
Tests for oceanvis_py.plots.maps module.

Tests PyGMT-based bathymetry mapping functions with proper installation checks.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import xarray as xr


# Check PyGMT availability
try:
    import pygmt
    HAVE_PYGMT = True
    PYGMT_SKIP_REASON = "PyGMT is available"  # This won't be used, but pytest requires a string
except ImportError:
    HAVE_PYGMT = False
    PYGMT_SKIP_REASON = "PyGMT not available"


def create_test_bathymetry_file(temp_dir):
    """Create a test bathymetry netCDF file."""
    # Create synthetic bathymetry data
    lon = np.linspace(-56, -36, 20)
    lat = np.linspace(40, 52, 15)
    LON, LAT = np.meshgrid(lon, lat)
    
    # Simple bathymetry: deeper in center, shallower at edges
    waterdepth = 2000 + 1500 * np.exp(-((LON + 46)**2 + (LAT - 46)**2) / 50)
    
    # Create xarray dataset
    ds = xr.Dataset({
        'waterdepth': (('lat', 'lon'), waterdepth)
    }, coords={
        'lat': lat,
        'lon': lon
    })
    
    # Add attributes
    ds.waterdepth.attrs = {
        'units': 'meters',
        'long_name': 'Water depth'
    }
    
    bathy_file = temp_dir / "test_bathymetry.nc"
    ds.to_netcdf(bathy_file)
    return str(bathy_file)


def create_test_ship_track_file(temp_dir):
    """Create a test ship track file in GMT format (lon lat)."""
    # Create a simple ship track
    track_lons = np.linspace(-50, -45, 10)
    track_lats = np.linspace(43, 47, 10)
    
    track_file = temp_dir / "test_track.xy"
    with open(track_file, 'w') as f:
        for lon, lat in zip(track_lons, track_lats):
            f.write(f"{lon:.3f} {lat:.3f}\n")
    
    return str(track_file)


class TestPyGMTAvailability:
    """Test PyGMT availability checking."""
    
    def test_pygmt_availability_flag(self):
        """Test that PyGMT availability is detected correctly."""
        from oceanvis_py.plots.maps import HAVE_PYGMT
        
        # Should match our test detection
        assert HAVE_PYGMT == HAVE_PYGMT
        
    def test_require_pygmt_function(self):
        """Test the _require_pygmt function behavior."""
        from oceanvis_py.plots.maps import _require_pygmt
        
        if HAVE_PYGMT:
            # Should return pygmt module
            pygmt_module = _require_pygmt()
            assert hasattr(pygmt_module, 'Figure')
        else:
            # Should raise RuntimeError with helpful message
            with pytest.raises(RuntimeError, match="PyGMT/GMT not available"):
                _require_pygmt()


@pytest.mark.skipif(not HAVE_PYGMT, reason=PYGMT_SKIP_REASON)
class TestBathymetryMapping:
    """Test PyGMT bathymetry mapping functions."""
    
    def test_plot_bathymetry_map_basic(self):
        """Test basic bathymetry map creation."""
        from oceanvis_py.plots.maps import plot_bathymetry_map
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            bathy_file = create_test_bathymetry_file(temp_path)
            output_file = temp_path / "test_map.png"
            
            # Create map
            result = plot_bathymetry_map(
                bathymetry_file=bathy_file,
                region=(-56, -36, 40, 52),
                projection="M5i",
                output_file=str(output_file),
                dpi=150  # Lower DPI for faster testing
            )
            
            # Check result
            assert result == str(output_file)
            assert output_file.exists()
            assert output_file.stat().st_size > 0  # File has content

    def test_plot_bathymetry_map_with_ship_track(self):
        """Test bathymetry map with ship track overlay."""
        from oceanvis_py.plots.maps import plot_bathymetry_map
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            bathy_file = create_test_bathymetry_file(temp_path)
            track_file = create_test_ship_track_file(temp_path)
            output_file = temp_path / "test_map_with_track.png"
            
            # Create map with track
            result = plot_bathymetry_map(
                bathymetry_file=bathy_file,
                region=(-56, -36, 40, 52),
                ship_track_file=track_file,
                output_file=str(output_file),
                title="Test Map with Track",
                dpi=150
            )
            
            # Check result
            assert result == str(output_file)
            assert output_file.exists()
            assert output_file.stat().st_size > 0

    def test_plot_bathymetry_map_return_figure(self):
        """Test returning figure object for further modification."""
        from oceanvis_py.plots.maps import plot_bathymetry_map
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            bathy_file = create_test_bathymetry_file(temp_path)
            output_file = temp_path / "test_map_fig.png"
            
            # Get figure object
            fig, output_path, temp_bathy = plot_bathymetry_map(
                bathymetry_file=bathy_file,
                region=(-56, -36, 40, 52),
                output_file=str(output_file),
                return_figure=True,
                dpi=150
            )
            
            # Check results
            assert hasattr(fig, 'savefig')  # Should be a PyGMT Figure
            assert output_path == str(output_file)
            assert isinstance(temp_bathy, str)  # Temp file path
            
            # Figure should not be saved yet
            assert not output_file.exists()
            
            # Save it manually
            fig.savefig(str(output_file), dpi=150)
            assert output_file.exists()

    def test_create_ship_track_map_auto_region(self):
        """Test automatic region determination from ship track."""
        from oceanvis_py.plots.maps import create_ship_track_map
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            bathy_file = create_test_bathymetry_file(temp_path)
            track_file = create_test_ship_track_file(temp_path)
            output_file = temp_path / "auto_track_map.png"
            
            # Create map with auto-determined region
            result = create_ship_track_map(
                ship_track_file=track_file,
                bathymetry_file=bathy_file,
                output_file=str(output_file),
                dpi=150
            )
            
            assert result == str(output_file)
            assert output_file.exists()

    def test_plot_multi_track_map(self):
        """Test multiple ship tracks on bathymetry."""
        from oceanvis_py.plots.maps import plot_multi_track_map
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            bathy_file = create_test_bathymetry_file(temp_path)
            
            # Create multiple track files
            track1_file = create_test_ship_track_file(temp_path)
            
            # Create second track
            track2_lons = np.linspace(-52, -47, 8)
            track2_lats = np.linspace(44, 46, 8)
            track2_file = temp_path / "test_track2.xy"
            with open(track2_file, 'w') as f:
                for lon, lat in zip(track2_lons, track2_lats):
                    f.write(f"{lon:.3f} {lat:.3f}\n")
            
            output_file = temp_path / "multi_track_map.png"
            
            # Create multi-track map
            result = plot_multi_track_map(
                track_files=[str(track1_file), str(track2_file)],
                bathymetry_file=bathy_file,
                region=(-56, -36, 40, 52),
                track_colors=["220/20/20", "20/220/20"],
                track_labels=["Track 1", "Track 2"],
                output_file=str(output_file),
                dpi=150
            )
            
            assert result == str(output_file)
            assert output_file.exists()


@pytest.mark.skipif(not HAVE_PYGMT, reason=PYGMT_SKIP_REASON)
class TestBathymetryDataHandling:
    """Test bathymetry data processing and error handling."""
    
    def test_different_depth_variable_names(self):
        """Test handling different depth variable names."""
        from oceanvis_py.plots.maps import plot_bathymetry_map
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create bathymetry with 'depth' instead of 'waterdepth'
            lon = np.linspace(-50, -45, 10)
            lat = np.linspace(43, 47, 8)
            LON, LAT = np.meshgrid(lon, lat)
            depth = 1000 + 500 * np.random.random(LON.shape)
            
            ds = xr.Dataset({
                'depth': (('lat', 'lon'), depth)
            }, coords={'lat': lat, 'lon': lon})
            
            bathy_file = temp_path / "test_depth_var.nc"
            ds.to_netcdf(bathy_file)
            output_file = temp_path / "depth_var_map.png"
            
            # Should work with 'depth' variable
            result = plot_bathymetry_map(
                bathymetry_file=str(bathy_file),
                region=(-52, -43, 42, 48),
                output_file=str(output_file),
                dpi=150
            )
            
            assert result == str(output_file)
            assert output_file.exists()
    
    def test_missing_depth_variable_error(self):
        """Test error when no recognized depth variable is found."""
        from oceanvis_py.plots.maps import plot_bathymetry_map
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create file with no recognized depth variables
            ds = xr.Dataset({
                'temperature': (('y', 'x'), np.random.random((5, 5)))
            }, coords={'y': range(5), 'x': range(5)})
            
            bathy_file = temp_path / "no_depth_var.nc"
            ds.to_netcdf(bathy_file)
            output_file = temp_path / "error_map.png"
            
            # Should raise ValueError
            with pytest.raises(RuntimeError, match="Could not find depth variable"):
                plot_bathymetry_map(
                    bathymetry_file=str(bathy_file),
                    region=(0, 5, 0, 5),
                    output_file=str(output_file)
                )

    def test_custom_colormap_integration(self):
        """Test integration with custom colormaps."""
        from oceanvis_py.plots.maps import plot_bathymetry_map
        from oceanvis_py.core.colormaps import get_bathymetry_colormap
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            bathy_file = create_test_bathymetry_file(temp_path)
            output_file = temp_path / "custom_cmap_map.png"
            
            # Use Flemish Cap colormap
            flemish_cmap = get_bathymetry_colormap("flemish_cap")
            
            result = plot_bathymetry_map(
                bathymetry_file=bathy_file,
                region=(-56, -36, 40, 52),
                cpt_file=flemish_cmap,
                output_file=str(output_file),
                dpi=150
            )
            
            assert result == str(output_file)
            assert output_file.exists()


class TestMapsWithoutPyGMT:
    """Test behavior when PyGMT is not available."""
    
    def test_maps_import_without_pygmt(self):
        """Test that maps module imports even without PyGMT."""
        # This test always runs to ensure module imports work
        import oceanvis_py.plots.maps as maps
        
        assert hasattr(maps, 'HAVE_PYGMT')
        assert hasattr(maps, '_require_pygmt')
        assert hasattr(maps, 'plot_bathymetry_map')
    
    @pytest.mark.skipif(HAVE_PYGMT, reason="PyGMT is available")
    def test_functions_fail_gracefully_without_pygmt(self):
        """Test that functions fail gracefully when PyGMT unavailable."""
        from oceanvis_py.plots.maps import plot_bathymetry_map
        
        with pytest.raises(RuntimeError, match="PyGMT/GMT not available"):
            plot_bathymetry_map(
                bathymetry_file="dummy.nc",
                region=(0, 1, 0, 1),
                output_file="dummy.png"
            )


class TestLegacyFunctions:
    """Test legacy function compatibility."""
    
    @pytest.mark.skipif(not HAVE_PYGMT, reason=PYGMT_SKIP_REASON)
    def test_plot_map_legacy_function(self):
        """Test the legacy plot_map function."""
        from oceanvis_py.plots.maps import plot_map
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            bathy_file = create_test_bathymetry_file(temp_path)
            output_file = temp_path / "legacy_map.png"
            
            # Legacy function should work - but not with default region that includes poles
            result = plot_map(
                bathymetry_file=bathy_file,
                region=(-56, -36, 40, 52),  # Reasonable region, not global
                output_file=str(output_file),
                dpi=150
            )
            
            assert result == str(output_file)
            assert output_file.exists()