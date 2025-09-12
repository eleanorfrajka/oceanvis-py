# oceanvis-py

> 🌊 A Python package for visualizing physical oceanographic data from netCDF files — specialized for CTD profiles, LADCP velocity data, and publication-quality scientific plots.

This package provides tools for creating section plots, bathymetry maps, and interactive colormap exploration for physical oceanographic data, with emphasis on finescale variations like internal waves, temperature/salinity interleaving, and density overturns.

---

## 🚀 What's Included

- ✅ **Data Loading**: Robust netCDF file handling with xarray
- 📊 **Section Plots**: Variable vs distance/pressure plots with nonlinear colorbars
- 🗺️ **Bathymetry Maps**: Cartopy-based maps with profile tracks
- 🎨 **Interactive Colormaps**: Jupyter widgets for colormap exploration
- 🔧 **Coordinate Utilities**: Great circle distance calculations
- 📐 **Publication Quality**: Predefined figure sizes for scientific papers
- ⚙️ **Configurable**: YAML-based variable mappings and saved preferences

---

## Package Structure

```
oceanvis_py/
├── core/
│   ├── data_loader.py      # netCDF loading and validation
│   ├── coordinates.py      # lat/lon to distance conversion
│   └── colormaps.py       # colormap utilities and storage
├── backends/
│   ├── matplotlib_backend.py  # matplotlib + cartopy plots
│   └── pygmt_backend.py       # future pygmt implementation
├── plots/
│   ├── sections.py        # section plot functions
│   ├── maps.py           # bathymetry maps
│   └── widgets.py        # interactive colormap widgets
├── config/
│   ├── variable_mappings.yaml
│   ├── matplotlib_style.mplstyle
│   └── saved_colormaps/   # directory for user colormap preferences
└── examples/
    ├── example_section_plot.ipynb
    ├── example_map_plot.ipynb
    └── example_interactive_colormaps.ipynb
```

---

## 🔧 Installation

Install in development mode:

```bash
git clone https://github.com/eleanorfrajka/oceanvis-py.git
cd oceanvis-py
pip install -r requirements-dev.txt
pip install -e .
```

Or using conda:

```bash
conda env create -f environment.yml
conda activate oceanvis-py
pip install -e .
```

---

## 📊 Quick Example

```python
import oceanvis_py as ov

# Load CTD data
data = ov.load_netcdf('cruise_data.nc')

# Create section plot
fig, ax = ov.plot_section(
    data, 
    variable='temperature',
    colormap='thermal',
    bathymetry_file='bathymetry.nc'
)

# Create bathymetry map
fig_map, ax_map = ov.plot_map(
    bathymetry_file='bathymetry.nc',
    profile_locations=data,
    contour_intervals=[1000, 2000, 3000]
)
```

---

## 🎯 Key Features

### Data Types Supported
- **CTD Profiles**: Temperature, salinity, pressure data
- **LADCP Velocity**: u/v velocity components from LADCP instruments  
- **Towyo Data**: High-resolution sections with distance coordinates
- **Bathymetry**: Water depth for mapping and section overlays

### Visualization Capabilities
- **Section Plots**: Emphasize finescale variations with nonlinear colorbars
- **Map Plots**: Publication-quality bathymetry maps with profile tracks
- **Interactive Widgets**: Explore colormaps in Jupyter notebooks
- **Publication Ready**: Predefined figure sizes for scientific papers

### Scientific Focus
- Internal waves and finescale oceanographic features
- Temperature/salinity interleaving and mixing processes
- Density overturns and water mass analysis
- LADCP velocity structure and ocean circulation

---

## 🧪 Testing

To run tests:

```bash
pytest
```

To run linting and type checking:

```bash
ruff check .
black --check .
```

---

## 📚 Documentation

Full documentation and examples are available in the `notebooks/` directory:

- `example_section_plot.ipynb`: Creating section plots from CTD data
- `example_map_plot.ipynb`: Bathymetry mapping with profile tracks  
- `example_interactive_colormaps.ipynb`: Interactive colormap exploration

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Common development tasks:
- Adding new colormap schemes
- Supporting additional netCDF variable naming conventions
- Enhancing publication figure formatting
- Adding new plot types for oceanographic data

---

## 📣 Citation

If you use oceanvis-py in your research, please cite it using the information in [CITATION.cff](CITATION.cff).

---

## 🏗️ Development Status

This package is currently in active development. Core functionality for section plots and data loading is implemented, with ongoing work on:

- Interactive colormap widgets
- Bathymetry mapping features
- PyGMT backend support  
- Extended variable mapping configurations