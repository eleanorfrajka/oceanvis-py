# Getting Started: Setup and Installation

This guide walks you through installing and setting up oceanvis-py for oceanographic data visualization. Since the package is in active development, you'll install directly from the GitHub repository.

---

## Step 1: Installation Options

Currently, oceanvis-py is in development and not yet published to PyPI. You'll need to install from the GitHub repository.

### Install from GitHub Repository

#### a. Clone the repository
From a **terminal**:
```bash
git clone https://github.com/eleanorfrajka/oceanvis-py
cd oceanvis-py
```

Or using **GitHub Desktop**:
1. Navigate to [https://github.com/eleanorfrajka/oceanvis-py](https://github.com/eleanorfrajka/oceanvis-py)
2. Click the green `<> Code` button
3. Choose **Open with GitHub Desktop**

#### b. Contributing to the project
See the contributing guide for full instructions on forking, branching, and submitting pull requests.

---

## Step 2: Set Up a Python Environment

We recommend using a clean Python environment. Oceanvis-py requires Python 3.9+ and has specific dependencies for scientific computing.

### Option A: Using conda/mamba (Recommended)
Conda handles the complex scientific dependencies more reliably:
```bash
conda create -n oceanvis python=3.11
conda activate oceanvis
conda install -c conda-forge matplotlib cartopy xarray netcdf4 cmocean

# For development
pip install -r requirements-dev.txt
```

### Option B: Using venv and pip
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

> ⚠️ **Note**: Some dependencies (especially cartopy) can be challenging to install with pip alone. Conda is recommended for easier setup.

---

## Step 3: Install the Package

Install oceanvis-py in editable mode so you have access to the latest features:
```bash
pip install -e .
```
This installs oceanvis-py in *editable* mode, so any updates to the source code are immediately available when you import the package.

---

## Step 4: Verify Installation

### Quick Test
Try importing the package:
```python
import oceanvis_py
print(oceanvis_py.__version__)
```

### Run the Test Suite (Development)
For development installations, run the full test suite:
```bash
pytest
```
This validates that all oceanographic data handling and plotting functions work correctly.

### Try an Example
Load and plot some example data (once example data is available):
```python
from oceanvis_py.plots import plot_section
from oceanvis_py.core import data_loader

# Example with test data (when implemented)
# data = data_loader.load_example_data('ctd_section')
# fig, ax = plot_section(data, variable='temperature')
```

---

## Troubleshooting

### Common Installation Issues

**Cartopy installation fails:**
```bash
# Use conda instead of pip for geospatial dependencies
conda install -c conda-forge cartopy
```

**PyGMT backend issues:**
```bash
# Optional - install GMT and PyGMT for additional backend
conda install -c conda-forge gmt pygmt
```

**Missing test data:**
```bash
# Download example datasets (development only)
python -c "from oceanvis_py.core.data_loader import download_test_data; download_test_data()"
```

### VSCode Setup
For the best development experience:
1. Install the Python extension
2. Select your oceanvis environment as the interpreter
3. Enable pytest for test discovery
4. Install Jupyter extension for notebook examples


---

## Next Steps

### For Users
- Explore the [example notebooks](../notebooks/) for common oceanographic plotting workflows
- Check out the [API documentation](oceanvis_py.rst) for detailed function references
- Read the [plotting guide](tutorials/plotting_guide.md) for best practices

### For Contributors
- Review the [contributing guidelines](../CONTRIBUTING.md)
- Read the [development workflow](development.md) documentation
- Check the [style guide](style_guide.md) for coding standards

---

## You're Ready to Visualize!

Oceanvis-py is now installed and ready for creating publication-quality oceanographic plots. Start with the example notebooks to see the package in action, or dive into the API documentation for detailed function references.
