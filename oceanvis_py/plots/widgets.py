"""
Interactive widgets for colormap exploration and nonlinear colorbar tuning.

Uses ipywidgets to create interactive interfaces for exploring different
colormaps and colorbar normalization strategies for oceanographic data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import ipywidgets as widgets
from IPython.display import display, clear_output
import xarray as xr
from typing import Union
from pathlib import Path

try:
    import cmocean

    CMOCEAN_AVAILABLE = True

    # Available cmocean colormaps for oceanographic data
    CMOCEAN_COLORMAPS = {
        "thermal": cmocean.cm.thermal,
        "haline": cmocean.cm.haline,
        "balance": cmocean.cm.balance,
        "dense": cmocean.cm.dense,
        "deep": cmocean.cm.deep,
        "turbid": cmocean.cm.turbid,
        "speed": cmocean.cm.speed,
        "amp": cmocean.cm.amp,
        "tempo": cmocean.cm.tempo,
        "rain": cmocean.cm.rain,
    }
except ImportError:
    CMOCEAN_AVAILABLE = False
    CMOCEAN_COLORMAPS = {}

# Import custom colormaps
try:
    from ..core.custom_colormaps import CUSTOM_COLORMAPS

    CUSTOM_AVAILABLE = True
except ImportError:
    CUSTOM_AVAILABLE = False
    CUSTOM_COLORMAPS = {}

# Standard matplotlib colormaps good for oceanographic data
STANDARD_COLORMAPS = {
    "RdYlBu_r": plt.cm.RdYlBu_r,
    "viridis": plt.cm.viridis,
    "plasma": plt.cm.plasma,
    "cividis": plt.cm.cividis,
    "RdBu_r": plt.cm.RdBu_r,
    "coolwarm": plt.cm.coolwarm,
    "seismic": plt.cm.seismic,
    "Spectral_r": plt.cm.Spectral_r,
}


def round_to_human_readable(value: float, precision: int = 2) -> float:
    """
    Round a value to be human-readable with specified precision.

    Parameters
    ----------
    value : float
        Value to round
    precision : int, default 2
        Maximum decimal places

    Returns
    -------
    float
        Rounded value
    """
    # Round to specified precision
    rounded = round(value, precision)

    # Remove trailing zeros after decimal point
    if rounded == int(rounded):
        return int(rounded)

    return rounded


def determine_optimal_precision(data_range: float, n_levels: int) -> int:
    """
    Determine optimal precision for level boundaries based on data range.

    Parameters
    ----------
    data_range : float
        Range of the data (max - min)
    n_levels : int
        Number of levels desired

    Returns
    -------
    int
        Optimal decimal precision
    """
    # Calculate the typical step size between levels
    typical_step = data_range / n_levels

    # Determine precision needed to represent this step size
    if typical_step >= 1:
        return 0
    elif typical_step >= 0.1:
        return 1
    elif typical_step >= 0.01:
        return 2
    elif typical_step >= 0.005:
        return 3
    elif typical_step >= 0.001:
        return 3
    elif typical_step >= 0.0001:
        return 4
    else:
        return 5


def calculate_histogram_levels(
    data: np.ndarray,
    n_levels: int = 20,
    method: str = "percentile",
    tail_compression: float = 0.8,
) -> np.ndarray:
    """
    Calculate nonlinear colorbar levels based on data distribution.

    Parameters
    ----------
    data : np.ndarray
        Data array (with NaNs removed)
    n_levels : int, default 20
        Number of discrete levels
    method : str, default 'percentile'
        Method for level calculation:
        - 'percentile': Use percentiles for more detail in concentrated areas
        - 'linear': Linear spacing
        - 'histogram': Use histogram bin edges
        - 'symmetric_zero': Symmetric levels around zero (for anomalies/velocity)
    tail_compression : float, default 0.8
        Factor for compressing tails (0.5-1.0, higher = more compression)

    Returns
    -------
    np.ndarray
        Array of level boundaries (length n_levels + 1)
    """
    # Remove NaN values
    clean_data = data[~np.isnan(data)]

    if len(clean_data) == 0:
        return np.linspace(0, 1, n_levels + 1)

    data_min, data_max = clean_data.min(), clean_data.max()

    # If all data is constant or range is tiny, always return at least [min, max]
    if np.isclose(data_min, data_max):
        # If n_levels > 1, return [min, max] (or n_levels+1 identical values if desired)
        return np.array([data_min, data_max])

    if method == "linear":
        levels = np.linspace(data_min, data_max, n_levels + 1)

    elif method == "percentile":
        # Use percentiles to get more levels in concentrated areas
        percentiles = np.linspace(0, 100, n_levels + 1)
        levels = np.percentile(clean_data, percentiles)

    elif method == "histogram":
        # Use histogram to find natural breaks in the data
        hist, bin_edges = np.histogram(clean_data, bins=n_levels)
        levels = bin_edges

    elif method == "compressed_percentile":
        # Compress tails to focus on central distribution
        p_low = (1 - tail_compression) * 50
        p_high = 100 - p_low

        # Get percentiles with compressed tails
        tail_percentiles = np.linspace(0, p_low, n_levels // 4)
        middle_percentiles = np.linspace(p_low, p_high, n_levels // 2)
        upper_tail_percentiles = np.linspace(p_high, 100, n_levels // 4 + 1)

        all_percentiles = np.concatenate(
            [
                tail_percentiles,
                middle_percentiles[1:],  # Skip duplicate
                upper_tail_percentiles[1:],  # Skip duplicate
            ]
        )

        levels = np.percentile(clean_data, all_percentiles)

    elif method == "symmetric_zero":
        # Create symmetric levels around zero (perfect for velocity/anomaly data)
        data_abs_max = max(abs(clean_data.min()), abs(clean_data.max()))

        # Always create symmetric levels around zero
        # The key is to ensure we have exactly n_levels discrete bands
        half_levels = n_levels // 2

        if n_levels % 2 == 1:
            # Odd number of levels - zero is the center level
            positive_levels = np.linspace(0, data_abs_max, half_levels + 1)
            negative_levels = -positive_levels[1:][::-1]  # Exclude 0, reverse order
            levels = np.concatenate([negative_levels, positive_levels])
        else:
            # Even number of levels - zero is a boundary between levels
            positive_levels = np.linspace(0, data_abs_max, half_levels + 1)[
                1:
            ]  # Exclude 0
            negative_levels = -positive_levels[::-1]  # Reverse order for negative
            levels = np.concatenate([negative_levels, [0], positive_levels])

    # Determine optimal precision based on data range and number of levels
    data_range = data_max - data_min
    optimal_precision = determine_optimal_precision(data_range, n_levels)

    # Round to human-readable values with adaptive precision
    levels = np.array(
        [round_to_human_readable(level, optimal_precision) for level in levels]
    )

    # Ensure levels are strictly increasing
    levels = np.unique(levels)

    # If we lost levels due to rounding, interpolate to get back to n_levels
    if len(levels) < n_levels + 1:
        levels = np.linspace(levels[0], levels[-1], n_levels + 1)
        levels = np.array(
            [round_to_human_readable(level, optimal_precision) for level in levels]
        )
        levels = np.unique(levels)

    return levels


def rgb_to_hex(rgb):
    """Convert RGB values (0-1 range) to hex color string."""
    rgb_255 = (np.array(rgb) * 255).astype(int)
    return f"#{rgb_255[0]:02x}{rgb_255[1]:02x}{rgb_255[2]:02x}"


def hex_to_rgb(hex_color):
    """Convert hex color string to RGB values (0-1 range)."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def read_cpt_file(cpt_file: Union[str, Path]) -> tuple:
    """
    Read a PyGMT .cpt file and return colormap and normalization.

    Parameters
    ----------
    cpt_file : str or Path
        Path to the .cpt file

    Returns
    -------
    tuple
        (colormap, norm, levels) where:
        - colormap is a matplotlib ListedColormap
        - norm is a matplotlib BoundaryNorm
        - levels is the array of level boundaries

    Examples
    --------
    >>> cmap, norm, levels = read_cpt_file('salinity.cpt')
    >>> plt.pcolormesh(x, y, data, cmap=cmap, norm=norm)
    """
    cpt_file = Path(cpt_file)

    if not cpt_file.exists():
        raise FileNotFoundError(f"CPT file not found: {cpt_file}")

    levels = []
    colors = []

    with open(cpt_file, "r") as f:
        for line in f:
            line = line.strip()

            # Skip comments and special lines
            if (
                line.startswith("#")
                or line.startswith("B")
                or line.startswith("F")
                or line.startswith("N")
            ):
                continue

            if not line:
                continue

            # Parse color interval line: "lower_val #color upper_val #color"
            parts = line.split()
            if len(parts) >= 4:
                lower_val = float(parts[0])
                lower_color = parts[1]
                upper_val = float(parts[2])
                upper_color = parts[3]

                # For the first interval, add the lower boundary and color
                if not levels:
                    levels.append(lower_val)

                # Always add the upper boundary
                levels.append(upper_val)

                # Each interval gets one color (we'll use the color for that interval)
                colors.append(hex_to_rgb(lower_color))

    if not levels or not colors:
        raise ValueError(f"No valid color intervals found in CPT file: {cpt_file}")

    # Create matplotlib colormap and normalization
    levels = np.array(levels)
    n_intervals = len(colors)

    # Create discrete colormap
    discrete_cmap = ListedColormap(colors)

    # Create boundary normalization
    # BoundaryNorm expects n_boundaries for n_intervals colors
    norm = BoundaryNorm(levels, n_intervals, extend="neither")

    return discrete_cmap, norm, levels


def generate_cpt_file(
    levels: np.ndarray,
    colormap,
    output_file: Union[str, Path],
    background_color: str = None,
    foreground_color: str = None,
    nan_color: str = "#ffffff",
) -> None:
    """
    Generate a PyGMT-compatible .cpt (color palette table) file.

    Parameters
    ----------
    levels : np.ndarray
        Level boundaries (length n_levels + 1)
    colormap : matplotlib colormap
        Colormap to sample colors from
    output_file : str or Path
        Output file path for the .cpt file
    background_color : str, optional
        Hex color for background (B). If None, uses first color
    foreground_color : str, optional
        Hex color for foreground (F). If None, uses last color
    nan_color : str, default "#ffffff"
        Hex color for NaN values (N)

    Examples
    --------
    >>> from oceanvis_py.plots.widgets import calculate_histogram_levels, generate_cpt_file
    >>> from oceanvis_py.core.custom_colormaps import CUSTOM_COLORMAPS
    >>>
    >>> # Calculate levels for your data
    >>> levels = calculate_histogram_levels(salinity_data, 19, 'percentile')
    >>>
    >>> # Generate CPT file
    >>> generate_cpt_file(levels, CUSTOM_COLORMAPS['SAL'], 'salinity.cpt')
    """
    output_file = Path(output_file)

    # Calculate colors for each level interval
    n_intervals = len(levels) - 1
    color_positions = np.linspace(0, 1, n_intervals)
    colors = [colormap(pos) for pos in color_positions]

    # Convert to hex
    hex_colors = [rgb_to_hex(color[:3]) for color in colors]  # Only RGB, ignore alpha

    # Set background and foreground colors if not provided
    if background_color is None:
        background_color = hex_colors[0]
    if foreground_color is None:
        foreground_color = hex_colors[-1]

    # Write CPT file
    with open(output_file, "w") as f:
        # Write header comment
        f.write("# PyGMT Color Palette Table generated by oceanvis-py\n")
        f.write(f"# Colormap: {getattr(colormap, 'name', 'custom')}\n")
        f.write(f"# Levels: {len(levels)} boundaries, {n_intervals} intervals\n")
        f.write(f"# Range: {levels[0]} to {levels[-1]}\n")
        f.write("#\n")

        # Write color intervals
        for i in range(n_intervals):
            lower = levels[i]
            upper = levels[i + 1]
            color = hex_colors[i]

            # CPT format: lower_val color upper_val color
            f.write(f"{lower} {color} {upper} {color}\n")

        # Write background, foreground, and NaN colors
        f.write(f"B {background_color}\n")
        f.write(f"F {foreground_color}\n")
        f.write(f"N {nan_color}\n")

    print(f"CPT file saved to: {output_file}")
    print(f"Contains {n_intervals} color intervals from {levels[0]} to {levels[-1]}")


class InteractiveColormapWidget:
    """
    Interactive widget for exploring colormaps and colorbar normalizations.

    Allows real-time experimentation with different colormaps, discrete levels,
    and nonlinear normalizations based on data distribution.
    """

    def __init__(
        self,
        data: Union[xr.DataArray, np.ndarray],
        variable_name: str = "variable",
        initial_colormap: str = "viridis",
    ):
        """
        Initialize interactive colormap widget.

        Parameters
        ----------
        data : xarray.DataArray or numpy.ndarray
            Data to visualize
        variable_name : str, default "variable"
            Name of the variable for labels
        initial_colormap : str, default "viridis"
            Initial colormap to display
        """
        # Handle both xarray and numpy data
        if hasattr(data, "values"):
            self.data = data.values.flatten()
        else:
            self.data = np.asarray(data).flatten()

        self.variable_name = variable_name
        self.clean_data = self.data[~np.isnan(self.data)]

        # Combine available colormaps - prioritize custom colormaps
        self.available_colormaps = {**STANDARD_COLORMAPS}
        if CUSTOM_AVAILABLE:
            self.available_colormaps.update(CUSTOM_COLORMAPS)
        if CMOCEAN_AVAILABLE:
            self.available_colormaps.update(CMOCEAN_COLORMAPS)

        # Initialize widgets
        self._create_widgets(initial_colormap)

        # Output widget for plots
        self.output = widgets.Output()

    def _create_widgets(self, initial_colormap: str):
        """Create the interactive widgets."""
        # Colormap selection
        colormap_options = list(self.available_colormaps.keys())
        if initial_colormap not in colormap_options:
            initial_colormap = colormap_options[0]

        self.colormap_dropdown = widgets.Dropdown(
            options=colormap_options,
            value=initial_colormap,
            description="Colormap:",
            style={"description_width": "initial"},
        )

        # Number of discrete levels
        self.n_levels_slider = widgets.IntSlider(
            value=20,
            min=5,
            max=30,
            step=1,
            description="# Levels:",
            style={"description_width": "initial"},
        )

        # Normalization method
        self.norm_method_dropdown = widgets.Dropdown(
            options=[
                ("Linear", "linear"),
                ("Percentile", "percentile"),
                ("Histogram", "histogram"),
                ("Compressed Percentile", "compressed_percentile"),
                ("Symmetric Around Zero", "symmetric_zero"),
            ],
            value="percentile",
            description="Method:",
            style={"description_width": "initial"},
        )

        # Tail compression (only active for compressed_percentile)
        self.tail_compression_slider = widgets.FloatSlider(
            value=0.8,
            min=0.5,
            max=1.0,
            step=0.05,
            description="Tail Compress:",
            style={"description_width": "initial"},
        )

        # Colorbar extend option
        self.extend_dropdown = widgets.Dropdown(
            options=[
                ("Neither", "neither"),
                ("Both", "both"),
                ("Min", "min"),
                ("Max", "max"),
            ],
            value="neither",
            description="Extend:",
            style={"description_width": "initial"},
        )

        # Generate code button
        self.generate_code_button = widgets.Button(
            description="Generate Code",
            button_style="success",
            tooltip="Generate copy-paste code for current settings",
        )

        # Generate CPT button
        self.generate_cpt_button = widgets.Button(
            description="Generate CPT",
            button_style="info",
            tooltip="Generate PyGMT .cpt file for current settings",
        )

        # Attach observers
        self.colormap_dropdown.observe(self._on_change, names="value")
        self.n_levels_slider.observe(self._on_change, names="value")
        self.norm_method_dropdown.observe(self._on_change, names="value")
        self.tail_compression_slider.observe(self._on_change, names="value")
        self.extend_dropdown.observe(self._on_change, names="value")
        self.generate_code_button.on_click(self._generate_code)
        self.generate_cpt_button.on_click(self._generate_cpt)

    def _on_change(self, change):
        """Handle widget value changes."""
        self.update_plot()

    def update_plot(self):
        """Update the colorbar visualization."""
        with self.output:
            clear_output(wait=True)

            # Get current settings
            colormap_name = self.colormap_dropdown.value
            n_levels = self.n_levels_slider.value
            method = self.norm_method_dropdown.value
            tail_compression = self.tail_compression_slider.value
            extend = self.extend_dropdown.value

            # Calculate levels
            if method == "compressed_percentile":
                levels = calculate_histogram_levels(
                    self.clean_data, n_levels, method, tail_compression
                )
            else:
                levels = calculate_histogram_levels(self.clean_data, n_levels, method)

            # Get colormap
            cmap = self.available_colormaps[colormap_name]

            # Create discrete colormap and normalization
            # Account for extension bins when calculating ncolors
            if extend == "both":
                ncolors = n_levels + 2
            elif extend in ["min", "max"]:
                ncolors = n_levels + 1
            else:  # 'neither'
                ncolors = n_levels

            discrete_cmap = ListedColormap(cmap(np.linspace(0, 1, ncolors)))
            norm = BoundaryNorm(levels, ncolors, extend=extend)

            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            # Plot 1: Data histogram
            ax1.hist(
                self.clean_data,
                bins=50,
                alpha=0.7,
                density=True,
                color="skyblue",
                edgecolor="black",
            )
            ax1.axvline(
                self.clean_data.mean(),
                color="red",
                linestyle="--",
                label=f"Mean: {self.clean_data.mean():.3f}",
            )
            ax1.axvline(
                np.median(self.clean_data),
                color="orange",
                linestyle="--",
                label=f"Median: {np.median(self.clean_data):.3f}",
            )
            ax1.set_xlabel(f"{self.variable_name}")
            ax1.set_ylabel("Density")
            ax1.set_title("Data Distribution")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Colorbar levels
            level_positions = (levels[:-1] + levels[1:]) / 2
            colors = discrete_cmap(np.linspace(0, 1, len(level_positions)))

            ax2.bar(
                range(len(level_positions)),
                np.ones(len(level_positions)),
                color=colors,
                width=0.8,
                edgecolor="black",
                linewidth=0.5,
            )
            ax2.set_xlabel("Level Index")
            ax2.set_ylabel("Color")
            ax2.set_title(f"Discrete Colormap ({n_levels} levels)")

            # Add level values as x-tick labels (every few levels to avoid crowding)
            step = max(1, len(levels) // 10)
            tick_indices = range(0, len(level_positions), step)
            ax2.set_xticks(tick_indices)
            ax2.set_xticklabels(
                [f"{level_positions[i]:.2f}" for i in tick_indices], rotation=45
            )

            # Plot 3: Colorbar preview - make discrete levels clearly visible
            # Create data that shows discrete bands more clearly
            colorbar_height = len(levels) - 1
            colorbar_data = np.zeros((colorbar_height, 10))

            # Fill each band with the middle value of that level
            for i in range(colorbar_height):
                band_value = (levels[i] + levels[i + 1]) / 2
                colorbar_data[i, :] = band_value

            # Flip so it displays top-to-bottom correctly
            colorbar_data = np.flipud(colorbar_data)

            # Create extent for proper positioning
            extent = [0, 1, levels[0], levels[-1]]

            im = ax3.imshow(
                colorbar_data,
                aspect="auto",
                cmap=discrete_cmap,
                norm=norm,
                extent=extent,
                interpolation="nearest",
            )
            ax3.set_xlim(0, 1)
            ax3.set_ylabel(f"{self.variable_name}")
            ax3.set_xlabel("")
            ax3.set_title(f"Colorbar Preview ({len(levels)-1} bands)")
            ax3.set_xticks([])

            # Add level boundaries as horizontal lines to make bands even clearer
            for level in levels:
                ax3.axhline(level, color="white", linewidth=1.0, alpha=0.8)

            plt.tight_layout()
            plt.show()

            # Display level information
            print(
                f"\nLevel boundaries ({len(levels)} boundaries for {n_levels} levels):"
            )
            print(f"Min: {levels[0]}, Max: {levels[-1]}")
            print(f"Levels: {levels}")

    def _generate_code(self, button):
        """Generate copy-paste code for current settings."""
        with self.output:
            colormap_name = self.colormap_dropdown.value
            n_levels = self.n_levels_slider.value
            method = self.norm_method_dropdown.value
            tail_compression = self.tail_compression_slider.value
            extend = self.extend_dropdown.value

            # Calculate current levels
            if method == "compressed_percentile":
                levels = calculate_histogram_levels(
                    self.clean_data, n_levels, method, tail_compression
                )
            else:
                levels = calculate_histogram_levels(self.clean_data, n_levels, method)

            print("\n" + "=" * 60)
            print("COPY-PASTE CODE FOR CURRENT SETTINGS:")
            print("=" * 60)
            # Determine import statements and colormap access
            imports = [
                "import numpy as np",
                "import matplotlib.pyplot as plt",
                "from matplotlib.colors import ListedColormap, BoundaryNorm",
            ]

            if colormap_name in CMOCEAN_COLORMAPS:
                imports.append("import cmocean")
                cmap_access = f"cmap = cmocean.cm.{colormap_name}"
            elif colormap_name in CUSTOM_COLORMAPS:
                imports.append(
                    "from oceanvis_py.core.custom_colormaps import CUSTOM_COLORMAPS"
                )
                cmap_access = f"cmap = CUSTOM_COLORMAPS['{colormap_name}']"
            else:
                cmap_access = f"cmap = plt.cm.{colormap_name}"

            import_str = "\n".join(imports)

            print(
                f"""
{import_str}

# Define custom levels
levels = np.array({levels.tolist()})

# Create discrete colormap
{cmap_access}
discrete_cmap = ListedColormap(cmap(np.linspace(0, 1, {n_levels})))
norm = BoundaryNorm(levels, {n_levels}, extend='{extend}')

# Use in plotting
# plt.pcolormesh(x, y, data, cmap=discrete_cmap, norm=norm)
# OR
# from oceanvis_py.plots import plot_section
# fig, ax = plot_section(dataset, variable='your_variable',
#                       colormap=discrete_cmap, norm=norm)
            """
            )
            print("=" * 60)

    def _generate_cpt(self, button):
        """Generate PyGMT .cpt file for current settings."""
        with self.output:
            colormap_name = self.colormap_dropdown.value
            n_levels = self.n_levels_slider.value
            method = self.norm_method_dropdown.value
            tail_compression = self.tail_compression_slider.value

            # Calculate current levels
            if method == "compressed_percentile":
                levels = calculate_histogram_levels(
                    self.clean_data, n_levels, method, tail_compression
                )
            else:
                levels = calculate_histogram_levels(self.clean_data, n_levels, method)

            # Get colormap
            cmap = self.available_colormaps[colormap_name]

            # Generate clean filename (remove problematic characters)
            import re

            clean_var_name = re.sub(
                r"[^\w\-_]", "", self.variable_name.replace(" ", "_")
            ).lower()
            filename = f"{clean_var_name}_{colormap_name}_{n_levels}levels.cpt"

            # Create output directory if it doesn't exist
            # Use relative path from module location
            output_dir = Path(__file__).parent.parent / "config" / "saved_colormaps"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Full file path
            full_path = output_dir / filename

            # Generate CPT file
            try:
                generate_cpt_file(levels, cmap, full_path)
                print("\n✅ CPT file generated successfully!")
                print(f"📁 File: {full_path}")
                print(f"🎨 Colormap: {colormap_name}")
                print(
                    f"📊 {len(levels)-1} color intervals from {levels[0]} to {levels[-1]}"
                )
                print("\nTo use in PyGMT:")
                print(f"fig.grdimage(grid, cmap='{full_path}')")
            except Exception as e:
                print(f"❌ Error generating CPT file: {e}")

    def display(self):
        """Display the complete interactive widget."""
        # Create layout
        controls_left = widgets.VBox(
            [self.colormap_dropdown, self.n_levels_slider, self.norm_method_dropdown]
        )

        controls_right = widgets.VBox(
            [
                self.tail_compression_slider,
                self.extend_dropdown,
                widgets.HBox([self.generate_code_button, self.generate_cpt_button]),
            ]
        )

        controls = widgets.HBox([controls_left, controls_right])

        full_widget = widgets.VBox(
            [
                widgets.HTML("<h3>Interactive Colormap Explorer</h3>"),
                controls,
                self.output,
            ]
        )

        display(full_widget)

        # Initial plot
        self.update_plot()

        return self


# Backwards compatibility
class ColorMapWidget(InteractiveColormapWidget):
    """Alias for InteractiveColormapWidget for backwards compatibility."""

    pass


def create_colormap_widget(
    data: Union[xr.DataArray, np.ndarray], variable_name: str = "variable"
) -> InteractiveColormapWidget:
    """
    Create an interactive colormap widget for exploring colorbars.

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        Data to visualize
    variable_name : str, default "variable"
        Name of the variable for labels

    Returns
    -------
    InteractiveColormapWidget
        Widget instance

    Examples
    --------
    >>> import xarray as xr
    >>> from oceanvis_py.plots.widgets import create_colormap_widget
    >>>
    >>> # Load data
    >>> ds = xr.open_dataset('data.nc')
    >>>
    >>> # Create widget
    >>> widget = create_colormap_widget(ds.salinity, 'Salinity')
    >>> widget.display()
    """
    return InteractiveColormapWidget(data, variable_name)
