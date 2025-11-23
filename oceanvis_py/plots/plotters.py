from pandas import DataFrame
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

# --- Optional PyGMT support (lazy import to avoid hard dependency in tests) ---
HAVE_PYGMT = False
try:
    import importlib

    _pygmt_spec = importlib.util.find_spec("pygmt")
    HAVE_PYGMT = _pygmt_spec is not None
except Exception:
    HAVE_PYGMT = False


def _require_pygmt():
    """Import and return pygmt or raise a clear error if unavailable."""
    if not HAVE_PYGMT:
        raise RuntimeError(
            "PyGMT/GMT not available. Install 'gmt' and 'pygmt' (e.g., via conda-forge) "
            "to use PyGMT plotting."
        )
    import pygmt  # local import to avoid import-time failure

    return pygmt


# Define the POLAR colormap colors (flipped order)
polar_colors = (
    np.array(
        [
            [5, 48, 97],
            [33, 102, 172],
            [67, 147, 195],
            [146, 197, 222],
            [209, 229, 240],
            [255, 255, 240],
            [253, 219, 199],
            [244, 165, 130],
            [214, 96, 77],
            [178, 24, 43],
            [103, 0, 31],
        ]
    )
    / 255.0
)

cmap_polar = ListedColormap(polar_colors, name="polar")


def create_bathy_poly(ds_towyo, pmax=3000):
    # Replace first NaN in waterdepth with the first non-NaN value
    distance = np.concatenate(
        [
            [0],
            [0],
            ds_towyo.distance.values,
            [ds_towyo.distance.values[-1]],
            [ds_towyo.distance.values[-1]],
            [0],
        ]
    )
    print(f"Max water depth is {np.max(ds_towyo.waterdepth.values)} m")
    waterdepth = np.concatenate(
        [
            [pmax],
            [ds_towyo.waterdepth.values[0]],
            ds_towyo.waterdepth.values,
            [ds_towyo.waterdepth.values[-1]],
            [pmax],
            [pmax],
        ]
    )
    bathymetry_df = pd.DataFrame({"distance": distance, "waterdepth": waterdepth})
    return bathymetry_df


def make_pygmt_grid(ds, x_var_name, y_var_name, z_data_matrix):
    pygmt = _require_pygmt()
    """
    Create a PyGMT-compatible xarray.DataArray grid from an xarray.Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing coordinate variables.
    x_var_name : str
        Name of the longitude (x) variable.
    y_var_name : str
        Name of the latitude (y) variable.
    z_data_matrix : str or np.ndarray
        Name of the data variable in ds, or a 2D numpy array of data.

    Returns
    -------
    grid : xarray.DataArray
        DataArray with dims (y, x) and coordinates (y, x) for use in PyGMT.
    """
    x = ds[x_var_name].values
    y = ds[y_var_name].values
    if isinstance(z_data_matrix, str):
        z = ds[z_data_matrix].values
    else:
        z = z_data_matrix
    grid = xr.DataArray(
        z, coords={y_var_name: y, x_var_name: x}, dims=[y_var_name, x_var_name]
    )
    return grid


def generate_cpt_file_simple(boundaries, cmap_name, output_cpt):
    """
    Generate a .cpt file from a matplotlib colormap and value boundaries.

    Note: This is a simpler version. For advanced features, use
    oceanvis_py.plots.widgets.generate_cpt_file instead.

    Parameters
    ----------
    boundaries : list of float
        The contour boundaries for the color map.
    cmap_name : str
        Name of the matplotlib colormap.
    output_cpt : str
        Filename for the output .cpt file.
    """
    cmap, norm = cm.get_cmap(cmap_name), plt.Normalize(boundaries[0], boundaries[-1])

    with open(output_cpt, "w") as f:
        for i in range(len(boundaries) - 1):
            norm_val = norm((boundaries[i] + boundaries[i + 1]) / 2)
            rgb = cmap(norm_val)[:3]
            color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            f.write(f"{boundaries[i]} {color} {boundaries[i + 1]} {color}\n")
            if i == 0:
                color1 = color
            if i == len(boundaries) - 2:
                color2 = color
        f.write(f"B {color1}\nF {color2}\nN #ffffff\n")


def plot_section_pygmt(
    ds_towyo,
    bathymetry_df,
    var_name,
    dist_var,
    pres_var,
    boundaries,
    cmap_name,
    cpt_file,
    output_file,
    a_int=None,
    dens_var=None,
    dens_col="black",
    dens_dist="distance_shift",
):
    pygmt = _require_pygmt()
    """
    Plot a section figure using PyGMT for a specified variable.

    Parameters
    ----------
    ds_towyo : xarray.Dataset
        Dataset containing the towyo data.
    bathymetry_df : pandas.DataFrame
        DataFrame containing bathymetry distance and depth.
    var_name : str
        Variable name in ds_towyo to plot (e.g., 'CT' or 'SA').
    dist_var : str
        Name of the distance variable in ds_towyo (e.g., 'distance_shift').
    pres_var : str
        Name of the pressure/depth variable in ds_towyo (e.g., 'depth_ctd').
    boundaries : list of float
        Contour boundaries for the variable.
    cmap_name : str
        Matplotlib colormap name (e.g., 'RdYlBu_r').
    cpt_file : str
        Output .cpt file name.
    output_file : str
        Output figure file name.
    """
    if dens_var is not None:
        dens_int = 0.02
        sig_boundaries = [
            36.84,
            36.86,
            36.88,
            36.90,
            36.91,
            36.92,
            36.93,
            36.94,
            36.95,
            36.96,
            36.97,
            36.98,
            37,
        ]
        cmap_sig = "Purples"
        cpt_sig = "sig2.cpt"
        generate_cpt_file_simple(sig_boundaries, cmap_sig, cpt_sig)
        grid_min, grid_max = 34.1, 37

    if a_int is None:
        a_int = boundaries[1] - boundaries[0]

    generate_cpt_file_simple(boundaries, cmap_name, cpt_file)

    plot_name = ds_towyo[var_name].attrs.get("plot_name", var_name)
    plot_units = ds_towyo[var_name].attrs.get("plot_units", "")
    cruise_label = ds_towyo.attrs.get("short_cruise_name", "")

    grid = make_pygmt_grid(ds_towyo, dist_var, pres_var, var_name)
    fig = pygmt.Figure()

    with pygmt.config(
        FONT_ANNOT_PRIMARY="11p",
        FONT_LABEL="12p",
        FONT_TITLE="12p",
        FONT_ANNOT_SECONDARY="11p",
        MAP_FRAME_PEN="thin,darkgrey",
        MAP_FRAME_AXES="WS",
        MAP_FRAME_TYPE="plain",
    ):
        dmin = 1000
        distmax = bathymetry_df["distance"].max() - 0.5
        region = [
            bathymetry_df["distance"].min(),
            distmax,
            dmin,
            bathymetry_df["waterdepth"].max(),
        ]
        projection = "X8c/-6c"

        fig.basemap(
            region=region,
            projection=projection,
            frame=["xaf+lDistance (km)", "yaf+lDepth (m)", f"+t{cruise_label}"],
        )

        fig.grdimage(grid=grid, region=region, cmap=cpt_file)

        if dens_var is not None:
            grid = make_pygmt_grid(ds_towyo, dens_dist, pres_var, dens_var)
            fig.grdcontour(
                grid=grid.where(~np.isnan(grid)),
                region=region,
                levels=0.02,
                annotation="0.04+f11p,Helvetica," + dens_col,
                pen="0.5p," + dens_col + ",2_1:1p",
                limit=[grid_min, grid_max],
                label_placement="d3c",
            )

        fig.plot(
            x=bathymetry_df["distance"],
            y=bathymetry_df["waterdepth"] + 200,
            close=True,
            pen="0.7p,black",
        )

        fig.text(
            x=0,
            y=3000,
            text=plot_name,
            font="12p,Helvetica,black",
            justify="LB",
            offset="0.15c/0.15c",
            no_clip=True,
        )

        with pygmt.config(
            FONT_ANNOT_PRIMARY="16p", FONT_LABEL="16p", FONT_ANNOT_SECONDARY="16p"
        ):
            fig.colorbar(
                cmap=cpt_file,
                position="JMR+w5c/.5c+o.4c/.2c",
                frame=[f"a{a_int}", f"y+l{plot_units}"],
            )

        fig.savefig(output_file, dpi=300, transparent=True)
        fig.show()


def nonlinear_colormap(cmap_name="YlGnBu", boundaries=None, ticks=None, plotflag=True):
    """
    Create a nonlinear discrete colormap and colorbar.

    Parameters:
        cmap_name (str): Name of the matplotlib colormap.
        boundaries (list or np.ndarray): Boundaries for discrete colors.
        ticks (list or np.ndarray): Locations for colorbar ticks.

    Returns:
        cmap, norm, cb: The colormap, normalization, and colorbar object.
    """
    # Use default boundaries if not provided
    if boundaries is None:
        boundaries = np.linspace(33, 35, 6)  # Example for salinity

    cmap = cm.get_cmap(cmap_name, len(boundaries) - 1)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    if plotflag:
        fig, ax = plt.subplots(figsize=(1.5, 6))
        cb = plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            boundaries=boundaries,
            ticks=ticks if ticks is not None else boundaries,
            spacing="proportional",
        )
        cb.set_label("Salinity")
        plt.show()
    return cmap, norm


def plot_scalar(
    ax, merged_towyo, x_var, y_var, data_var, cmap, clim, title, cbar_label
):
    pc = ax.pcolormesh(
        merged_towyo[x_var],
        merged_towyo[y_var],
        merged_towyo[data_var],
        shading="auto",
        cmap=cmap,
    )
    pc.set_clim(*clim)
    cbar = plt.colorbar(pc, ax=ax, label=cbar_label)
    ax.set_title(title)
    ax.set_ylabel(y_var.capitalize())
    ax.invert_yaxis()
    return pc, cbar


def add_sigma2_contours(ax, merged_towyo, x_var, y_var, sigma2, levels):
    cs = ax.contour(
        merged_towyo[x_var],
        merged_towyo[y_var],
        sigma2,
        levels=levels,
        colors="w",
        linewidths=1,
        linestyles="dashed",
    )
    ax.clabel(cs, fmt="%1.2f", fontsize=9)
    return cs


def plot_towyo_panels(
    merged_towyo,
    x_var="castnum",
    cruise_name="MSM121",
    sigma2_levels=np.arange(36.8, 37.0, 0.02),
    slim=(35.04, 35.08),
):
    sigma2 = merged_towyo["sigma2"]
    has_uvelo = (
        "u_velocity" in merged_towyo.data_vars
        and "v_velocity" in merged_towyo.data_vars
    )

    if x_var == "distance":
        # Use distance_ladcp if available, otherwise fallback to distance
        x2_var = "distance_ladcp" if "distance_ladcp" in merged_towyo else "distance"
    elif x_var == "castnum":
        x2_var = "profile_number"

    if has_uvelo:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        # Temperature
        pc1, cbar1 = plot_scalar(
            axs[0, 0],
            merged_towyo,
            x_var,
            "pressure",
            "temperature",
            cmap="RdYlBu_r",
            clim=(2.8, 3.6),
            title=f"Temperature ({cruise_name} towyo)",
            cbar_label="Cons. Temp. (°C)",
        )
        cs1 = add_sigma2_contours(
            axs[0, 0], merged_towyo, x_var, "pressure", sigma2, sigma2_levels
        )

        # Salinity
        pc2, cbar2 = plot_scalar(
            axs[0, 1],
            merged_towyo,
            x_var,
            "pressure",
            "SA",
            cmap="YlGnBu",
            clim=slim,
            title=f"Salinity ({cruise_name} towyo)",
            cbar_label="Abs. Sal.",
        )
        cs2 = add_sigma2_contours(
            axs[0, 1], merged_towyo, x_var, "pressure", sigma2, sigma2_levels
        )

        # U velocity
        pc3, cbar3 = plot_scalar(
            axs[1, 0],
            merged_towyo,
            x2_var,
            "pressure",
            "u_velocity",
            cmap="RdBu_r",
            clim=(-0.4, 0.4),
            title="LADCP Eastward Velocity",
            cbar_label="U velocity (m/s)",
        )
        cs3 = add_sigma2_contours(
            axs[1, 0], merged_towyo, x_var, "pressure", sigma2, sigma2_levels
        )
        axs[1, 0].set_xlabel("Profile Number")
        axs[1, 0].set_ylabel("Depth (m)")

        # V velocity
        pc4, cbar4 = plot_scalar(
            axs[1, 1],
            merged_towyo,
            x2_var,
            "pressure",
            "v_velocity",
            cmap="RdBu_r",
            clim=(-0.4, 0.4),
            title="LADCP Northward Velocity",
            cbar_label="V velocity (m/s)",
        )
        cs4 = add_sigma2_contours(
            axs[1, 1], merged_towyo, x_var, "pressure", sigma2, sigma2_levels
        )
        axs[1, 1].set_xlabel("Profile Number")
        axs[1, 1].set_ylabel("Depth (m)")
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
        # Temperature
        pc1, cbar1 = plot_scalar(
            axs[0],
            merged_towyo,
            x_var,
            "pressure",
            "temperature",
            cmap="RdYlBu_r",
            clim=(2.8, 3.6),
            title=f"Temperature ({cruise_name} towyo)",
            cbar_label="Cons. Temp. (°C)",
        )
        cs1 = add_sigma2_contours(
            axs[0], merged_towyo, x_var, "pressure", sigma2, sigma2_levels
        )

        # Salinity
        pc2, cbar2 = plot_scalar(
            axs[1],
            merged_towyo,
            x_var,
            "pressure",
            "SA",
            cmap="RdYlBu_r",
            clim=slim,
            title=f"Salinity ({cruise_name} towyo)",
            cbar_label="Abs. Sal.",
        )
        cs2 = add_sigma2_contours(
            axs[1], merged_towyo, x_var, "pressure", sigma2, sigma2_levels
        )

    # Set x-label
    xlabel = "Distance (km)" if x_var == "distance" else "Cast Number"
    for ax in axs.flat:
        ax.set_xlabel(xlabel)

    # Invert y-axis for pressure
    for ax in axs.flat:
        # ax.invert_yaxis()
        ax.set_ylim(merged_towyo["pressure"].max(), merged_towyo["pressure"].min())

    plt.tight_layout()
    return fig, axs


#    plt.show()


def check_castnum_assigned(ds):
    fig, axs = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
    )

    # Upper plot: up_dn values vs time
    axs[0].plot(ds.time, ds.castnum, ".", markersize=2)
    axs[0].set_ylabel("castnum")
    axs[0].set_title("castnum values vs Time")
    axs[0].grid(True)

    # Lower plot: Pressure vs Time colored by up_dn
    valid_up_dn = ds.castnum.values[~np.isnan(ds.castnum.values)]
    if len(valid_up_dn) > 0:
        unique_vals = np.unique(valid_up_dn)
        n_colors = len(unique_vals)
        cmap = plt.get_cmap("tab10", n_colors)
        sc = axs[1].scatter(ds.time, ds.pressure, c=ds.castnum, cmap=cmap, s=2)
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Pressure (dbar)")
        axs[1].set_title("Pressure vs Time colored by up_dn")
        axs[1].invert_yaxis()
    else:
        print("No valid up_dn values to plot")

    plt.tight_layout()
    plt.show()


def check_cleaned_ctd(ds, vars_to_check=["temperature", "salinity", "conductivity"]):
    fig, axes = plt.subplots(ncols=len(vars_to_check), figsize=(10, 3), sharey=True)
    for ax, var in zip(axes, vars_to_check):
        min_cond = np.percentile(ds[var].values[~np.isnan(ds[var].values)], 2.5)
        max_cond = np.percentile(ds[var].values[~np.isnan(ds[var].values)], 97.5)
        minmin_cond = np.nanmin(ds[var].values)
        maxmax_cond = np.nanmax(ds[var].values)
        print(
            f"Range in {var} from (2.5%, 97.5%) = ({min_cond:.2f}, {max_cond:.2f}), on range ({minmin_cond:.2f}, {maxmax_cond:.2f})"
        )
        ds[var].plot.hist(bins=20, ax=ax, color="blue", alpha=0.7)
        ax.set_title(f"Histogram of {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Frequency")
        ax.legend()
    plt.tight_layout()
    # Conductivity stats for low salinity
    min_sal = np.percentile(ds["salinity"].values, 2.5)
    min_cond_sal = ds["conductivity"].where(ds["salinity"] < min_sal)
    min_cond_sal_mean = min_cond_sal.mean().values
    min_cond_sal_min = min_cond_sal.min().values
    min_cond_sal_max = min_cond_sal.max().values
    print(
        f"Conductivity values for salinity < {min_sal:.2f} are: mean = {min_cond_sal_mean:.2f}, min = {min_cond_sal_min:.2f}, max = {min_cond_sal_max:.2f}"
    )
    plt.show()


def show_variables(data):
    """
    Processes an xarray Dataset or a netCDF file, extracts variable information,
    and returns a styled DataFrame with details about the variables.

    Parameters:
    data (str or xr.Dataset): The input data, either a file path to a netCDF file or an xarray Dataset.

    Returns:
    pandas.io.formats.style.Styler: A styled DataFrame containing the following columns:
        - dims: The dimension of the variable (or "string" if it is a string type).
        - name: The name of the variable.
        - units: The units of the variable (if available).
        - comment: Any additional comments about the variable (if available).
    """

    if isinstance(data, str):
        print("information is based on file: {}".format(data))
        dataset = xr.Dataset(data)
        variables = dataset.variables
    elif isinstance(data, xr.Dataset):
        print("information is based on xarray Dataset")
        variables = data.variables
    else:
        raise TypeError("Input data must be a file path (str) or an xarray Dataset")

    info = {}
    for i, key in enumerate(variables):
        var = variables[key]
        if isinstance(data, str):
            dims = var.dimensions[0] if len(var.dimensions) == 1 else "string"
            units = "" if not hasattr(var, "units") else var.units
            comment = "" if not hasattr(var, "comment") else var.comment
        else:
            dims = var.dims[0] if len(var.dims) == 1 else "string"
            units = var.attrs.get("units", "")
            comment = var.attrs.get("comment", "")

        info[i] = {
            "name": key,
            "dims": dims,
            "units": units,
            "comment": comment,
            "standard_name": var.attrs.get("standard_name", ""),
            "dtype": str(var.dtype) if isinstance(data, str) else str(var.data.dtype),
        }

    vars = DataFrame(info).T

    dim = vars.dims
    dim[dim.str.startswith("str")] = "string"
    vars["dims"] = dim

    vars = (
        vars.sort_values(["dims", "name"])
        .reset_index(drop=True)
        .loc[:, ["dims", "name", "units", "comment", "standard_name", "dtype"]]
        .set_index("name")
        .style
    )

    return vars


def show_attributes(data):
    """
    Processes an xarray Dataset or a netCDF file, extracts attribute information,
    and returns a DataFrame with details about the attributes.

    Parameters:
    data (str or xr.Dataset): The input data, either a file path to a netCDF file or an xarray Dataset.

    Returns:
    pandas.DataFrame: A DataFrame containing the following columns:
        - Attribute: The name of the attribute.
        - Value: The value of the attribute.
    """
    from netCDF4 import Dataset

    if isinstance(data, str):
        print("information is based on file: {}".format(data))
        rootgrp = Dataset(data, "r", format="NETCDF4")
        attributes = rootgrp.ncattrs()
        get_attr = lambda key: getattr(rootgrp, key)
    elif isinstance(data, xr.Dataset):
        print("information is based on xarray Dataset")
        attributes = data.attrs.keys()
        get_attr = lambda key: data.attrs[key]
    else:
        raise TypeError("Input data must be a file path (str) or an xarray Dataset")

    info = {}
    for i, key in enumerate(attributes):
        dtype = type(get_attr(key)).__name__
        info[i] = {"Attribute": key, "Value": get_attr(key), "DType": dtype}

    attrs = DataFrame(info).T

    return attrs
