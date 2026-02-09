import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from datetime import datetime, timedelta
import os
import glob
import re
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


def generate_file_paths(start_year, start_month, start_day, start_hour,
                        end_year, end_month, end_day, end_hour,
                        base_path_main):
    """
    Generate list of file paths for the specified date range

    Parameters
    ----------
    start_year, start_month, start_day, start_hour : int
        Start date and hour for file path
    end_year, end_month, end_day, end_hour : int
        End date and hour for file path
    base_path_main : str
        Base directory path for main dataset files, with filename pattern like "sliced_YYYYMMDD_HH.nc"

    Returns
    -------
    main_paths : list of str
        List of file paths for the specified date range
    
    """
    start_date = datetime(start_year, start_month, start_day, start_hour)
    end_date = datetime(end_year, end_month, end_day, end_hour)

    main_paths = []
    current_date = start_date

    while current_date <= end_date:
        filename = f"sliced_{current_date.strftime('%Y%m%d_%H')}.nc"
        main_paths.append(os.path.join(base_path_main, filename))
        current_date += timedelta(hours=3)

    return main_paths


def load_dataset(main_path, bounds=None):
    """
    Load dataset 

    Parameters
    ----------
    main_path : str
        Path to the main dataset file
    bounds : dict, optional
        Dict with lat and lon keys
    
    Returns
    -------
    xarray.Dataset or None
        The loaded and sliced dataset, or None 

    """
    try:
        ds_main = xr.open_dataset(main_path).squeeze()

        if bounds is None:
            bounds = dict(
                latitude=slice(15.0, 60.0),
                longitude=slice(165, 250))

        return ds_main.sel(**bounds)

    except Exception as e:
        print(f"Error loading dataset:\n  File: {main_path}\n  Error: {e}")
        return None


def parse_detect_nodes_file(filepath):
    """
    Parse DetectNodes output .dat or .txt file

    Parameters
    ----------
    filepath : str
        Path to DetectNodes output file

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: year, month, day, hour, count, lon, lat
     
    """
    data = []
    current_time = None
    columns = None

    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith('#'):
                # Only use the second header line (with 'i', 'j', 'lon', 'lat', ...)
                if columns is None and 'lon' in line and 'lat' in line:  
                    columns = line.lstrip('#').strip().split()
                continue
            # Date/time line (no leading tab/space, 5 columns)
            if not line.startswith('\t') and not line.startswith(' '):
                parts = line.strip().split()
                if len(parts) == 5:
                    year, month, day, count, hour = map(int, parts)
                    current_time = {
                        "year": year,
                        "month": month,
                        "day": day,
                        "count": count,
                        "hour": hour
                    }
                continue
            # Data line (leading tab or space)
            parts = line.strip().split()
            if columns is None or current_time is None:
                continue  # Skip if headers or time info not found
            row = dict(zip(columns, parts))
            # Add time info
            row.update(current_time)
            filtered_row = {
                "year": row["year"],
                "month": row["month"],
                "day": row["day"],
                "count": row["count"],
                "hour":  row["hour"],
                "lon": float(row["lon"]),
                "lat": float(row["lat"])
            }
            data.append(filtered_row)

    if not data:
        return pd.DataFrame(columns=["year", "month", "day", "count", "hour", "lon", "lat"])
    
    df = pd.DataFrame(data)
    for col in ["year", "month", "day", "count", "hour"]: 
        df[col] = df[col].astype(int)
    return df


def make_plots(ds_sliced, save_path=None, directions=None, detect_nodes_df=None):
    """
    Create 4-panel plot of IVT, 925-hPa theta-e gradient, 925-hPa PV, and 925-hPa shearing deformation with DetectNodes

    Parameters
    ----------
    ds_sliced : xarray.Dataset
        ERA5 data 
    save_path : str, optional
        If provided, path to save the plot image
    directions : dict, optional
        Dict with keys 'North', 'South', 'East', 'West' for plot extent
    detect_nodes_df : pandas.DataFrame
        DataFrame containing DetectNodes output
    
    Returns 
    -------
    None

    """
    if ds_sliced is None:
        return

    if directions is None:
        directions = {'North': 52.5, 'South': 30.0, 'East': 240.0, 'West': 215.0}

    # Extract variables
    mslp = ds_sliced['MSLP'] # units: hPa
    ivt = ds_sliced['IVT'] # units: kg/m/s
    thetae_grad = ds_sliced['thetae_grad'] # units: K / 100 km
    deform_925 = ds_sliced['shearing_def_925'] # units: 1/s
    pv_925 = ds_sliced['pv_925'] # units: PVU
    tadv_925 = ds_sliced['tadv_925'] # units:  K/hr
    z_500 = ds_sliced['z_500'] # units: dam
    lat, lon = ds_sliced['latitude'], ds_sliced['longitude']

    current_time = pd.to_datetime(mslp.time.values.item())

    # Match DetectNodes to current time
    df_matched = pd.DataFrame()
    if detect_nodes_df is not None and not detect_nodes_df.empty:
        mask = (
            (detect_nodes_df['year'] == current_time.year) &
            (detect_nodes_df['month'] == current_time.month) &
            (detect_nodes_df['day'] == current_time.day) &
            (detect_nodes_df['hour'] == current_time.hour))
        df_matched = detect_nodes_df[mask]
        print(f"Current plot time: {current_time} | Matched DetectNodes: {len(df_matched)}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12),
                                subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    # IVT subplot
    ivt_levels = [250, 300, 400, 500, 600, 700, 800, 1000, 1200, 1400, 1600, 1800]
    ivt_colors = ['#ffff00', '#ffe400', '#ffc800', '#ffad00', '#ff8200', '#ff5000', '#ff1e00',
                    '#eb0010', '#b8003a', '#850063', '#921972', '#570088']
    ivt_cmap = mcolors.ListedColormap(ivt_colors)
    ivt_norm = mcolors.BoundaryNorm(ivt_levels, ivt_cmap.N)
    
    im1 = axes[0].contourf(lon, lat, ivt, levels=ivt_levels, cmap=ivt_cmap,
                            norm=ivt_norm, transform=ccrs.PlateCarree(), extend='max')
    z_contour = axes[0].contour(lon, lat, z_500, levels=np.arange(500, 620, 6),
                                    colors='black', transform=ccrs.PlateCarree(), linewidths=0.5)
    axes[0].clabel(z_contour, inline=True, fontsize=8, fmt='%d', colors='black')
    axes[0].set_title('IVT (shaded) and 500-hPa Z (contoured; dam)')
    
    divider1 = make_axes_locatable(axes[0])
    cax1 = divider1.append_axes("right", size="6%", pad=0.1, axes_class=plt.Axes)
    cbar1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
    cbar1.set_label('kg/m/s', fontsize=10)
    
    # Theta-E gradient subplot
    im2 = axes[1].contourf(lon, lat, thetae_grad, levels=np.arange(5, 41, 2.5),
                            cmap='RdYlBu_r', transform=ccrs.PlateCarree(), extend='max')
    slp_contour2 = axes[1].contour(lon, lat, mslp, levels=np.arange(960, 1041, 2),
                                    colors='black', transform=ccrs.PlateCarree(), linewidths=0.5)
    axes[1].clabel(slp_contour2, inline=True, fontsize=8, fmt='%d', colors='black')
    axes[1].set_title(r'925-hPa $\nabla\theta_e$ (shaded) and MSLP (contoured; hPa)')
    divider2 = make_axes_locatable(axes[1])
    cax2 = divider2.append_axes("right", size="6%", pad=0.1, axes_class=plt.Axes)
    cbar2 = fig.colorbar(im2, cax=cax2, orientation='vertical')
    cbar2.set_label('K/100km', fontsize=10)
    
    # PV subplot
    pv_levels = np.arange(0, 3.26, 0.25)
    pv_colors = ['white', '#d1e9f7', '#a5cdec', '#79a3d5', '#69999b', '#78af58', '#b0cc58', '#f0d95f', '#de903e', '#cb5428', '#b6282a', '#9b1622', '#7a1419']
    pv_cmap = mcolors.ListedColormap(pv_colors)
    pv_norm = mcolors.BoundaryNorm(pv_levels, pv_cmap.N)
    im3 = axes[2].contourf(lon, lat, pv_925, levels=pv_levels,
                            cmap=pv_cmap, alpha=0.5, transform=ccrs.PlateCarree(), extend='max')
    slp_contour3 = axes[2].contour(lon, lat, mslp, levels=np.arange(960, 1041, 4),
                                    colors='black', transform=ccrs.PlateCarree(), linewidths=0.5)
    axes[2].clabel(slp_contour3, inline=True, fontsize=8, fmt='%d', colors='black')
    axes[2].set_title('925-hPa PV (shaded) and MSLP (contoured; hPa)')
    divider3 = make_axes_locatable(axes[2])
    cax3 = divider3.append_axes("right", size="6%", pad=0.1, axes_class=plt.Axes)
    cbar3 = fig.colorbar(im3, cax=cax3, orientation='vertical')
    cbar3.set_label('PVU', fontsize=10)
    
    # Deformation subplot
    im4 = axes[3].contourf(lon, lat, deform_925, levels=np.arange(-30, 31, 2.5),
                            cmap='RdYlGn', transform=ccrs.PlateCarree(), extend='both')
    slp_contour4 = axes[3].contour(lon, lat, mslp, levels=np.arange(960, 1041, 4),
                                    colors='black', transform=ccrs.PlateCarree(), linewidths=0.5)
    axes[3].clabel(slp_contour4, inline=True, fontsize=8, fmt='%d', colors='black')
    axes[3].set_title('925-hPa Shearing Deformation (shaded) and MSLP (contoured; hPa)')
    divider4 = make_axes_locatable(axes[3])
    cax4 = divider4.append_axes("right", size="6%", pad=0.1, axes_class=plt.Axes)
    cbar4 = fig.colorbar(im4, cax=cax4, orientation='vertical')
    cbar4.set_label('1/s', fontsize=10)

    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()

    for ax in axes:
        ax.set_extent([directions['West'], directions['East'], directions['South'], directions['North']])
        ax.add_feature(cfeature.BORDERS)
        ax.coastlines()
        ax.add_feature(cfeature.LAND, color='#fbf5e9')
        gls = ax.gridlines(draw_labels=False, color='black', linestyle='-', alpha=0.35)
        gls.top_labels = False
        gls.right_labels = False
        ax.set_xticks(ax.get_xticks(), crs=ccrs.PlateCarree())
        ax.set_yticks(ax.get_yticks(), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
    
        if not df_matched.empty:
            ax.scatter(df_matched['lon'], df_matched['lat'],
                      marker='x', color='black', s=50, label='DetectNode', 
                      zorder=99, transform=ccrs.PlateCarree())
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle(current_time.strftime('%Y-%m-%d %H UTC'), fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def process_time_series(main_files,
                        save_plots=False,
                        output_dir=None,
                        directions=None,
                        bounds=None,
                        detect_nodes_df=None):
    """
    Process everything at one time 

    Parameters
    ----------
    main_files : list of str
        List of file paths for the main dataset
    save_plots : bool, optional
        Save plots Y/N
    output_dir : str, optional
        Directory to save plots
    directions : dict, optional
        Dict with keys 'North', 'South', 'East', 'West' for plot extent
    bounds : dict, optional
        Dict with lat and lon keys 
    detect_nodes_df : pandas.DataFrame, optional
        DetectNodes df

    Returns
    -------
    list of xarray.Dataset
        List of loaded datasets for each time step
    
    """
    if save_plots and output_dir: 
        os.makedirs(output_dir, exist_ok=True)

    datasets = []

    for main_path in main_files:
        print(f"Processing: {os.path.basename(main_path)}")

        ds_sliced = load_dataset(main_path, bounds)

        if ds_sliced is not None:
            datasets.append(ds_sliced)

            if save_plots and output_dir:
                timestamp = pd.to_datetime(ds_sliced.time.values).strftime('%Y%m%d_%H')
                save_path = os.path.join(output_dir, f"base_variables_{timestamp}.png")
                make_plots(ds_sliced, save_path=save_path, directions=directions, 
                          detect_nodes_df=detect_nodes_df)
            else:
                make_plots(ds_sliced, directions=directions, detect_nodes_df=detect_nodes_df)

    return datasets


if __name__ == "__main__": 
    BASE_PATH_MAIN = "/cw3e/mead/projects/csg101/aillenden/era5_data/wy2017"
    OUTPUT_DIR = "/cw3e/mead/projects/csg101/aillenden/plots/detectnodes/wy2017/"
    DETECT_NODES_PATH = r"/cw3e/mead/projects/csg101/aillenden/tempest_extremes/latest/output/detectnodes/wy2017/*.dat"
    DIRECTIONS = {'North': 52.5, 'South': 30.0, 'East': 240.0, 'West':  215.0}
    DATA_BOUNDS = {
        'latitude': slice(15.0, 60.0),
        'longitude': slice(165.0, 250.0)}

    # Define time range
    start_dt = datetime(2016, 10, 1, 0)   # start of WY
    end_dt   = datetime(2017, 9, 30, 21)  # end of WY

    print("=" * 50)
    print("Generating file paths...")
    print("=" * 50)

    main_files = generate_file_paths(
        start_dt.year, start_dt.month, start_dt.day, start_dt.hour,
        end_dt.year, end_dt.month, end_dt.day, end_dt.hour,
        base_path_main=BASE_PATH_MAIN)

    print(f"Total timesteps to process:  {len(main_files)}")

    print("="*50)
    print("Loading DetectNodes data...")
    print("="*50)
    
    file_list = glob.glob(DETECT_NODES_PATH)
    print(f"Found {len(file_list)} files matching: {DETECT_NODES_PATH}")
    
    if not file_list:
        print(f"WARNING: No files found for pattern: {DETECT_NODES_PATH}")
        print("Processing without DetectNodes overlay...")
        detect_nodes_df = None
    else:
        all_detect_dfs = []
        for filepath in file_list: 
            df = parse_detect_nodes_file(filepath)
            all_detect_dfs.append(df)
        
        detect_nodes_df = pd.concat(all_detect_dfs, ignore_index=True)
        print(f"Total DetectNodes loaded: {len(detect_nodes_df)} rows")
    
    print("="*50)
    print("Processing time series...")
    print("="*50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    process_time_series(
        main_files=main_files,
        save_plots=True,
        output_dir=OUTPUT_DIR,
        directions=DIRECTIONS,
        bounds=DATA_BOUNDS,
        detect_nodes_df=detect_nodes_df)

    print("="*50)
    print("Script is complete!")
    print("="*50)