import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import re
import ipywidgets as widgets
import random
import math
import os

from IPython.display import display, clear_output
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def read_stitch_file(filepath):
    """
    Read StitchNodes output CSV as a pandas df
    
    Parameters
    ----------
    filepath : str
        Path to StitchNodes output CSV file
    
    Returns
    -------
    df_stitch : pandas.DataFrame
        DataFrame containing stitched tracks with a datetime column

    """
    df_stitch = pd.read_csv(filepath)
    df_stitch.columns = df_stitch.columns.str.strip() 
    df_stitch["datetime"] = pd.to_datetime(dict(year=df_stitch["year"], month=df_stitch["month"], day=df_stitch["day"], hour=df_stitch["hour"]))
    return df_stitch

def read_landfalling_file(filepath):
    """
    Read landfalling AR CSV as a pandas df
    
    Parameters
    ----------
    filepath : str
        Path to landfalling AR CSV file
    
    Returns
    -------
    df_landfalling : pandas.DataFrame
        DataFrame containing landfalling AR events with datetime columns

    """
    df_landfalling = pd.read_csv(filepath, skipinitialspace=True)
    df_landfalling["Start Date of AR"] = pd.to_datetime(df_landfalling["Start Date of AR"].astype(int), format="%Y%m%d%H")
    df_landfalling["End Date of AR"] = pd.to_datetime(df_landfalling["End Date of AR"].astype(int), format="%Y%m%d%H")
    df_landfalling['Longitude'] = df_landfalling['Longitude'] * -1
    return df_landfalling

def filter_stitched_by_landfalling(df_stitched, df_landfalling, overlap_threshold=0.75):
    """
    Filter stitched tracks to only those that exist during a landfalling AR event

    Parameters
    ----------
    df_stitched : pandas.DataFrame
        DataFrame containing stitched tracks with a datetime column
    df_landfalling : pandas.DataFrame
        DataFrame containing landfalling AR events with datetime columns
    overlap_threshold : float, optional
        Minimum fraction of overlap between track duration and landfalling AR duration 

    Returns
    -------
    df_stitched_new_filtered : pandas.DataFrame
        Filtered DataFrame of stitched tracks that meet the landfalling AR overlap threshold

    """
    # ensure df_stitched has a datetime column
    if 'datetime' not in df_stitched.columns and 'time' in df_stitched.columns:
        df_stitched['datetime'] = pd.to_datetime(df_stitched['time'], errors='coerce')
    elif 'datetime' in df_stitched.columns:
        df_stitched['datetime'] = pd.to_datetime(df_stitched['datetime'], errors='coerce')
    else:
        raise RuntimeError("df_stitched missing 'datetime' or 'time' column")

    min_dt, max_dt = df_stitched['datetime'].min(), df_stitched['datetime'].max()
    df_landfalling_small = df_landfalling[
        (df_landfalling['End Date of AR'] >= min_dt) &
        (df_landfalling['Start Date of AR'] <= max_dt)].copy()

    if df_landfalling_small.empty:
        print("No landfalling events overlap stitched time span")
        return df_stitched.iloc[0:0].copy()

    kept_ids = []
    for obj_id, group in df_stitched.groupby('track_id'):
        dt_vals = group['datetime'].dropna().sort_values()
        if dt_vals.empty:
            continue
        
        track_start = dt_vals.min()
        track_end = dt_vals.max()
        track_duration = (track_end - track_start).total_seconds()
        
        if track_duration == 0:
            # Single timestep track
            # Check if it falls within any AR
            for _, ar_row in df_landfalling_small.iterrows():
                if track_start >= ar_row['Start Date of AR'] and track_start <= ar_row['End Date of AR']:
                    kept_ids.append(obj_id)
                    break
            continue
        
        # Calculate total overlap with landfalling ARs
        total_overlap = 0
        for _, ar_row in df_landfalling_small.iterrows():
            ar_start = ar_row['Start Date of AR']
            ar_end = ar_row['End Date of AR']
            
            # Calculate overlap between track and this AR
            overlap_start = max(track_start, ar_start)
            overlap_end = min(track_end, ar_end)
            
            if overlap_start <= overlap_end:
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                total_overlap += overlap_duration
        
        # Calculate overlap %
        overlap_fraction = total_overlap / track_duration
        
        if overlap_fraction >= overlap_threshold:
            kept_ids.append(obj_id)

    df_stitched_new_filtered = df_stitched[df_stitched['track_id'].isin(kept_ids)].copy()
    print(f"stitched span: {min_dt} to {max_dt}")
    print(f"landfall events considered: {len(df_landfalling_small)} (after filtering to stitched times)")
    print(f"kept unique track_ids: {len(kept_ids)} (overlap_threshold={overlap_threshold*100:.0f}%)")
    return df_stitched_new_filtered

df_stitch = read_stitch_file(r"C:\Users\Tony\Desktop\stitch_m3.0_r3p5_mt9h_mg1.csv")
df_landfalling = read_landfalling_file(r'C:\Users\Tony\Documents\GitHub\aware\preprocessing\landfalling_events\concatenated_landfalling_events.csv')

# Convert lons from 0-360 to -180-180 
if 'lon' in df_stitch.columns and df_stitch['lon'].max() > 180:
    df_stitch['lon'] = df_stitch['lon'].apply(lambda x: x-360 if x > 180 else x)
    print("Converted df_stitch longitude to -180 to 180 format")

# Also convert landfalling AR lons
lon_cols = [col for col in df_landfalling.columns if 'lon' in col.lower()]
if lon_cols:
    for col in lon_cols:
        if df_landfalling[col].max() > 180:
            df_landfalling[col] = df_landfalling[col].apply(lambda x: x-360 if x > 180 else x)
            print(f"Converted df_landfalling {col} to -180 to 180 format")

filtered = filter_stitched_by_landfalling(df_stitch, df_landfalling, overlap_threshold=0.75)

new_directions = {'North': 52.5, 'South': 30.0, 'East': 240.0, 'West': 215.0}

# Calculate and display max/min coordinates
print(f"Longitude range: {filtered['lon'].min():.2f}° to {filtered['lon'].max():.2f}°")
print(f"Latitude range: {filtered['lat'].min():.2f}°N to {filtered['lat'].max():.2f}°N")

fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})

ax.set_extent([new_directions['West'], new_directions['East'], new_directions['South'], new_directions['North']])

ax.add_feature(cfeature.BORDERS)
ax.coastlines()
ax.add_feature(cfeature.LAND, color='#fbf5e9')
ax.add_feature(cfeature.OCEAN, color='#e6f9fe')

# Set gridlines with latitude every 5 degrees
gls = ax.gridlines(draw_labels=True, color='black', linestyle='-', alpha=0.35,
                   xlocs=range(-180, 181, 5), ylocs=range(-90, 91, 5))

# Plot each track separately
for track_id, track_data in filtered.groupby('track_id'):
    ax.plot(track_data["lon"], track_data["lat"], marker='o', linestyle='-', 
            markersize=4, linewidth=2, label=f"Track {track_id}", alpha=0.7)

lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
gls.xformatter = lon_formatter
gls.yformatter = lat_formatter

plt.title("WY2015 - Landfalling AR-MFW Tracks", fontsize=16)
plt.tight_layout()
plt.savefig("wy2015_tracks.png", dpi=300, bbox_inches='tight')
plt.show()