#!/bin/bash
#SBATCH --job-name="te_tracking"
#SBATCH --output="te_tracking.%j.%N.out.txt"
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --export=ALL
#SBATCH --account=${ACCOUNT}
#SBATCH --time=4:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${MAILUSER}
#######################################################

#######################################################
#                     User Settings
#######################################################

# For a single WY do WY=(2015)
# For a range of specific WYs, do WY=(2015 2017 2020)
# For a range of WYs, do WY ($(seq 2010 2020)) 
WATER_YEARS=($(seq 2010 2020))

# Plot StitchNodes landfalling AR-MFW cases for an entire WY  
PLOT_OUTPUT=true  # true or false

# TempestExtremes parameters (see userguide for details)
MERGDIST=3.0          # float
RANGE=3.5             # float
MIN_ENDPOINT_DIST=3.0 # float
MINTIME="9h"          # str
MAXGAP=1              # int

#######################################################
#                    END USER SETTINGS                #
#######################################################

# Define paths
BASE_DIR="/cw3e/mead/projects/csg101/aillenden/tempest_extremes/latest/scripts/"
INPUT_PATH_DIR="/cw3e/mead/projects/csg101/aillenden/tempest_extremes/input_paths"
LANDFALLING_PATH="/cw3e/mead/projects/csg101/aillenden/concatenated_landfalling_events.csv"

# Loop through specified water years
for WY in "${WATER_YEARS[@]}"; do

    echo "=========================================="
    echo "   Processing Water Year: ${WY}"
    echo "=========================================="

    # ERA5 input directory and list file
    ERA5_DIR="/cw3e/mead/projects/csg101/aillenden/era5_data/wy${WY}/"
    INPUT_LIST="/cw3e/mead/projects/csg101/aillenden/tempest_extremes/input_paths/wy${WY}_extra_era5_paths.txt"

    # Output directories
    DETECT_OUT_DIR="/cw3e/mead/projects/csg101/aillenden/tempest_extremes/latest/output/detectnodes/wy${WY}"
    STITCH_OUT_DIR="/cw3e/mead/projects/csg101/aillenden/tempest_extremes/latest/output/stitchnodes/"
    STITCH_OUT_PLOT_DIR="/cw3e/mead/projects/csg101/aillenden/tempest_extremes/latest/plots/"
    mkdir -p "${DETECT_OUT_DIR}" "${STITCH_OUT_DIR}"
    
    # Only create plot dir if plotting is enabled
    if [ "$PLOT_OUTPUT" = true ]; then
        mkdir -p "${STITCH_OUT_PLOT_DIR}"
    fi

    #######################################################
    #               Create DetectNodes Input List         #
    #######################################################

    # Find all .nc files in ERA5_DIR and create sorted list
    find "${ERA5_DIR}" -type f -name "*.nc" | sort > "${INPUT_LIST}"
    FILE_COUNT=$(wc -l < "${INPUT_LIST}")
    echo "Wrote ${FILE_COUNT} ERA5 files to ${INPUT_LIST}"

    rm -f "${DETECT_OUT_DIR}"/*

    #######################################################
    #                     Run DetectNodes                 #
    #######################################################

    echo ">>> DetectNodes: mergdist=${MERGDIST} -> output dir: ${DETECT_OUT_DIR}"

    DetectNodes \
        --in_data_list "${INPUT_LIST}" \
        --out "${DETECT_OUT_DIR}/possible_pv_objects_m${MERGDIST}.txt" \
        --searchbymax "pv_925" \
        --mergedist "${MERGDIST}" \
        --closedcontourcmd "pv_925,-0.50,2,0" \
        --noclosedcontourcmd "z_500,4,5,0" \
        --thresholdcmd "IVT,>=,400,0.25;shearing_def_925,>=,12.5,0;thetae_grad,>=,5,0" \
        --lonname "longitude" \
        --latname "latitude" \
        --maxlat 52.5 \
        --minlat 30.0 \
        --minlon 215 \
        --maxlon 250 \
        --regional \
        --timefilter "3hr" \
        --outputcmd "pv_925,max,0;IVT,max,0.25;thetae_grad,max,0;tadv_925,min,0.25;tadv_925,max,0;tadv_925,max,0.25;z_500,min,0.25;fgen_925,min,0.25;fgen_925,max,0;fgen_925,max,0.25;shearing_def_925,max,0;MSLP,max,0" \
        --out_header

    find "${BASE_DIR}" -type f -name "log*.txt" -delete

    #######################################################
    #             Create StitchNodes Input List           #
    #######################################################

    # Create list of DetectNodes output files for StitchNodes input
    LIST_FILE="${INPUT_PATH_DIR}/pv_ivt_def_objects_detect_nodes_paths_${MERGDIST}.txt"
    find "${DETECT_OUT_DIR}" -type f ! -name ".*" ! -name "*.tmp" ! -name "*~" | sort > "${LIST_FILE}"
    echo "Wrote list file: ${LIST_FILE}"

    #######################################################
    #                     Run StitchNodes                 #
    #######################################################

    echo ">>> StitchNodes: mergdist=${MERGDIST}, range=${RANGE}, mintime=${MINTIME}, maxgap=${MAXGAP}"

    # Create filename with parameters
    MIN_END_LABEL="$(printf '%s' "${MIN_ENDPOINT_DIST}" | sed 's/\./p/')"
    RANGE_LABEL="$(printf '%s' "${RANGE}" | sed 's/\./p/')"
    STITCH_OUT_FILE="${STITCH_OUT_DIR}/wy${WY}_stitch_m${MERGDIST}_r${RANGE_LABEL}_mt${MINTIME}_mg${MAXGAP}.csv"

    StitchNodes \
        --in_list "${LIST_FILE}" \
        --in_fmt "lon,lat,pv_925,IVT,thetae_grad,tadv_925,tadv_925,tadv_925,z_500,fgen_925,fgen_925,fgen_925,shearing_def_925,MSLP" \
        --out "${STITCH_OUT_FILE}" \
        --range "${RANGE}" \
        --mintime "${MINTIME}" \
        --min_endpoint_dist "${MIN_ENDPOINT_DIST}" \
        --maxgap "${MAXGAP}" \
        --out_file_format "csv"

    find "${BASE_DIR}" -type f -name "log*.txt" -delete

    ##############################################################
    # Plot StitchNodes landfalling AR-MFW cases for an entire WY #
    ##############################################################
    
    if [ "$PLOT_OUTPUT" = true ]; then
        echo ">>> Plotting StitchNodes output for WY${WY}..."
        source activate thesis # note that thesis is the name of my Python environment
        python - <<EOF
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
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

def convert_longitudes(df, lon_col):
    """
    Convert longitude column from 0-360 to -180-180

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing longitude column
    lon_col : str
        Name of longitude column in df
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with longitude column converted to -180 to 180 format if it was in 0-360 format

    """
    if lon_col in df.columns and df[lon_col].max() > 180:
        df[lon_col] = df[lon_col].apply(lambda x: x-360 if x > 180 else x)
        print(f"Converted {lon_col} to -180 to 180 format")
    return df

def plot_tracks(filtered, new_directions, output_file):
    """
    Plot filtered stitched tracks on a map
    
    Parameters
    ----------
    filtered : pandas.DataFrame
        DataFrame containing filtered stitched tracks with 'lon', 'lat', and 'track_id' columns
    new_directions : dict
        Dict with keys 'North', 'South', 'East', 'West' for plot extent
    output_file : str
        Path to save the plot
    
    Returns
    -------
    Saves plot to output_file path
    
    """
    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([new_directions['West'], new_directions['East'], new_directions['South'], new_directions['North']])
    ax.add_feature(cfeature.BORDERS)
    ax.coastlines()
    ax.add_feature(cfeature.LAND, color='#fbf5e9')
    ax.add_feature(cfeature.OCEAN, color='#e6f9fe')
    gls = ax.gridlines(draw_labels=True, color='black', linestyle='-', alpha=0.35,
                       xlocs=range(-180, 181, 5), ylocs=range(-90, 91, 5))
    for track_id, track_data in filtered.groupby('track_id'):
        ax.plot(track_data["lon"], track_data["lat"], marker='o', linestyle='-', 
                markersize=4, linewidth=2, label=f"Track {track_id}", alpha=0.7)
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    gls.xformatter = lon_formatter
    gls.yformatter = lat_formatter
    plt.title(f"WY${WY} - Landfalling AR-MFW Tracks", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    #plt.show()


def main():
    stitch_path = "${STITCH_OUT_FILE}"
    landfalling_path = "${LANDFALLING_PATH}"
    df_stitch = read_stitch_file(stitch_path)
    df_landfalling = read_landfalling_file(landfalling_path)

    # Convert longitudes from 0-360 to -180-180 format BEFORE filtering
    df_stitch = convert_longitudes(df_stitch, 'lon')
    lon_cols = [col for col in df_landfalling.columns if 'lon' in col.lower()]
    for col in lon_cols:
        df_landfalling = convert_longitudes(df_landfalling, col)

    filtered = filter_stitched_by_landfalling(df_stitch, df_landfalling, overlap_threshold=0.75)
    new_directions = {'North': 52.5, 'South': 30.0, 'East': 240.0, 'West': 215.0}
    plot_tracks(filtered, new_directions, output_file="${STITCH_OUT_PLOT_DIR}/wy${WY}_tracks.png")

if __name__ == "__main__":
    main()

EOF
    else
        echo ">>> Skipping plotting"
    fi

    echo "=========================================="
    echo "   Completed WY: ${WY}"
    echo "=========================================="
    echo ""

done
echo "Script is complete!"
echo "==================================================="