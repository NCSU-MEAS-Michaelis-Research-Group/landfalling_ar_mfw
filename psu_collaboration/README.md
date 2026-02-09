# Objective Detection of Mesoscale Frontal Waves along Landfalling Atmospheric Rivers
## Directory Structure

```
psu_collaboration/
├── data/                                      
│   ├── concatenated_landfalling_events.csv    # Landfalling AR Catalog from Jay
│   └── era5_landmask.nc                       # ERA5 land-sea mask
│
├── docs/                                      
│   └── landfalling_events.md                  # Description of Landfalling AR Catalog from Jay
│
├── figures/                                    
│   └── detectnodes_example.png                # DetectNodes output example 4-panel plot
│   └── stitchnodes_example.png                # StitchNodes output example for an entire WY
├── scripts/                                    
│   ├── preprocessing/                         
│   │   ├── download_era5.py                   # Python script to download ERA5 data
│   │   └── download_era5.sh                   # Shell script that runs the Python script to download ERA5 data
│   │
│   ├── tracking/                              
│   │   └── detect_and_stitch.sh               # Objective Tracking Algorithm using TempestExtremes
│   │
│   ├── postprocessing/                        
│   │   └── apply_landfalling_filter.py        # Apply landfalling filter to StitchNodes output
│   │
│   └── plotting/                              
│       ├── interactive_stitchnodes.ipynb      # Interactive notebook for StitchNodes output
│       ├── plot_detectnodes.py                # Plot DetectNodes output on 4-panel plot
│       └── plot_stitchnodes.py                # Plot StitchNodes output for an entire WY
│
└── environment.yml                            # Python environment needed for the Python scripts used throughout this repo
```
## TempestExtremes Setup
Can install using CMake or conda, which are outlined in TempestExtremes' [user guide](https://github.com/NCSU-MEAS-Michaelis-Research-Group/landfalling_ar_mfw/blob/main/psu_collaboration/docs/TempestExtremes_userguide.pdf), [GitHub](https://github.com/ClimateGlobalChange/tempestextremes), and [online](https://climate.ucdavis.edu/tempestextremes.php). If you have issues installing via CMake, GitHub's Copilot (or ChatGPT) may be able to help debug. 


## Python Environment Setup

Create the conda environment with all required dependencies:

```bash
conda env create -f environment.yml
conda activate landfalling_ar_mfw
```

Note: It is possible that you will have to install additional dependencies. 

## Workflow

1. **Preprocessing**: Download ERA5 reanalysis data using the script in `scripts/preprocessing/` (when downloading the data it applies the land-sea mask filter, so be sure to download that netCDF and update the file path)
2. **Tracking**: Detect and stitch AR nodes using `scripts/tracking/detect_and_stitch.sh`
3. **Postprocessing/Plotting**: Apply landfalling filter to StitchNodes output and plot it with scripts in `scripts/plotting/`

Note: The current `detect_and_stitch.sh` script includes a Python script that applies the landfalling AR filter to the StitchNodes output and plots all tracks. You can turn this off by changing the user setting to false. 

## TempestExtremes Tracking

The objective tracking algorithm consists of DetectNodes and StitchNodes.

### DetectNodes
DetectNodes detects node features based on the following thresholds:

**Configuration:**
```bash
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
```

**Parameters:**
| Parameter | Purpose |
|-----------|---------|
| `--searchbymax "pv_925"` | Search for local maxima in 925-hPa PV |
| `--mergedist` | Merge distance between nodes (note that it is possible to have several candidates closer than this) |
| `--closedcontourcmd "pv_925,-0.50,2,0"` | Must have a closed contour of 925-hPa PV >= 0.50 PVU than the field within 2 degrees |
| `--noclosedcontourcmd "z_500,4,5,0"` | Must not have a closed contour of 500-hPa geopotential height that decreases more than 4 dam within 5 degrees of node  |
| `--thresholdcmd` | Apply filtering for other thresholds: IVT >= 400 kg/m/s with 0.25 degrees of node, 925-hPa shearing deformation >= 12.5 1/s on node, and 925-hPa theta-e gradient >= 5 K on node|

### StitchNodes

StitchNodes stitches those node features from DetectNodes, but they must be within the following thresholds before they are stitched. 

**Configuration:**
```bash
StitchNodes \
  --in_list "${LIST_FILE}" \
  --in_fmt "lon,lat,pv_925,IVT,thetae_grad,tadv_925,tadv_925,tadv_925,z_500,fgen_925,fgen_925,fgen_925,shearing_def_925,MSLP" \
  --out "${STITCH_OUT_FILE}" \
  --range "${RANGE}" \
  --mintime "${MINTIME}" \
  --min_endpoint_dist "${MIN_ENDPOINT_DIST}" \
  --maxgap "${MAXGAP}" \
  --out_file_format "csv"
```

**Parameters:**
| Parameter | Purpose |
|-----------|---------|
| `--range` | Stitch candidates within this distance |
| `--mintime` | Minimum duration |
| `--min_endpoint_dist` | Minimum distance between first and last nodes |
| `--maxgap` | Maximum temporal gap allowed |

You can edit these parameters to your own liking.

## Final Steps / Comments
Before running these scripts, be sure to adjust all file paths. Additionally, while some shell scripts automatically create directories, you will likely need to manually create some as well. If you run into any issues, please reach out to me! 

On average, the algorithm takes about 8-10 minutes per WY. Applying the landfalling filter takes seconds. Other TempestExtremes features, such as DetectBlobs and StitchBlobs, are also available. We tried using these to track AR objects using IVT, but found no added value (and it added additional compute time).

## Data Sources

Landfalling AR event data is from Dr. Jay Cordeira's [CW3E AR Catalog](https://cw3e.ucsd.edu/Projects/ARCatalog/catalog.html). See [docs/landfalling_events.md](docs/landfalling_events.md) for detailed information about the data format and columns. ERA5 data is from the [University Corporation for Atmospheric Research](https://rda.ucar.edu/datasets/d633000/dataaccess/). TempestExtremes tracking algorithm is from Ullrich and Zarzycki (2017).