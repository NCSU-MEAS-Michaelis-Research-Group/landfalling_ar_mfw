import xarray as xr
import pandas as pd
import os
import numpy as np
import metpy.calc as mpcalc
from siphon.catalog import TDSCatalog
from metpy.units import units

lsm_file = "/cw3e/mead/projects/csg101/aillenden/preprocessing/lsm/era5_landmask.nc"
ds_lsm = xr.open_dataset(lsm_file)

def load_datasets(year, month, start_day, start_hour=0, end_day=None, end_hour=23):
    # Set end_day to start_day if not provided
    if end_day is None:
        end_day = start_day
    
    # Get the last day of the month
    last_day_of_month = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)
    last_day_str = f"{last_day_of_month.day:02d}"  # format last day as two digits

    # Format date and time strings
    year_month = f'{year}{month:02d}'
    start_time = f'{year}{month:02d}{start_day:02d}{start_hour:02d}'  # yyyymmddhh (start)
    end_time = f'{year}{month:02d}{end_day:02d}{end_hour:02d}'  # yyyymmddhh (end)

    # Define URLs for pressure level datasets with specific time ranges
    urls = {
        'temperature_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_130_t.ll025sc.{start_time}_{end_time}.nc',
        'humidity_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_133_q.ll025sc.{start_time}_{end_time}.nc',
        'v_wind_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_132_v.ll025uv.{start_time}_{end_time}.nc',
        'u_wind_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_131_u.ll025uv.{start_time}_{end_time}.nc',
        'pv_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_060_pv.ll025sc.{start_time}_{end_time}.nc',
        'geopotential_pl': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.pl/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.pl/{year_month}/e5.oper.an.pl.128_129_z.ll025sc.{start_time}_{end_time}.nc',
        'mslp_sfc': f'https://thredds.rda.ucar.edu/thredds/catalog/files/g/d633000/e5.oper.an.sfc/{year_month}/catalog.html?dataset=files/g/d633000/e5.oper.an.sfc/{year_month}/e5.oper.an.sfc.128_151_msl.ll025sc.{year_month}0100_{year_month}{last_day_str}23.nc'}

    # Initialize empty dictionaries for datasets
    datasets = {}

    # Try to load datasets from the URLs
    for var, url in urls.items():
        try:
            tds_catalog = TDSCatalog(url)
            ds_url = tds_catalog.datasets[0].access_urls['OPENDAP']
            ds = xr.open_dataset(ds_url)
            datasets[var] = ds
            print("="*50)
            print(f"Successfully loaded {var}")
            print("="*50)
        except Exception as e:
            print("="*50)
            print(f"Error loading {var}: {e}")
            print("="*50)

    # Merge pressure level datasets if available
    ds_pl, ds_sfc = None, None

    try:
        ds_pl = xr.merge([datasets['temperature_pl'], datasets['humidity_pl'], 
                        datasets['v_wind_pl'], datasets['u_wind_pl'],
                        datasets['pv_pl'], datasets['geopotential_pl']])
        print("="*50)
        print("Successfully merged pressure level datasets")
        print("="*50)
    except KeyError as e:
        print("="*50)
        print(f"Error merging pressure level datasets: {e}")
        print("="*50)

    # Merge surface datasets if available
    try:
        ds_sfc = xr.merge([datasets['mslp_sfc']])
        print("="*50)
        print("Successfully merged surface datasets")
        print("="*50)
    except KeyError as e:
        print("="*50)
        print(f"Error merging surface datasets: {e}")
        print("="*50)

    # Synchronize time dimensions
    try:
        if ds_pl is not None and ds_sfc is not None:
            first_time_pl, last_time_pl = ds_pl['time'].min().values, ds_pl['time'].max().values
            ds_sfc = ds_sfc.sel(time=slice(first_time_pl, last_time_pl))
    except KeyError as e:
        print("="*50)
        print(f"Error accessing 'time' in the datasets: {e}")
        print("="*50)
    except Exception as e:
        print("="*50)
        print(f"An error occurred during slicing: {e}")
        print("="*50)
        
    return ds_pl, ds_sfc

def merge_ds(ds_pl, ds_sfc):
    ds_merged = xr.merge([ds_pl, ds_sfc])
    return ds_merged

def slice_domain(ds, directions):
    ds_sliced = ds.sel(latitude=slice(directions['North']+10, directions['South']-10), longitude=slice(directions['West']-10, directions['East']+10))
    return ds_sliced

def calc_vars(ds, level, output_dir):
    g = 9.81  # units: m/s2
    level_hpa = level * units.hPa  # units: hPa
    # Filter time to include only the specified hours
    selected_hours = [0, 3, 6, 9, 12, 15, 18, 21]
    time_list = ds['time'].sel(time=ds['time'].dt.hour.isin(selected_hours))

    # Loop over filtered time steps and perform calculations
    for time in time_list:
        try:
            # Select variables at all pressure levels
            u_sliced = ds['U'].sel(time=time, level=slice(300, 1000))  # units: m/s
            v_sliced = ds['V'].sel(time=time, level=slice(300, 1000))  # units: m/s
            q_sliced = ds['Q'].sel(time=time, level=slice(300, 1000))  # units: kg/kg
            t_sliced = ds['T'].sel(time=time, level=slice(300, 1000))  # units: K
            pv_sliced = ds['PV'].sel(time=time, level=slice(300, 1000))  * 1e6 # units: PVU
            geopotential_sliced = ds['Z'].sel(time=time, level=slice(300, 1000))  / g # units: m
            mslp = ds['MSL'].sel(time=time) / 100  # Convert Pa to hPa

            # Select variables at the specified level
            u_lev = u_sliced.sel(level=level_hpa)  # units: m/s
            v_lev = v_sliced.sel(level=level_hpa)  # units: m/s
            q_lev = q_sliced.sel(level=level_hpa)  # units: kg/kg
            t_lev = t_sliced.sel(level=level_hpa)  # units: K
            pv_lev = pv_sliced.sel(level=level_hpa)  # units: PVU
            p_lev = level * units.hPa  # units: hPa
            z_500 = geopotential_sliced.sel(level=500 * units.hPa)  / 10  # units: m
            u_500 = u_sliced.sel(level=500 * units.hPa)  # units: m/s
            v_500 = v_sliced.sel(level=500 * units.hPa)  # units: m/s

            # Get grid spacing 
            lats, lons = u_sliced['latitude'], u_sliced['longitude']
            dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats) # units: m

            # Extract pressure levels
            pressure_levels = u_sliced['level'][::-1] * 100  # units: Pa

            # Calculate IVT
            u_ivt = -1 / g * np.trapz(u_sliced * q_sliced, pressure_levels, axis=0)
            v_ivt = -1 / g * np.trapz(v_sliced * q_sliced, pressure_levels, axis=0)
            
            # Save IVT u- and v-components
            u_ivt_da = xr.DataArray(u_ivt, dims=['latitude', 'longitude'],
                                  coords={'latitude': lats, 'longitude': lons})
            u_ivt_da.name = 'uIVT'
            
            v_ivt_da = xr.DataArray(v_ivt, dims=['latitude', 'longitude'],
                                  coords={'latitude': lats, 'longitude': lons})
            v_ivt_da.name = 'vIVT'

            # Calculate IVT magnitude
            ivt = np.sqrt(u_ivt**2 + v_ivt**2)
            ivt_da = xr.DataArray(ivt, dims=['latitude', 'longitude'],
                                  coords={'latitude': lats, 'longitude': lons})
            ivt_da.name = 'IVT'

            # Calculate dewpoint and theta-e
            td_lev = mpcalc.dewpoint_from_specific_humidity(p_lev, t_lev, q_lev)
            theta_e = mpcalc.equivalent_potential_temperature(p_lev, t_lev, td_lev)

            # Compute theta-e gradient
            dtheta_e_dx, dtheta_e_dy = mpcalc.geospatial_gradient(theta_e)
            theta_e_gradient = np.sqrt(dtheta_e_dx**2 + dtheta_e_dy**2) * 1e5  # Convert to K/100 km
            theta_e_gradient_da = xr.DataArray(theta_e_gradient, dims=['latitude', 'longitude'],
                                              coords={'latitude': lats, 'longitude': lons})
            theta_e_gradient_da.name = 'thetae_grad'

            # Calculate 925-hPa temperature advection
            tadv_925 = mpcalc.advection(t_lev, u_lev, v_lev) * 3600 # units: K/hr

            # Calculate 925-hPa potential temperature
            theta_925 = mpcalc.potential_temperature(p_lev, t_lev)

            # Calculate 925-hPa 2D frontogenesis
            fgen_925 = mpcalc.frontogenesis(theta_925, u_lev, v_lev) * 1000*100*3600*3 # units: K per 100 km 3h

            # Calculate 925-hPa shearing deformation
            shearing_def_925 = mpcalc.shearing_deformation(u_lev, v_lev, dx, dy) * 1e5 # units: 1/s 

            # Assign new variables
            ds = ds.assign({
                'MSLP': mslp,
                'uIVT': u_ivt_da,
                'vIVT': v_ivt_da,
                'IVT': ivt_da,
                'thetae_grad': theta_e_gradient_da,
                'pv_925': pv_lev,
                'tadv_925': tadv_925,
                'fgen_925': fgen_925,
                'shearing_def_925': shearing_def_925,
                'z_500': z_500,
                'u_500': u_500,
                'v_500': v_500})

            # Keep only final variables and add time dimension
            final_variables = ['MSLP', 'uIVT', 'vIVT', 'IVT', 'thetae_grad', 'pv_925', 'tadv_925', 'fgen_925', 'shearing_def_925', 'z_500', 'u_500', 'v_500']
            ds_final = ds[final_variables]

            formatted_time = pd.to_datetime(time.values)
            ds_final = ds_final.expand_dims(time=[formatted_time])
            time_str = formatted_time.strftime('%Y%m%d_%H')
            
            ds_final_filtered = ds_final.where(ds_lsm['LSM'] <= 0)
            save_ds(ds_final_filtered, output_dir, time_str)
            print(f"Processed time step {time_str}!")

        except Exception as e:
            print(f"Error processing time step: {e}")
            continue  # continue to next time step

def save_ds(ds, output_dir, time_str):
    """
    Saves the dataset to a NetCDF file in the specified output directory.
    
    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to save.
    output_dir : str
        The directory where the dataset will be saved.
    time_str : str
        The time string used for naming the output file.
    Returns
    -------
    None

    """
    if ds is None:
        print(f"Dataset is None for time {time_str}. Skipping save.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"sliced_{time_str}.nc")
    ds = ds.sortby('latitude')
    ds.to_netcdf(output_file)

def main(start_date, end_date, directions, output_dir='', level=925):
    """
    Loops over dates from start_date to end_date (inclusive).
    
    Parameters
    ----------
    start_date : str or pd.Timestamp
        Start date in 'YYYY-MM-DD' format (or a pandas Timestamp).
    end_date : str or pd.Timestamp
        End date in 'YYYY-MM-DD' format (or a pandas Timestamp).
    output_dir : str
        Directory to save processed files.
    level : int
        Pressure level in hPa.
    """
    print("="*50)
    print("Script is running...")
    print("="*50)

    # Generate all dates between start_date and end_date
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    for date in dates:
        year, month, day = date.year, date.month, date.day
        print("="*50)
        print(f"Processing data for {year}-{month:02d}-{day:02d}...")
        print("="*50)

        try:
            # Load datasets for this day
            ds_pl, ds_sfc = load_datasets(year, month, start_day=day, end_day=day)

            if ds_pl is None:
                print("="*50)
                print(f"Skipping {date.strftime('%Y-%m-%d')} due to missing datasets.")
                print("="*50)
                continue

            # Merge datasets if both exist
            ds_merged = merge_ds(ds_pl, ds_sfc) if ds_sfc is not None else ds_pl

            # Slice the ds to the specified domain
            ds_sliced = slice_domain(ds_merged, directions)

            # Calculate variables and save
            calc_vars(ds_sliced, level, output_dir)

        except Exception as e:
            print("="*50)
            print(f"Error processing {date.strftime('%Y-%m-%d')}: {e}")
            print("="*50)
            continue

    print("="*50)
    print("Script is complete!")
    print("="*50)


if __name__ == '__main__':
    start_date = "2014-10-01" # YYYY-MM-DD
    end_date = "2015-09-30" # YYYY-MM-DD
    WY = 2015
    level = 925 # units: hPa
    directions = {'North': 55, 'East': 250, 'South': 20, 'West': 200} # units: degrees North, degrees East
    output_dir = f"/cw3e/mead/projects/csg101/aillenden/era5_data/wy{WY}/"

    main(start_date, end_date, directions, output_dir, level)