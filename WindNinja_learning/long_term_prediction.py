import numpy as np
import pandas as pd
import xarray as xr
from osgeo import gdal
from scipy.spatial import cKDTree
import concurrent.futures
import os
from time import time as t

from downscale.data_source.observation import Observation
from downscale.data_source.data_2D import Data_2D
from WindNinja_learning.prm_windninja import create_prm
from WindNinja_learning.dem_processing import crop_dem, find_nearest_neighbor_in_grid


"""
77 minutes for one month at 1 station
"""

def preprocess_function(netCDF_file):
    try:
        netCDF_file = netCDF_file.assign_coords(time=("time", netCDF_file.time.data))
    except:
        netCDF_file = netCDF_file.assign_coords(time=("oldtime", netCDF_file.time.data))

    netCDF_file = netCDF_file.assign_coords(xx=("xx", list(range(netCDF_file.dims['xx']))))
    netCDF_file = netCDF_file.assign_coords(yy=("yy", list(range(netCDF_file.dims['yy']))))

    try:
        netCDF_file = netCDF_file.rename({'oldtime': 'time'})
    except:
        pass

    return netCDF_file


# Create a prm file containing the path to data, the considered date, dx and dy of the MNT the options etc
prm = create_prm()

# Load the correct AROME files
AROME = xr.open_mfdataset(prm["AROME_files"],
                          preprocess=preprocess_function,
                          concat_dim='time').astype(np.float32, copy=False)

# Select the date considered
AROME = AROME.sel(time=slice(prm["begin"], prm["end"]))

# Load AROME cloud cover
# todo load cloud cover

# Load AROME temperature
temperature = xr.open_mfdataset(prm["AROME_files_temperature"],
                          preprocess=preprocess_function,
                          concat_dim='time').astype(np.float32, copy=False)

# Load observation metadata
BDclim = pd.read_csv(prm["BDclim_stations_path"])
list_variables_str = ['index_AROME_NN_0_ref_AROME']
BDclim[list_variables_str] = BDclim[list_variables_str].apply(lambda x: x.apply(eval))

results = {}

t0 = t()
# Iterate for each station
for station in prm["stations"]:

    x_l93, y_l93 = BDclim[['X', 'Y']][BDclim["name"] == station].values[0]

    # Select the nearest neighbor to the station
    idx_str = f"index_{prm['nwp_name']}_NN_0_ref_{prm['nwp_name']}"
    y_idx_nwp, x_idx_nwp = BDclim[idx_str][BDclim["name"] == station].values[0]
    y_idx_nwp, x_idx_nwp = np.int16(y_idx_nwp), np.int16(x_idx_nwp)

    AROME_station = AROME.isel(xx=x_idx_nwp, yy=y_idx_nwp)
    temperature_station = temperature.isel(xx=x_idx_nwp, yy=y_idx_nwp)

    # Modify name station
    station_lower = station.lower().replace("'", "_").replace("-", "_").replace(" ", "_")

    # Crop MNT around the stations and save it somewhere
    crop_dem(x_l93-1_000, y_l93+1_000, x_l93+1_000, y_l93-1_000,
             "m", f"{station_lower}_2km", prm["topo_path"], prm["windninja_topo"], 2154, 2154)

    times = []
    speeds = []
    directions = []

    # Iterate on time
    for index, time in enumerate(AROME_station.time.values):
        if index <= 1:
            speed = np.round(AROME_station.Wind.sel(time=time).values)
            direction = np.round(AROME_station.Wind_DIR.sel(time=time).values)
            temp = np.round(temperature_station.Tair.sel(time=time).values-273.15)
            cc = 0.3

            date = pd.to_datetime(AROME_station.time.sel(time=time).values)
            year = date.year
            month = date.month
            day = date.day
            hour = date.hour
            minute = date.minute
            elevation_file = prm["windninja_topo"] + f"{station_lower}_2km.tif"
            output_path = prm["windninja_topo"]
            cfg_file = prm["cfg_file"]

            # Launch windninja simulation
            exp = f"{prm['cfg_file']}  " \
                  f"--elevation_file {elevation_file} " \
                  f"--input_speed {speed} " \
                  f"--input_direction {direction} " \
                  f"--uni_air_temp {temp} " \
                  f"--uni_cloud_cover {cc} " \
                  f"--year {year} " \
                  f"--month {month} " \
                  f"--day {day} " \
                  f"--hour {hour} " \
                  f"--minute {minute} " \
                  f"--output_path {output_path}"

            path_to_WindNinja = "C:/WindNinja/WindNinja-3.7.2/bin/"

            # Launch experience
            os.system(path_to_WindNinja + "WindNinja_cli " + exp)

            # Convert outputs from .asc to netcdf
            day_str = f"0{str(day)}" if day < 10 else str(day)
            month_str = f"0{str(month)}" if month < 10 else str(month)
            hour_str = f"0{str(hour)}" if hour < 10 else str(hour)
            minute_str = f"0{str(minute)}" if minute < 10 else str(minute)

            # Convert the predictions to netcdf
            output_str = f"{station_lower}_2km_{np.int(direction)}_{np.int(speed)}_{month_str}-{day_str}-{year}_{hour_str}{minute_str}_30m"
            speed_asc = f"{output_path}{output_str}_vel.asc"
            ang_asc = f"{output_path}{output_str}_ang.asc"
            speed_nc = f"{output_path}{output_str}_vel.nc"
            ang_nc = f"{output_path}{output_str}_ang.nc"
            gdal.Translate(speed_nc, speed_asc)
            gdal.Translate(ang_nc, ang_asc)

            # Read the predictions
            speed_pred = xr.open_dataset(speed_nc)
            ang_pred = xr.open_dataset(ang_nc)

            # Select the nearest neighbor to the station
            list_coord_station = [(x_l93, y_l93)]
            index_nearest_neighbor = find_nearest_neighbor_in_grid(speed_pred.x.values, speed_pred.y.values, list_coord_station, number_of_neighbors=1)
            x_idx = np.intp(index_nearest_neighbor[0])
            y_idx = np.intp(index_nearest_neighbor[1])

            speed_pred = speed_pred.Band1.isel(x=x_idx, y=y_idx).values
            ang_pred = ang_pred.Band1.isel(x=x_idx, y=y_idx).values

            # Save predictions
            times.append(time)
            speeds.append(speed_pred)
            directions.append(ang_pred)

            print(f"\n\n Time: {time} "
                  f"\n\n AROME speed: {speed}"
                  f"\n\n AROME direction: {direction}"
                  f"\n\n AROME temperature: {temp}"
                  f"\n\n WindNinja speed: {speed_pred}"
                  f"\n\n WindNinja direction: {ang_pred}")

            # Delete elevation file and outputs
            del speed_pred
            del ang_pred
            output = os.listdir(output_path)
            for item in output:
                if item.endswith(".asc") or item.endswith(".nc") or item.endswith(".kmz") or item.endswith(".prj"):
                    os.remove(os.path.join(output_path, item))

    # Convert list to DataFrames
    results[station] = pd.DataFrame(np.transpose([speeds, directions]), columns=["Wind", "Wind_DIR"], index=times)

    t1 = t()
    print(f"{(t1-t0)/60} minutes")

    # Save a dictionary containing for each station a .pkl file with time series of the predictions