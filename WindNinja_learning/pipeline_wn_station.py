import numpy as np
import pandas as pd
import xarray as xr
from osgeo import gdal
import os
from time import time as t

from WindNinja_learning.prm_windninja import create_prm
from WindNinja_learning.dem import crop_and_save_dem
from WindNinja_learning.nwp import load_netcdf_and_preprocess, select_time_range_xr, select_station_grid_point_in_NWP
from WindNinja_learning.observations import load_observations, select_idx_station_in_NWP_grid, lower_station_name
from WindNinja_learning.utils_wind_ninja import split_np_datetime, print_current_prediction, delete_temporary_files, \
    reconstruct_datetime, print_with_frame
from WindNinja_learning.wind_ninja_processing import launch_wind_ninja_experiment, asc_to_netcdf, \
    read_speed_and_angle_prediction, extract_nearest_neighbor, remove_case_and_relaunch_simu_at_time_step

"""
momentum solver: 13 minutes for four months at 1 station
"""

# Create a prm file containing the path to data, the considered date, dx and dy of the MNT the options etc
prm = create_prm()

# Load the correct AROME files
AROME = load_netcdf_and_preprocess(prm["AROME_files"])
AROME = select_time_range_xr(AROME, prm["begin"], prm["end"])

# Load AROME cloud cover
# todo load cloud cover

# Load AROME temperature
temperature = load_netcdf_and_preprocess(prm["AROME_files_temperature"])
temperature = select_time_range_xr(temperature, prm["begin"], prm["end"])

# Load observation metadata
stations = load_observations(prm["BDclim_stations_path"])

results = {}

t0 = t()

for station in prm["stations"]:

    x_l93, y_l93 = stations[['X', 'Y']][stations["name"] == station].values[0]
    y_idx_nwp, x_idx_nwp = select_idx_station_in_NWP_grid(stations, station, prm=prm)

    AROME_station = select_station_grid_point_in_NWP(AROME, x_idx_nwp, y_idx_nwp)
    temperature_station = select_station_grid_point_in_NWP(temperature, x_idx_nwp, y_idx_nwp)
    # todo select pixel cloud cover

    # Modify name station
    station_lower = lower_station_name(station)
    prm["_elevation_file"] = prm["windninja_topo"] + f"{station_lower}_2km.tif"

    # Crop MNT around the stations and save it somewhere
    crop_and_save_dem(x_l93 - 1_000, y_l93 + 1_000, x_l93 + 1_000, y_l93 - 1_000,
                      unit="m",
                      name=f"{station_lower}_2km",
                      input_topo=prm["topo_path"],
                      output_dir=prm["windninja_topo"],
                      crs_in=2154,
                      crs_out=2154)

    times = []
    speeds = []
    directions = []

    # Iterate on time
    for index, time in enumerate(AROME_station.time.values):
        if index <= 10_000:

            speed = np.round(AROME_station.Wind.sel(time=time).values)
            if speed > 0:
                direction = np.round(AROME_station.Wind_DIR.sel(time=time).values)
                temp = np.round(temperature_station.Tair.sel(time=time).values - 273.15)
                cc = 0.3  # todo load CC
                date = AROME_station.time.sel(time=time).values

                prm = split_np_datetime(date, prm)
                current_date = reconstruct_datetime(prm)
                print_with_frame(current_date)

                try:
                    launch_wind_ninja_experiment(index, speed, direction, temp, cc, prm)
                    name_speed_nc, name_ang_nc = asc_to_netcdf(speed, direction, station_lower, prm)
                except ValueError:
                    print("\nValueError encountered."
                          "\nTrying to relaunch simulation without using existing case\n")
                    remove_case_and_relaunch_simu_at_time_step(index, prm, speed, direction, temp, cc, station_lower)

                speed_pred, ang_pred = read_speed_and_angle_prediction(name_speed_nc, name_ang_nc)

                speed_pred, ang_pred = extract_nearest_neighbor(speed_pred, ang_pred, x_l93, y_l93)

                # Save predictions
                times.append(time)
                speeds.append(speed_pred)
                directions.append(ang_pred)

                print_current_prediction(time, speed, direction, temp, speed_pred, ang_pred)

                # Delete elevation file and outputs
                del speed_pred
                del ang_pred
                delete_temporary_files(prm)
            else:
                print(f"\n\nSpeed=0 at time {time}\n\n")
                times.append(time)
                speeds.append(0)
                directions.append(0)
                delete_temporary_files(prm)

    # Convert list to DataFrames
    results[station] = pd.DataFrame(np.transpose([speeds, directions]), columns=["Wind", "Wind_DIR"], index=times)

    t1 = t()
    print(f"{(t1 - t0) / 60} minutes")

    # Save a dictionary containing for each station a .pkl file with time series of the predictions
