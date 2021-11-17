import numpy as np
import xarray as xr
import gdal
import os

from WindNinja_learning.dem import find_nearest_neighbor_in_grid


def launch_wind_ninja_experiment(speed, direction, temp, cc, prm):
    # Launch windninja simulation
    exp = f"{prm['cfg_file']}  " \
          f"--elevation_file {prm['_elevation_file']} " \
          f"--input_speed {speed} " \
          f"--input_direction {direction} " \
          f"--uni_air_temp {temp} " \
          f"--uni_cloud_cover {cc} " \
          f"--year {prm['_year']} " \
          f"--month {prm['_month']} " \
          f"--day {prm['_day']} " \
          f"--hour {prm['_hour']} " \
          f"--minute {prm['_minute']} " \
          f"--output_path {prm['output_path']}"

    # Launch experience
    os.system(prm["path_to_WindNinja"] + "WindNinja_cli " + exp)


def asc_to_netcdf(speed, direction, station_lower, prm):
    # Convert outputs from .asc to netcdf
    day_str = f"0{str(prm['_day'])}" if prm['_day'] < 10 else str(prm['_day'])
    month_str = f"0{str(prm['_month'])}" if prm['_month'] < 10 else str(prm['_month'])
    hour_str = f"0{str(prm['_hour'])}" if prm['_hour'] < 10 else str(prm['_hour'])
    minute_str = f"0{str(prm['_minute'])}" if prm['_minute'] < 10 else str(prm['_minute'])

    # Convert the predictions to netcdf
    speed = np.intp(speed)
    direction = np.intp(direction)
    output_str = f"{station_lower}_2km_{direction}_{speed}_{month_str}-{day_str}-{prm['_year']}_{hour_str}{minute_str}_30m"
    name_speed_asc = f"{prm['output_path']}{output_str}_vel.asc"
    name_ang_asc = f"{prm['output_path']}{output_str}_ang.asc"
    name_speed_nc = f"{prm['output_path']}{output_str}_vel.nc"
    name_ang_nc = f"{prm['output_path']}{output_str}_ang.nc"
    gdal.Translate(name_speed_nc, name_speed_asc)
    gdal.Translate(name_ang_nc, name_ang_asc)

    return name_speed_nc, name_ang_nc


def read_speed_and_angle_prediction(name_speed_nc, name_ang_nc):
    speed_pred = xr.open_dataset(name_speed_nc).Band1
    ang_pred = xr.open_dataset(name_ang_nc).Band1
    return speed_pred, ang_pred


def extract_nearest_neighbor(speed_pred, ang_pred, x_l93, y_l93):
    # Select the nearest neighbor to the station
    index_nearest_neighbor = find_nearest_neighbor_in_grid(speed_pred.x.values, speed_pred.y.values, [(x_l93, y_l93)],
                                                           number_of_neighbors=1)
    x_idx = np.intp(index_nearest_neighbor[0])
    y_idx = np.intp(index_nearest_neighbor[1])

    speed_pred = speed_pred.isel(x=x_idx, y=y_idx).values
    ang_pred = ang_pred.isel(x=x_idx, y=y_idx).values

    return speed_pred, ang_pred
