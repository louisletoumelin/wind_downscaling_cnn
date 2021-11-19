import numpy as np
import xarray as xr
import gdal
import os

from WindNinja_learning.dem import find_nearest_neighbor_in_grid


def detect_existing_case(index, path):

    if index > 0:
        for element in os.listdir(path):
            if "NINJAFOAM" in element:
                return element
            else:
                return None


def launch_wind_ninja_experiment(index, speed, direction, temp, cc, prm):
    # Launch windninja simulation

    exp = f"{prm['cfg_file']}  " \
          f"--elevation_file {prm['_elevation_file']} " \
          f"--input_speed {speed} " \
          f"--input_direction {direction} " \
          f"--mesh_resolution 30 " \
          f"--units_mesh_resolution m " \
          f"--uni_air_temp {temp} " \
          f"--uni_cloud_cover {cc} " \
          f"--year {prm['_year']} " \
          f"--month {prm['_month']} " \
          f"--day {prm['_day']} " \
          f"--hour {prm['_hour']} " \
          f"--minute {prm['_minute']} " \
          f"--output_path {prm['output_path']}"

    # Use existing case if it exists
    case = detect_existing_case(index, prm["output_path"])

    if case is not None:

        exp = f"{prm['cfg_file']}  " \
              f"--elevation_file {prm['_elevation_file']} " \
              f"--existing_case_directory {prm['output_path']+case} " \
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


def _float_to_string_for_dates(day_float):
    day_str = f"0{str(day_float)}" if day_float < 10 else str(day_float)
    return day_str


def _create_filename(path, filename, format_file):
    name_speed = f"{path}{filename}_vel.{format_file}"
    name_ang = f"{path}{filename}_vel.{format_file}"
    return name_speed, name_ang


def asc_to_netcdf(speed, direction, station_lower, prm):
    # Convert outputs from .asc to netcdf
    dates = [prm['_day'], prm['_month'], prm['_hour'], prm['_minute']]
    day_str, month_str, hour_str, min_str = [_float_to_string_for_dates(date) for date in dates]

    # Convert the predictions to netcdf
    speed = np.intp(speed)
    direction = np.intp(direction)
    output_str = f"{station_lower}_2km_{direction}_{speed}_{month_str}-{day_str}-{prm['_year']}_{hour_str}{min_str}_30m"
    name_speed_asc, name_ang_asc = _create_filename(prm['output_path'], output_str, "asc")
    name_speed_nc, name_ang_nc = _create_filename(prm['output_path'], output_str, "nc")

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


def remove_case_and_relaunch_simu_at_time_step(index, prm, speed, direction, temp, cc, station_lower):
    import shutil
    case = detect_existing_case(index, prm["output_path"])
    if case is not None:
        shutil.rmtree(prm["output_path"] + case, ignore_errors=True)
        launch_wind_ninja_experiment(0, speed, direction, temp, cc, prm)
        name_speed_nc, name_ang_nc = asc_to_netcdf(speed, direction, station_lower, prm)
    else:
        raise
