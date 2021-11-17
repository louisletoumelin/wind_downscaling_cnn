import pandas as pd
import os


def split_np_datetime(date, prm):

    date = pd.to_datetime(date)
    prm["_year"] = date.year
    prm["_month"] = date.month
    prm["_day"] = date.day
    prm["_hour"] = date.hour
    prm["_minute"] = date.minute

    return prm


def reconstruct_datetime(prm):
    date = prm["_year"] + '-' + prm["_month"] + '-' + prm["_month"] + '-' + prm["_day"] + '_' + prm["_hour"]
    return date


def print_current_prediction(time, speed, direction, temp, speed_pred, ang_pred):
    print(f"\n Time: {time} "
          f"\n AROME speed: {speed}"
          f"\n AROME direction: {direction}"
          f"\n AROME temperature: {temp}"
          f"\n WindNinja speed: {speed_pred}"
          f"\n WindNinja direction: {ang_pred}\n")


def delete_temporary_files(prm):
    output = os.listdir(prm["output_path"])
    for item in output:
        if item.endswith(".asc") or item.endswith(".nc") or item.endswith(".kmz") or item.endswith(".prj"):
            os.remove(os.path.join(prm["output_path"], item))


def print_begin_end(begin, end):
    print('\n\n__________________________')
    print('__________________________\n')
    print(f'_______{begin}___________\n')
    print(f'_______{end}___________\n')
    print('__________________________')
    print('__________________________\n\n')


def print_with_frame(text):
    print('\n\n__________________________')
    print('__________________________\n')
    print(f'_______{text}_______\n')
    print('__________________________')
    print('__________________________\n\n')