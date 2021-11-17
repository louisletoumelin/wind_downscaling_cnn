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


def print_current_prediction(time, speed, direction, temp, speed_pred, ang_pred):
    print(f"\n\n Time: {time} "
          f"\n\n AROME speed: {speed}"
          f"\n\n AROME direction: {direction}"
          f"\n\n AROME temperature: {temp}"
          f"\n\n WindNinja speed: {speed_pred}"
          f"\n\n WindNinja direction: {ang_pred}")


def delete_temporary_files(prm):
    output = os.listdir(prm["output_path"])
    for item in output:
        if item.endswith(".asc") or item.endswith(".nc") or item.endswith(".kmz") or item.endswith(".prj"):
            os.remove(os.path.join(prm["output_path"], item))
