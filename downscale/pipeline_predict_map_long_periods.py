import pandas as pd
from time import time as t

t_init = t()

"""
1h 50km x 40km
CPU: Downscaling with scipy rotation in 77.09 seconds
GPU: Downscaling with scipy rotation in 28.16 seconds

24h 50km x 40km
GPU: Downscaling scipy in 542.96 seconds (9 min)
By rule of three, this give 2 days and 2h for downscaling one year at 1h and 25m resolution
"""

import xarray as xr
import numpy as np

from downscale.operators.devine import Devine
from visu.visualization import Visualization
from downscale.data_source.MNT import MNT
from downscale.data_source.NWP import NWP
from eval.evaluation import Evaluation
from PRM_predict import create_prm
from downscale.utils.GPU import connect_on_GPU
from downscale.utils.utils_func import round, select_range_7days_for_long_periods_prediction, print_begin_end, \
    print_second_begin_end, print_intro
from downscale.utils.prm import update_selected_path_for_long_periods

# Create prm, horovod and GPU
prm = create_prm(month_prediction=True)
connect_on_GPU(prm)

IGN = MNT(prm=prm)

AROME = NWP(begin=prm["begin"], end=prm["end"], prm=prm)
p = Devine(mnt=IGN, nwp=AROME, prm=prm)

print_begin_end(prm['begin'], prm['end'])

datasets = []
if prm["launch_predictions"]:

    # Iterate on weeks
    begins, ends = select_range_7days_for_long_periods_prediction(begin=prm["begin"], end=prm["end"], prm=prm)
    surfex = xr.open_dataset(prm["path_SURFEX"])

    for index, (begin, end) in enumerate(zip(begins, ends)):
        t1 = t()
        print_second_begin_end(begin, end)

        # Update the name of the file to load
        prm = update_selected_path_for_long_periods(begin, end, prm)

        # Load NWP
        AROME = NWP(begin=str(begin.year) + "-" + str(begin.month) + "-" + str(begin.day),
                    end=str(end.year) + "-" + str(end.month) + "-" + str(end.day),
                    prm=prm)

        # Processing
        p = Processing(mnt=IGN, nwp=AROME, prm=prm)
        wind_xr = p.predict_maps(prm=prm)
        wind_xr = wind_xr.interp_like(surfex, method="linear")
        datasets.append(wind_xr)

        print(f"\n Prediction between {begin} and {end} in {round(t1, t()) / 60} minutes")

    wind_xr = xr.concat(datasets, dim="time")
    wind_xr = p.compute_speed_and_direction_xarray(xarray_data=wind_xr).drop(["U", "V", "W"]).astype(np.float32)
    wind_xr.to_netcdf(prm["path_save_prediction_on_surfex_grid"] + prm["results_name"] + ".nc")

    # Visualization and evaluation
    v = Visualization(p)
    e = Evaluation(v)

    path_to_prm_csv = prm["path_save_prediction_on_surfex_grid"] + "prm_" + prm['results_name'] + ".csv"
    pd.DataFrame.from_dict(data=prm, orient='index').to_csv(path_to_prm_csv, header=False)

    print(f"\n All prediction in {round(t_init, t()) / 60} minutes")
    print_intro()

