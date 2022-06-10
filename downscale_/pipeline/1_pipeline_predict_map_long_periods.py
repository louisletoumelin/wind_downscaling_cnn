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

import numpy as np
import pandas as pd
import xarray as xr

# Create prm
from PRM_predict import create_prm
prm = create_prm(month_prediction=True)

from downscale.operators.devine import Devine
from downscale.visu.visualization import Visualization
from downscale.data_source.MNT import MNT
from downscale.data_source.NWP import NWP
from downscale.eval.evaluation import Evaluation
from downscale.utils.utils_func import round, select_range_7days_for_long_periods_prediction, print_begin_end, \
    print_second_begin_end, print_intro
from utils_prm import *


#hours_begin = [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
#day_begin = [[2, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
#month_begin = [[8, 11, 2, 5, 8], [11, 2, 5, 8, 11], [2, 5, 8, 11, 2]]
#year_begin = [[2017, 2017, 2018, 2018, 2018], [2018, 2019, 2019, 2019, 2019], [2020, 2020, 2020, 2020, 2021]]

#hours_end = [[23, 23, 23, 23, 23], [23, 23, 23, 23, 23], [23, 23, 23, 23, 1]]
#day_end = [[31, 31, 30, 31, 31], [31, 30, 31, 31, 31], [30, 31, 31, 31, 31]]
#month_end = [[10, 1, 4, 7, 10], [1, 4, 7, 10, 1], [4, 7, 10, 1, 5]]
#year_end = [[2017, 2018, 2018, 2018, 2018], [2019, 2019, 2019, 2019, 2020], [2020, 2020, 2020, 2021, 2021]]

hours_begin = [[0]]  #, [0, 0, 0, 0]]
day_begin = [[1]]  #, [1, 1, 1, 1]]
month_begin = [[11]] #, [5, 8, 11, 2]]
year_begin = [[2019]] #, [2020, 2020, 2020, 2021]]

hours_end = [[23]] #, [23, 23, 23, 1]]
day_end = [[31]] #, [31, 31, 31, 31]]
month_end = [[12]] #, [7, 10, 1, 5]]
year_end = [[2019]] #, [2020, 2020, 2021, 2021]]


id_date = 0

for h0, d0, m0, y0, h1, d1, m1, y1 in zip(hours_begin[id_date], day_begin[id_date], month_begin[id_date], year_begin[id_date], hours_end[id_date], day_end[id_date], month_end[id_date], year_end[id_date]):
    prm["hour_begin"] = h0  # 1
    prm["day_begin"] = d0  # 2
    prm["month_begin"] = m0  # 8
    prm["year_begin"] = y0  # 2017

    # 31 May 2020 1h
    prm["hour_end"] = h1  # 1
    prm["day_end"] = d1  # 31
    prm["month_end"] = m1  # 5
    prm["year_end"] = y1  # 2020

    prm = create_begin_and_end_str(prm)

    prm["results_name"] = f"03_05_2022_Ange_{d0}_{m0}_{y0}_to_{d1}_{m1}_{y1}"

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
            p = Devine(mnt=IGN, nwp=AROME, prm=prm)
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

