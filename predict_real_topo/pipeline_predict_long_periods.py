from time import time as t
t_init = t()

import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from line_profiler import LineProfiler
from PRM_predict import create_prm, update_selected_path
from Utils import connect_GPU_to_horovod, select_range

def round(t1, t2):  return (np.round(t2 - t1, 2))


from Processing import Processing
from Visualization import Visualization
from MNT import MNT
from NWP import NWP
from Observation import Observation
from Data_2D import Data_2D
from MidpointNormalize import MidpointNormalize
from Evaluation import Evaluation
from Utils import connect_GPU_to_horovod, select_range

"""
Simulation parameters
"""

GPU = False
Z0 = True
load_z0 = True
save_z0 = False
peak_valley = True
variable = 'UV'

day_begin = 1
month_begin = 9
year_begin = 2018

day_end = 30
month_end = 9
year_end = 2018

date_begin = str(year_begin) + "-" + str(month_begin) + "-" + str(day_begin)
date_begin_after = str(year_begin) + "-" + str(month_begin) + "-" + str(day_begin+1)
date_end = str(year_end) + "-" + str(month_end) + "-" + str(day_end)

"""
Utils
"""

# Create prm
prm = create_prm(GPU, month_prediction=False)

"""
MNT and NWP
"""

# IGN
IGN = MNT(prm["topo_path"], name="IGN")

# NWP for initialization
prm = update_selected_path(year_begin, month_begin, day_begin, prm)
AROME = NWP(prm["selected_path"], name="AROME", begin=date_begin, end=date_begin_after,
                save_path=prm["save_path"], path_Z0_2018=prm["path_Z0_2018"], path_Z0_2019=prm["path_Z0_2019"],
                verbose=True, load_z0=load_z0, save=save_z0)

# BDclim
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"],
                     begin=date_begin, end=date_end, select_date_time_serie=False,
                     path_vallot=prm["path_vallot"],
                     path_saint_sorlin=prm["path_saint_sorlin"],
                     path_argentiere=prm["path_argentiere"])

if not(GPU):
    number_of_neighbors = 4
    BDclim.update_stations_with_KNN_from_NWP(number_of_neighbors, AROME)
    BDclim.update_stations_with_KNN_from_MNT_using_cKDTree(IGN)

del AROME

"""
Iteration for each month of the considered period
"""

# Select range
iterator = select_range(month_begin, month_end, year_begin, year_end, date_begin, date_end)

results = {}
results["nwp"] = {}
results["cnn"] = {}
results["obs"] = {}


for index, (day, month, year) in enumerate(iterator):

    begin = str(year) + "-" + str(month) + "-" + str(1)
    end = str(year) + "-" + str(month) + "-" + str(day)
    prm = update_selected_path(year, month, day, prm)

    # AROME
    AROME = NWP(prm["selected_path"], name="AROME", begin=begin, end=end,
                save_path=prm["save_path"], path_Z0_2018=prm["path_Z0_2018"], path_Z0_2019=prm["path_Z0_2019"],
                verbose=True, load_z0=load_z0, save=save_z0)

    # Processing
    p = Processing(obs=BDclim,
                   mnt=IGN,
                   nwp=AROME,
                   model_path=prm['model_path'],
                   GPU=GPU,
                   data_path=prm['data_path'])

    # Initialize results
    stations = BDclim.stations["name"]
    if index == 0:
        for station in stations:
            results["nwp"][station] = []
            results["cnn"][station] = []
            results["obs"][station] = []

    # Predictions
    array_xr = p.predict_UV_with_CNN(stations,
                                     verbose=True,
                                     Z0_cond=Z0,
                                     peak_valley=peak_valley)
    # Visualization
    v = Visualization(p)

    # Evaluation
    e = Evaluation(v, array_xr)

    # Store nwp, cnn predictions and observations
    for station in stations:
        nwp, cnn, obs = e._select_dataframe(array_xr, station_name=station,
                                            day=None, month=month, year=year,
                                            variable=variable,
                                            rolling_mean=None, rolling_window=None)
        results["nwp"][station].append(nwp)
        results["cnn"][station].append(cnn)
        results["obs"][station].append(obs)

    del p
    del v
    del e
    del array_xr
    del AROME
    del BDclim