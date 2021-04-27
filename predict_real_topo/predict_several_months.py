from time import time as t
t_init = t()

import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from line_profiler import LineProfiler

def round(t1, t2):  return (np.round(t2 - t1, 2))


from Processing import Processing
from Visualization import Visualization
from MNT import MNT
from NWP import NWP
from Observation import Observation
from Data_2D import Data_2D
from MidpointNormalize import MidpointNormalize
from Evaluation import Evaluation


GPU = False
Z0 = True
load_z0 = True
save_z0 = False
peak_valley = True

# Data path
data_path = "C:/Users/louis/git/wind_downscaling_CNN/Data/1_Raw/"
experience_path = "C:/Users/louis/git/wind_downscaling_CNN/Models/ARPS/"
topo_path = data_path + "MNT/IGN_25m/ign_L93_25m_alpesIG.tif"

# NWP
AROME_path_1 = data_path + "AROME/FORCING_alp_2017080106_2018080106.nc"
AROME_path_2 = data_path + "AROME/FORCING_alp_2018080106_2019060106.nc"
AROME_path_3 = data_path + "AROME/FORCING_alp_2019060107_2019070106.nc"
AROME_path = [AROME_path_1, AROME_path_2, AROME_path_3]

# Z0
path_Z0_2018 = data_path + "AROME/Z0/Z0_alp_2018010100_2018120700.nc"
path_Z0_2019 = data_path + "AROME/Z0/Z0_alp_20190103_20191227.nc"
save_path = data_path + "AROME/Z0/"

# Observation
BDclim_stations_path = data_path + "BDclim/precise_localisation/liste_postes_alps_l93.csv"
BDclim_data_path = data_path + "BDclim/extract_BDClim_et_sta_alp_20171101_20190501.csv"
path_vallot = data_path + "BDclim/Vallot/"

# CNN model
model_experience = "date_16_02_name_simu_FINAL_1_0_model_UNet/"
model_path = experience_path + model_experience

# IGN
IGN = MNT(topo_path, "IGN")

# Date to predict
day_begin = 1
month_begin = 9
year_begin = 2018

day_end = 31
month_end = 12
year_end = 2018

date_begin = str(year_begin) + "-" + str(month_begin) + "-" + str(day_begin)
date_end = str(year_end) + "-" + str(month_end) + "-" + str(day_end)

if (month_end != month_begin) or (year_begin != year_end):
    dates = pd.date_range(date_begin, date_end, freq='M')
    iterator = zip(dates.day, dates.month, dates.year)
else:
    dates = pd.to_datetime(date_end)
    iterator = zip([dates.day], [dates.month], [dates.year])


results = {}
results["nwp"] = {}
results["cnn"] = {}
results["obs"] = {}

for index, (day, month, year) in enumerate(iterator):
    print(month, year)
    begin = str(year) + "-" + str(month) + "-" + str(1)
    end = str(year) + "-" + str(month) + "-" + str(day)

    current_date = datetime.datetime(year, month, day)
    d1 = datetime.datetime(2017, 8, 1, 6)
    d2 = datetime.datetime(2018, 8, 1, 6)
    d3 = datetime.datetime(2019, 6, 1, 6)
    d4 = datetime.datetime(2019, 6, 1, 7)
    d5 = datetime.datetime(2019, 7, 1, 6)

    if d1 < current_date < d2:
        selected_path = AROME_path_1
    if d2 < current_date < d3:
        selected_path = AROME_path_2
    if d4 < current_date < d5:
        selected_path = AROME_path_3

    # AROME
    AROME = NWP(selected_path, "AROME", begin, end, save_path=save_path, path_Z0_2018=path_Z0_2018, path_Z0_2019=path_Z0_2019, verbose=True, load_z0=True, save=False)

    # BDclim
    BDclim = Observation(BDclim_stations_path, BDclim_data_path, path_vallot, begin=begin, end=end, select_date_time_serie=False, vallot=True)
    number_of_neighbors = 4
    BDclim.update_stations_with_KNN_from_NWP(number_of_neighbors, AROME)
    BDclim.update_stations_with_KNN_from_MNT_using_cKDTree(IGN)

    p = Processing(BDclim, IGN, AROME, model_path, GPU=GPU, data_path=data_path)

    stations = BDclim.stations["name"]
    if index == 0:
        for station in stations:
            results["nwp"][station] = []
            results["cnn"][station] = []
            results["obs"][station] = []
    array_xr = p.predict_UV_with_CNN(stations,
                                     verbose=True,
                                     Z0_cond=Z0,
                                     peak_valley=peak_valley)
    v = Visualization(p)
    e = Evaluation(v, array_xr)

    for station in stations:
        nwp, cnn, obs = e._select_dataframe(array_xr, station_name=station,
                                            day=None, month=month, year=year,
                                            variable='UV',
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