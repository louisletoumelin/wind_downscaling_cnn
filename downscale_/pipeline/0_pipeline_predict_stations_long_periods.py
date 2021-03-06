from time import time as t
t_init = t()

"""
1 station and 3 months 
CPU = 1.16 min

22 stations  and 3 months
GPU = 4 min

28.4 min
10 stations and 2.5 years

1 station and all years:
13 min
"""
import numpy as np
#from line_profiler import LineProfiler
from PRM_predict import create_prm, update_selected_path, select_path_to_coord_L93


def round(t1, t2):  return (np.round(t2 - t1, 2))


from downscale.operators.devine import Devine
from downscale.visu.visualization import Visualization
from downscale.data_source.MNT import MNT
from downscale.data_source.NWP import NWP
from downscale.data_source.observation import Observation
from downscale.eval.evaluation import Evaluation
from downscale.utils.utils_func import select_range


# Create prm
prm = create_prm(month_prediction=True)

"""
utils
"""



"""
MNT and NWP
"""


# IGN
IGN = MNT(prm=prm)

# NWP for initialization
prm = update_selected_path(prm, month_prediction=True)
AROME = NWP(begin=prm["begin"],
            end=prm["begin_after"],
            save_path=prm["save_path"],
            path_Z0_2018=None,
            path_Z0_2019=None,
            path_to_coord_L93=prm["path_to_coord_L93"],
            verbose=prm["verbose"],
            load_z0=False,
            save=False)

# BDclim
BDclim = Observation(prm["BDclim_stations_path"],
                     prm["BDclim_data_path"],
                     begin=prm["begin"],
                     end=prm["end"],
                     select_date_time_serie=prm["select_date_time_serie"],
                     path_vallot=prm["path_vallot"],
                     path_saint_sorlin=prm["path_saint_sorlin"],
                     path_argentiere=prm["path_argentiere"],
                     path_Dome_Lac_Blanc=prm["path_Dome_Lac_Blanc"],
                     path_Col_du_Lac_Blanc=prm["path_Col_du_Lac_Blanc"],
                     path_Muzelle_Lac_Blanc=prm["path_Muzelle_Lac_Blanc"],
                     path_Col_de_Porte=prm["path_Col_de_Porte"],
                     path_Col_du_Lautaret=prm["path_Col_du_Lautaret"],
                     GPU=prm["GPU"])

if not(prm["GPU"]):
    number_of_neighbors = 4
    BDclim.update_stations_with_KNN_from_NWP(number_of_neighbors, AROME)
    BDclim.update_stations_with_KNN_from_MNT_using_cKDTree(IGN)

del AROME

"""
Iteration for each month of the considered period
"""
if prm["stations_to_predict"] == 'all':
    prm["stations_to_predict"] = BDclim.stations["name"].values

# Select range
iterator = select_range(prm["month_begin"], prm["month_end"], prm["year_begin"], prm["year_end"], prm["begin"], prm["end"])

results = {}
results["nwp"] = {}
results["cnn"] = {}
results["obs"] = {}

t0 = t()
for index, (day, month, year) in enumerate(iterator):

    print("\n\n Date: \n")
    print(month, year)

    begin = str(year) + "-" + str(month) + "-" + str(1)
    end = str(year) + "-" + str(month) + "-" + str(day)

    prm = update_selected_path(prm, month_prediction=True, year_end=year, month_end=month, day_end=day, force_date=True)
    prm["path_to_coord_L93"] = select_path_to_coord_L93(prm, GPU=prm["GPU"])

    if year == 2018 and (month ==5 or month==6):
        continue

    # Initialize results
    if index == 0:
        for station in prm["stations_to_predict"]:
            results["nwp"][station] = []
            results["cnn"][station] = []
            results["obs"][station] = []

    # AROME
    AROME = NWP(path_to_file=prm["selected_path"], begin=begin, end=end, prm=prm)

    # Processing
    p = Devine(obs=BDclim, mnt=IGN, nwp=AROME, model_path=prm['model_path'], prm=prm)

    # Predictions
    array_xr = p.predict_at_stations(prm["stations_to_predict"], prm=prm)

    # Visualization
    v = Visualization(p)

    # Analysis
    e = Evaluation(v, array_xr)

    # Store nwp, cnn predictions and observations
    for station in prm["stations_to_predict"]:
        nwp, cnn, obs = e._select_dataframe(array_xr, station_name=station,
                                            day=None, month=month, year=year,
                                            variable=prm["variable"],
                                            rolling_mean=None, rolling_window=None)
        results["nwp"][station].append(nwp)
        results["cnn"][station].append(cnn)
        results["obs"][station].append(obs)

    del p
    del v
    del e
    del array_xr
    del AROME
t1 = t()
print(f"\n All prediction in  {round(t0, t1) / 60} minutes")