from time import time as t
import numpy as np
import pandas as pd

t_init = t()

def round_time(t1, t2):  return (np.round(t2 - t1, 2))

from Observation import Observation
from Visualization import Visualization
from Processing import Processing
from Evaluation import Evaluation
from PRM_predict import create_prm, update_selected_path
from Utils import connect_GPU_to_horovod, select_range

# Create prm
prm = create_prm(month_prediction=True)

# Initialize horovod and GPU
if prm["GPU"]: connect_GPU_to_horovod()

# BDclim
BDclim = Observation(prm["BDclim_stations_path"],
                     prm["BDclim_data_path"],
                     begin=prm["begin"],
                     end=prm["end"],
                     select_date_time_serie=prm["select_date_time_serie"],
                     GPU=prm["GPU"],
                     path_vallot=prm["path_vallot"],
                     path_saint_sorlin=prm["path_saint_sorlin"],
                     path_argentiere=prm["path_argentiere"],
                     path_Dome_Lac_Blanc=prm["path_Dome_Lac_Blanc"],
                     path_Col_du_Lac_Blanc=prm["path_Col_du_Lac_Blanc"],
                     path_Muzelle_Lac_Blanc=prm["path_Muzelle_Lac_Blanc"],
                     path_Col_de_Porte=prm["path_Col_de_Porte"],
                     path_Col_du_Lautaret=prm["path_Col_du_Lautaret"])

print(BDclim.time_series["name"].unique())

# Quality control
if prm["launch_predictions"]:
    BDclim.qc()
    BDclim.time_series.to_pickle(prm["QC_pkl"])
else:
    BDclim.time_series = pd.read_pickle(prm["QC_pkl"])

print(BDclim.time_series["name"].unique())

# Processing
p = Processing(obs=BDclim,
               model_path=prm['model_path'],
               GPU=prm["GPU"],
               data_path=prm['data_path'])

# Visualization
v = Visualization(p)

# Evaluation
e = Evaluation(v)

t_end = t()
print(f"\n All prediction in  {round_time(t_init, t_end) / 60} minutes")