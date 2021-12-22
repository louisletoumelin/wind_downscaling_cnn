from time import time as t
import pandas as pd

t_init = t()

# Create prm
from PRM_predict import create_prm
prm = create_prm(month_prediction=True)

from downscale.data_source.observation import Observation
from downscale.utils.utils_func import round

# BDclim
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)
print(BDclim.time_series["name"].unique())

# Quality control
if prm["launch_predictions"]:
    BDclim.qc(prm)
    BDclim.time_series.to_pickle(prm["QC_pkl"])
else:
    BDclim.time_series = pd.read_pickle(prm["QC_pkl"])

print(f"\n All prediction in  {round(t_init, t()) / 60} minutes")
