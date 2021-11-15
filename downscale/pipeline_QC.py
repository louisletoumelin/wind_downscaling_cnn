from time import time as t
import pandas as pd

t_init = t()

from downscale.data_source.observation import Observation
from visu.visualization import Visualization
from downscale.operators.devine import Devine
from eval.evaluation import Evaluation
from PRM_predict import create_prm
from downscale.utils.utils_func import round
from downscale.utils.GPU import connect_GPU_to_horovod

# Create prm
prm = create_prm(month_prediction=True)

# Initialize horovod and GPU
connect_GPU_to_horovod() if prm["GPU"] else None

# BDclim
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)
print(BDclim.time_series["name"].unique())

# Quality control
if prm["launch_predictions"]:
    BDclim.qc()
    BDclim.time_series.to_pickle(prm["QC_pkl"])
else:
    BDclim.time_series = pd.read_pickle(prm["QC_pkl"])

print(BDclim.time_series["name"].unique())

# Processing
p = Devine(obs=BDclim, prm=prm)

# Visualization
v = Visualization(p)

# Analysis
e = Evaluation(v)

print(f"\n All prediction in  {round(t_init, t()) / 60} minutes")