from time import time as t
import numpy as np
import pandas as pd

t_init = t()

from downscale.Data_family.Observation import Observation
from downscale.Analysis.Visualization import Visualization
from downscale.Operators.Processing import Processing
from downscale.Analysis.Evaluation import Evaluation
from PRM_predict import create_prm
from downscale.Utils.Utils import round
from downscale.Utils.GPU import connect_GPU_to_horovod

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
p = Processing(obs=BDclim, model_path=prm['model_path'], prm=prm)

# Visualization
v = Visualization(p)

# Analysis
e = Evaluation(v)

print(f"\n All prediction in  {round(t_init, t()) / 60} minutes")