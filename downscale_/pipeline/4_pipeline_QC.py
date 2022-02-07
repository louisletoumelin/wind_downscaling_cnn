from time import time as t
import pandas as pd

t_init = t()

# Create prm
from PRM_predict import create_prm
prm = create_prm(month_prediction=True)
"""
from downscale.data_source.observation import Observation
from downscale.data_source.MNT import MNT
from downscale.data_source.NWP import NWP
from downscale.operators.devine import Devine
from downscale.visu.visualization import Visualization
from downscale.eval.evaluation import Evaluation

from downscale.utils.utils_func import round

# BDclim
IGN = MNT(prm=prm)
AROME = NWP(begin=prm["begin"], end=prm["end"], prm=prm)
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)
print(BDclim.time_series["name"].unique())
p = Devine(mnt=IGN, nwp=AROME, prm=prm)
v = Visualization(p)
e = Evaluation(v)

# Quality control
if prm["launch_predictions"]:
    BDclim.qc(prm)
    BDclim.time_series.to_pickle(prm["QC_pkl"])
else:
    BDclim.time_series = pd.read_pickle(prm["QC_pkl"])

print(f"\n All prediction in  {round(t_init, t()) / 60} minutes")
"""

import tensorflow
print(tensorflow.data.experimental.SqlDataset)