from time import time as t
import uuid

import pandas as pd
import matplotlib
#matplotlib.use('Agg')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

t_init = t()

# Create prm
from PRM_predict import create_prm

prm = create_prm(month_prediction=True)

from downscale.data_source.observation import Observation
from downscale.data_source.MNT import MNT
from downscale.data_source.NWP import NWP
from downscale.operators.devine import Devine
from downscale.visu.visualization import Visualization
from downscale.eval.evaluation import Evaluation

# BDclim
IGN = MNT(prm=prm)
AROME = NWP(begin=prm["begin"], end=prm["end"], prm=prm)
#BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)
#BDclim.time_series = pd.read_pickle(prm["QC_pkl"])
#p = Devine(mnt=IGN, nwp=AROME, obs=BDclim, prm=prm)
#v = Visualization(p)
#e = Evaluation(v)
