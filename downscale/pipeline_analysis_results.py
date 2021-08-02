import numpy as np
import pandas as pd
import pickle

try:
    import seaborn as sns
    sns.set_style("white")
    sns.set_context('paper', font_scale=1.4)
except ImportError:
    pass

from downscale.Analysis.Visualization import Visualization
from downscale.Operators.Processing import Processing
from downscale.Analysis.Evaluation import EvaluationFromDict
from downscale.Data_family.MNT import MNT
from downscale.Data_family.NWP import NWP
from downscale.Data_family.Observation import Observation
from PRM_predict import create_prm
from downscale.Utils.GPU import connect_GPU_to_horovod

prm = create_prm(month_prediction=True)
connect_GPU_to_horovod() if prm["GPU"] else None

IGN = MNT(prm["topo_path"], name="IGN")
AROME = NWP(prm["AROME_path_1"], name="AROME", begin=prm["begin"], end=prm["begin_after"], prm=prm)
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)

with open('../../Data/3_Predictions/results.pickle', 'rb') as handle:
    results = pickle.load(handle)

p = Processing(obs=BDclim, mnt=IGN, nwp=AROME, model_path=prm['model_path'], prm=prm)
p.update_stations_with_neighbors(mnt=IGN, nwp=AROME, GPU=prm["GPU"], number_of_neighbors=4, interpolated=False)

v = Visualization(p)
e = EvaluationFromDict(v)

# Boxplots
metrics = ["abs_error", "bias", "abs_error_rel", "bias_rel"]
#e.plot_metric_all_stations(results, metrics=metrics)

# 1-1 plot with density
#e.plot_1_1_density(results, cmap="viridis")

# Wind speed distribution for CNN, NWP and observation
#e.plot_distribution_all_stations(results)

# todo implement this function
e.plot_heatmap(results, metrics=metrics)
