import pickle
import uuid

import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_style("white")
    sns.set_context('paper', font_scale=1.4)
except ImportError:
    pass

from downscale.visu.visualization import Visualization
from downscale.operators.devine import Devine
from downscale.eval.eval_from_dict import EvaluationFromDict
from downscale.data_source.MNT import MNT
from downscale.data_source.NWP import NWP
from downscale.data_source.observation import Observation
from PRM_predict import create_prm

prm = create_prm(month_prediction=True)

IGN = MNT(prm=prm)
AROME = NWP(prm["AROME_path_1"], begin=prm["begin"], end=prm["begin_after"], prm=prm)
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)

# Quality control
BDclim.replace_obs_by_QC_obs(prm, replace_old_wind=True)
use_QC = True
QC = BDclim.time_series

with open(prm["working_directory"]+'Data/3_Predictions/Stations/prediction_classic_stations_21_12_2021.pickle', 'rb') as handle:
    results_1 = pickle.load(handle)
"""
with open(prm["working_directory"]+'Data/3_Predictions/Stations/prediction_no_dropout_stations_21_12_2021.pickle', 'rb') as handle:
    results_2 = pickle.load(handle)

with open('../../Data/3_Predictions/linear_activation_norm_each_topography.pickle', 'rb') as handle:
    results_3 = pickle.load(handle)

with open('../../Data/3_Predictions/not_get_closer_and_linear_and_exposed.pickle', 'rb') as handle:
    results_4 = pickle.load(handle)
"""
results_list = [results_1]
title = ["With dropout"]

p = Devine(obs=BDclim, mnt=IGN, nwp=AROME, prm=prm)
p.update_stations_with_neighbors(mnt=IGN, nwp=AROME, GPU=prm["GPU"], number_of_neighbors=4, interpolated=False)

v = Visualization(p)
e = EvaluationFromDict(v)

variable = "UV"
df_results = pd.DataFrame()
all_predictions = []
for index, result in enumerate(results_list):

    print(f"Result: {title[index]}")

    # Integrated results
    cnn, nwp, obs = e.results_to_three_arrays(result, variable=variable, use_QC=True, time_series_qc=QC)
    all_predictions.append(cnn)
    print("\nRMSE CNN", e.RMSE(cnn, obs))
    print("\nRMSE NWP", e.RMSE(nwp, obs))
    print("\nBias CNN", e.mean_bias(cnn, obs))
    print("\nBias NWP", e.mean_bias(nwp, obs))

    cnn, nwp, obs = e.results_to_three_arrays(result, variable=variable, use_QC=True, time_series_qc=QC[QC["name"] != "Vallot"])
    all_predictions.append(cnn)
    print("\nRMSE CNN", e.RMSE(cnn, obs))
    print("\nRMSE NWP", e.RMSE(nwp, obs))
    print("\nBias CNN", e.mean_bias(cnn, obs))
    print("\nBias NWP", e.mean_bias(nwp, obs))

    # Create a dataFrame with results
    df_results[f"bias_CNN_{index}"] = e.bias(cnn, obs)
    if index == 0:
        df_results[f"bias_NWP_{index}"] = e.bias(nwp, obs)

# Boxplot
df_results = df_results.melt(var_name="model", value_name="values")
#plt.figure()
#sns.boxplot(data=df_results, y="values", x="model")
plt.figure()
sns.boxplot(data=df_results, y="values", x="model", showfliers=True)
plt.xticks([0, 1, 2], ['with dropout', 'AROME', 'Without dropout'], rotation=20)
plt.tight_layout()
ax = plt.gca()
fig = ax.get_figure()
save_path = prm["save_figure_path"]
fig.savefig(save_path + f"fliers_boxplot_compare_models_{str(uuid.uuid4())[:4]}.png")
