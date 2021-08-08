import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

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
"""
with open('../../Data/3_Predictions/results_linear.pickle', 'rb') as handle:
    results_0 = pickle.load(handle)

with open('../../Data/3_Predictions/linear_activation_norm_each_topography.pickle', 'rb') as handle:
    results_1 = pickle.load(handle)

with open('../../Data/3_Predictions/linear_activation_norm_each_topography_add_topography.pickle', 'rb') as handle:
    results_2 = pickle.load(handle)

with open('../../Data/3_Predictions/results_Arctan_30_1.pickle', 'rb') as handle:
    results_3 = pickle.load(handle)

with open('../../Data/3_Predictions/results_Arctan_30_2.pickle', 'rb') as handle:
    results_4 = pickle.load(handle)

with open('../../Data/3_Predictions/correct_norm_arctan_30_1.pickle', 'rb') as handle:
    results_5 = pickle.load(handle)

with open('../../Data/3_Predictions/correct_norm_arctan_30_2.pickle', 'rb') as handle:
    results_6 = pickle.load(handle)

results_list = [results_0, results_1, results_2, results_3, results_4, results_5, results_6]

p = Processing(obs=BDclim, mnt=IGN, nwp=AROME, model_path=prm['model_path'], prm=prm)
p.update_stations_with_neighbors(mnt=IGN, nwp=AROME, GPU=prm["GPU"], number_of_neighbors=4, interpolated=False)

v = Visualization(p)
e = EvaluationFromDict(v)

variable = "UV"
df_results = pd.DataFrame()
all_predictions = []
for index, result in enumerate(results_list):

    # Integrated results
    cnn, nwp, obs = e.create_three_df_results(result, variable=variable)
    all_predictions.append(cnn)
    print("\nRMSE CNN", e.RMSE(cnn, obs))
    print("\nRMSE NWP", e.RMSE(nwp, obs))
    print("\nBias CNN", e.mean_bias(cnn, obs))
    print("\nBias NWP", e.mean_bias(nwp, obs))

    # 1-1 density plots
    e.plot_1_1_density(result, variables=[variable])

    # Create a dataFrame with results
    df_results[f"bias_CNN_{index}"] = e.mean_bias(cnn, obs)
    df_results[f"bias_NWP_{index}"] = e.mean_bias(cnn, obs)

# Boxplot
df_results = df_results.melt(var_name="model", value_name="values")
ax = plt.gca()
sns.boxplot(data=df_results, y="values", x="model")
sns.boxplot(data=df_results, y="values", x="model", showfliers=False)

# 1-1 plot without density
#plt.figure()
#plt.scatter(all_predictions[0], all_predictions[1])
#plt.plot(all_predictions[0], all_predictions[0], color='red')
#xmax = np.nanmax((all_predictions[0], all_predictions[1]))
#xmin = np.nanmin((all_predictions[0], all_predictions[1]))
#plt.xlim(xmin, xmax)
#plt.ylim(xmin, xmax)
#plt.axis("square")

"""