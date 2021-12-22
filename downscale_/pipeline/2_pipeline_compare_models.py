import pandas as pd
import pickle
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

BDclim.replace_obs_by_QC_obs(prm)



with open('../../Data/3_Predictions/wind_exposed_at_max_elevation_Arctan_30_2.pickle', 'rb') as handle:
    results_1 = pickle.load(handle)

with open('../../Data/3_Predictions/unexposed_wind_not_activated_Arctan_30_2.pickle', 'rb') as handle:
    results_2 = pickle.load(handle)
"""
with open('../../Data/3_Predictions/linear_activation_norm_each_topography.pickle', 'rb') as handle:
    results_3 = pickle.load(handle)

with open('../../Data/3_Predictions/not_get_closer_and_linear_and_exposed.pickle', 'rb') as handle:
    results_4 = pickle.load(handle)
"""
results_list = [results_1, results_2]
title = ["Best predictions", "Not exposing wind", "Reference"]

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
    cnn, nwp, obs = e.results_to_three_arrays(result, variable=variable, use_QC=True, time_series_qc=BDclim.time_series)
    all_predictions.append(cnn)
    print("\nRMSE CNN", e.RMSE(cnn, obs))
    print("\nRMSE NWP", e.RMSE(nwp, obs))
    print("\nBias CNN", e.mean_bias(cnn, obs))
    print("\nBias NWP", e.mean_bias(nwp, obs))

    # 1-1 density plots
    e.plot_1_1_density(result, variables=[variable],
                       xlim_min=-5, xlim_max=60, ylim_min=-5, ylim_max=60)
    plt.title(title[index]+" QC")

    # Create a dataFrame with results
    df_results[f"bias_CNN_{index}"] = e.bias(cnn, obs)
    df_results[f"bias_NWP_{index}"] = e.bias(nwp, obs)
"""
# Boxplot
df_results = df_results.melt(var_name="model", value_name="values")
#plt.figure()
#sns.boxplot(data=df_results, y="values", x="model")
plt.figure()
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