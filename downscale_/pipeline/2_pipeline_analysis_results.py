import pickle

import numpy as np
import pandas as pd

try:
    import seaborn as sns
    sns.set_style("white")
    sns.set_context('paper', font_scale=1.4)
except ImportError:
    pass

import matplotlib
matplotlib.use('Agg')
#matplotlib.use('Qt5Agg')

from PRM_predict import create_prm
prm = create_prm(month_prediction=True)

from downscale.visu.visualization import Visualization
from downscale.operators.devine import Devine
from downscale.eval.eval_from_dict import EvaluationFromDict
from downscale.data_source.MNT import MNT
from downscale.data_source.NWP import NWP
from downscale.data_source.observation import Observation

prm = create_prm(month_prediction=True)

IGN = MNT(prm=prm)
AROME = NWP(path_to_file=prm["AROME_path_1"], begin=prm["begin"], end=prm["begin_after"], prm=prm)
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)

# Quality control
use_QC = True
BDclim.replace_obs_by_QC_obs(replace_old_wind=True, remove_Dome_and_Vallot_suspicious=True,
                             drop_not_valid=True, prm=prm)
QC = BDclim.time_series

# not_get_closer_and_linear_and_exposed.pickle
# wind_exposed_at_max_elevation_Arctan_30_2.pickle
# wind_exposed_at_10m_2nd_try.pickle
# Arctan_10_2_z_max_no_add.pickle

with open(prm["working_directory"]+'/Data/3_Predictions/Stations/21_02_2022v2.pickle', 'rb') as handle:
    results = pickle.load(handle)

p = Devine(obs=BDclim, mnt=IGN, nwp=AROME, prm=prm)
p.update_stations_with_neighbors(mnt=IGN, nwp=AROME, GPU=prm["GPU"], number_of_neighbors=4, interpolated=False)

v = Visualization(p)
e = EvaluationFromDict(v)

# Latex table
#e.create_latex_table(results, use_QC=use_QC, time_series_qc=QC)

#e.compute_stats(results, variable="UV_DIR_deg", metric="abs_bias_direction", metric_2="bias_direction", use_QC=use_QC, time_series_qc=QC, variable_qc="winddir(deg)")
#with open(prm["working_directory"]+'/Data/3_Predictions/Stations/21_02_2022v2.pickle', 'rb') as handle:
#    results = pickle.load(handle)
#e.compute_stats(results, variable="UV", metric="abs_error", metric_2="bias", use_QC=use_QC, time_series_qc=QC, variable_qc="vw10m(m/s)")

# Boxplots
#with open(prm["working_directory"]+'/Data/3_Predictions/Stations/21_02_2022v2.pickle', 'rb') as handle:
#    results = pickle.load(handle)
#e.plot_metric_all_stations(results, variables=["UV"], metrics=["bias_rel_wind_1"], use_QC=use_QC, time_series_qc=QC, variable_qc="vw10m(m/s)", prm=prm)
#with open(prm["working_directory"]+'/Data/3_Predictions/Stations/21_02_2022v2.pickle', 'rb') as handle:
#    results = pickle.load(handle)
#e.plot_metric_all_stations(results, variables=["UV_DIR_deg"], metrics=["bias_direction"], use_QC=use_QC, time_series_qc=QC, variable_qc="winddir(deg)", prm=prm)

# 1-1 plot with density
#e.plot_1_1_density(results, cmap="viridis", use_QC=use_QC, time_series_qc=QC, xlim_min=0, xlim_max=40, ylim_min=0, ylim_max=40, prm=prm)
#e.plot_1_1_density_by_station(results, cmap="viridis", use_QC=use_QC, time_series_qc=QC, prm=prm)

# Q-Q plot
#e.plot_quantile_quantile(results, use_QC=use_QC, time_series_qc=QC, prm=prm)

#with a notable improvement for the most elevated stations (elevation > 2500m) where the negative speed bias is reduced by 40\%
#e.plot_metric_all_stations(results, variables=["UV"], metrics=["bias"], use_QC=use_QC, time_series_qc=QC, variable_qc="vw10m(m/s)", prm=prm)

metrics = ["bias_rel_wind_1"]
#e.plot_heatmap(results, metrics=metrics, use_QC=use_QC, time_series_qc=QC, min_speed=1, vmax=0.2, vmax_diff=0.1, prm=prm)
#e.plot_heatmap_by_wind_category(results, variables=["UV"], metrics=metrics, min_speed=0.5, max_speed=3,
#                                use_QC=use_QC, time_series_qc=QC, vmax_diff=0.2, prm=prm)
#e.plot_heatmap_by_wind_category(results, variables=["UV"], metrics=metrics, min_speed=3.1, max_speed=7,
#                                use_QC=use_QC, time_series_qc=QC, vmax_diff=0.2, prm=prm)
#e.plot_heatmap_by_wind_category(results, variables=["UV"], metrics=metrics, min_speed=7.1, max_speed=1000,
#                                use_QC=use_QC, time_series_qc=QC, vmax_diff=0.2, prm=prm)

#e.plot_metric_distribution(results, variables=["UV_DIR_deg"], metrics=["abs_bias_direction"], use_QC=use_QC,
#                           time_series_qc=QC, prm=prm)

#e.plot_accelerations(results, sort_by="tpi_2000", showfliers=True, use_QC=use_QC, time_series_qc=QC)

#e.plot_accelerations_exposed_distributions(results, use_QC=use_QC, time_series_qc=QC)


# Wind speed distribution for CNN, NWP and observation
# Not working 7 Janver 2021
#e.plot_distribution_all_stations(results, use_QC=use_QC, time_series_qc=QC)
