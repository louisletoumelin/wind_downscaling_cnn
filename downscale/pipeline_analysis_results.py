import pickle

try:
    import seaborn as sns
    sns.set_style("white")
    sns.set_context('paper', font_scale=1.4)
except ImportError:
    pass

from visu.visualization import Visualization
from downscale.operators.devine import Devine
from downscale.eval.eval_from_dict import EvaluationFromDict
from downscale.data_source.MNT import MNT
from downscale.data_source.NWP import NWP
from downscale.data_source.observation import Observation
from PRM_predict import create_prm

prm = create_prm(month_prediction=True)
#connect_GPU_to_horovod() if prm["GPU"] else None

IGN = MNT(prm=prm)
AROME = NWP(path_to_file=prm["AROME_path_1"], begin=prm["begin"], end=prm["begin_after"], prm=prm)
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)

# Quality control
BDclim.replace_obs_by_QC_obs(prm)
use_QC = True
QC = BDclim.time_series

# not_get_closer_and_linear_and_exposed.pickle
# wind_exposed_at_max_elevation_Arctan_30_2.pickle
# wind_exposed_at_10m_2nd_try.pickle
# Arctan_10_2_z_max_no_add.pickle
with open('../../Data/3_Predictions/unexposed_wind_not_activated_Arctan_30_2.pickle', 'rb') as handle:
    results = pickle.load(handle)

p = Devine(obs=BDclim, mnt=IGN, nwp=AROME, prm=prm)
p.update_stations_with_neighbors(mnt=IGN, nwp=AROME, GPU=prm["GPU"], number_of_neighbors=4, interpolated=False)

v = Visualization(p)
e = EvaluationFromDict(v)

# Boxplots
metrics = ["bias_rel_wind_1"]
#metrics = ["bias"]
#e.plot_metric_all_stations(results, variables=["UV"], metrics=metrics, use_QC=use_QC, time_series_qc=QC)

# 1-1 plot with density
#e.plot_1_1_density(results, cmap="viridis", use_QC=use_QC, time_series_qc=QC)

# Wind speed distribution for CNN, NWP and observation
#e.plot_distribution_all_stations(results, use_QC=use_QC, time_series_qc=QC)

#metrics = ["bias_rel_wind_1"]
e.plot_heatmap(results, metrics=metrics, use_QC=use_QC, time_series_qc=QC)

#e.plot_accelerations(results, sort_by="tpi_2000", showfliers=True, use_QC=use_QC, time_series_qc=QC)

#e.plot_accelerations_exposed_distributions(results, use_QC=use_QC, time_series_qc=QC)
