import pickle

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
from downscale.eval.synthetic_topographies import GaussianTopo


prm = create_prm(month_prediction=True)

IGN = MNT(prm=prm)
AROME = NWP(prm["AROME_path_1"], begin=prm["begin"], end=prm["begin_after"], prm=prm)
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)
p = Devine(obs=BDclim, mnt=IGN, nwp=AROME, prm=prm)
p.update_stations_with_neighbors(mnt=IGN, nwp=AROME, GPU=prm["GPU"], number_of_neighbors=4, interpolated=False)
v = Visualization(p)
e = EvaluationFromDict(v)

# Quality control
BDclim.replace_obs_by_QC_obs(prm)
BDclim.delete_obs_not_passing_QC()
use_QC = True
QC = BDclim.time_series

# Results
with open('../../Data/3_Predictions/Arctan_40_2_z_max_no_add.pickle', 'rb') as handle:
    results = pickle.load(handle)

results = e.intersection_model_obs_on_results(results, variables=["UV"], use_QC=use_QC, time_series_qc=QC)

# Acceleration synthetic topographies
gaussian_topo = GaussianTopo()
df_all = gaussian_topo.load_initial_and_predicted_data_in_df(prm)
df_all = gaussian_topo.df_with_list_in_rows_to_flatten_df(df_all)
acceleration_gaussian = df_all['UVW_test'].values / 3
label_gaussian = ["Gaussian topography" for i in range(len(acceleration_gaussian))]

"""
# Acceleration final prediction
variable = "UV"
cnn = e.create_df_from_dict(results, data_type="cnn", variables=[variable])
nwp = e.create_df_from_dict(results, data_type="nwp", variables=[variable])
acceleration_final = cnn.copy()
acceleration_final[variable] = cnn[variable] / nwp[variable]
acceleration_final = acceleration_final[variable][nwp[variable] >= 0]
label_final = ["Real topographies: after CNN + AROME exposition + scaling" for i in range(len(acceleration_final))]

# Acceleration CNN
variable = "acceleration_CNN"
acceleration_cnn = e.create_df_from_dict(results, data_type="cnn", variables=[variable])[variable]
acceleration_cnn = acceleration_cnn[acceleration_cnn.index.isin(acceleration_final.index)]
label_cnn = ["Real topographies: after CNN" for i in range(len(acceleration_cnn))]

df_acceleration = pd.DataFrame()
df_acceleration["value"] = np.concatenate([acceleration_gaussian, acceleration_final, acceleration_cnn])
df_acceleration["acceleration"] = np.concatenate([label_gaussian, label_final, label_cnn])

sns.displot(data=df_acceleration, x="value", hue="acceleration", kind="kde", bw_adjust=3, cut=0)
"""
