import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_style("white")
    sns.set_context('paper', font_scale=1.4)
except ImportError:
    pass

from downscale.PRM_predict import create_prm
from downscale.Utils.GPU import connect_GPU_to_horovod
from downscale.Data_family.MNT import MNT
from downscale.Data_family.Observation import Observation
from downscale.scripts.synthetic_topographies import GaussianTopo
from downscale.Analysis.Visualization import VisualizationGaussian
from downscale.Analysis.Metrics import Metrics
from downscale.Operators.Processing import DwnscHelbig
from downscale.Operators.Processing import Processing

prm = create_prm(month_prediction=True)
connect_GPU_to_horovod() if prm["GPU"] else None
p = Processing(prm=prm)
gaussian_topo = GaussianTopo()

# Load model
p.load_model(dependencies=True)

# Load normalization parameter: std
_, std = p._load_norm_prm()

test = []

# Load data
for fold_nb in range(10):
    path = f"C:/Users/louis/git/wind_downscaling_CNN/Data/2_Pre_processed/ARPS/fold/fold{fold_nb}/df_all_{fold_nb}.csv"
    df_all = pd.read_csv(path)

    # Keep only test values
    df_all = df_all.drop(columns=["degree_xi", 'Unnamed: 0'])
    df_all = df_all[df_all["group"] == "test"]

    topos = gaussian_topo.filenames_to_array(df_all, prm["gaussian_topo_path"], 'topo_name')
    winds = gaussian_topo.filenames_to_array(df_all, prm["gaussian_topo_path"], 'wind_name').reshape(len(topos), 79, 69, 3)

    # Normalize data
    mean_topos = np.mean(topos, axis=(1, 2)).reshape(len(topos), 1, 1)
    std = std.reshape(1, 1, 1)
    topo_norm = p.normalize_topo(topos, mean_topos, std)

    # Predict test data
    predictions = p.model.predict(topo_norm)

    df_all["U_test"] = ""
    df_all["V_test"] = ""
    df_all["W_test"] = ""
    df_all["U_pred"] = ""
    df_all["V_pred"] = ""
    df_all["W_pred"] = ""

    # Define U_test, V_test, W_test, U_pred etc
    for index in range(len(df_all)):
        df_all["topo_name"].iloc[index] = topos[index, :, :]
        df_all["U_test"].iloc[index] = winds[index, :, :, 0]
        df_all["V_test"].iloc[index] = winds[index, :, :, 1]
        df_all["W_test"].iloc[index] = winds[index, :, :, 2]
        df_all["U_pred"].iloc[index] = predictions[index, :, :, 0]
        df_all["V_pred"].iloc[index] = predictions[index, :, :, 1]
        df_all["W_pred"].iloc[index] = predictions[index, :, :, 2]

    # DataFrame to array
    U_test = df_all["U_test"].values
    V_test = df_all["V_test"].values
    W_test = df_all["W_test"].values
    U_pred = df_all["U_pred"].values
    V_pred = df_all["V_pred"].values
    W_pred = df_all["W_pred"].values

    # Compute wind and direction
    df_all["UV_test"] = [gaussian_topo.compute_wind_speed(U_test[index], V_test[index], verbose=False) for index in range(len(U_test))]
    df_all["UVW_test"] = [gaussian_topo.compute_wind_speed(U_test[index], V_test[index], W_test[index], verbose=False) for index in range(len(U_test))]
    df_all["alpha_test"] = [gaussian_topo.angular_deviation(U_test[index], V_test[index], verbose=False) for index in range(len(U_test))]

    df_all["UV_pred"] = [gaussian_topo.compute_wind_speed(U_pred[index], V_pred[index], verbose=False) for index in range(len(U_pred))]
    df_all["UVW_pred"] = [gaussian_topo.compute_wind_speed(U_pred[index], V_pred[index], W_pred[index], verbose=False) for index in range(len(U_pred))]
    df_all["alpha_pred"] = [gaussian_topo.angular_deviation(U_pred[index], V_pred[index], verbose=False) for index in range(len(U_pred))]

    # Calculate TPI
    df_all["tpi_500"] = [gaussian_topo.tpi_map(topos[index, :, :], 500, 30) for index in range(len(df_all))]

    def border_array_to_nan(array):
        array[:17, :] = np.nan
        array[-17:, :] = np.nan
        array[:, :17] = np.nan
        array[:, -17:] = np.nan
        return array

    df_all["tpi_500"] = df_all["tpi_500"].apply(border_array_to_nan)


    # Calculate sx_300
    df_all["sx_300"] = [gaussian_topo.sx_map(topos[index, :, :], 30, 300, 270, 5, 30) for index in range(len(df_all))]

    def border_array_to_nan(array):
        array[:10, :] = np.nan
        array[-10:, :] = np.nan
        array[:, :10] = np.nan
        return array
    df_all["sx_300"] = df_all["sx_300"].apply(border_array_to_nan)

    # Calculate curvature
    df_all["curvature"] = [gaussian_topo.curvature_map(topos[index, :, :], verbose=False) for index in range(len(df_all))]
    def border_array_to_nan(array):
        array[:1, :] = np.nan
        array[-1:, :] = np.nan
        array[:, :1] = np.nan
        array[:, -1:] = np.nan
        return array
    df_all["curvature"] = df_all["curvature"].apply(border_array_to_nan)

    # Calculate laplacian
    df_all["laplacian"] = [gaussian_topo.laplacian_map(topos[index, :, :], 30, verbose=False, helbig=False) for index in range(len(df_all))]
    def border_array_to_nan(array):
        array[:1, :] = np.nan
        array[-1:, :] = np.nan
        array[:, :1] = np.nan
        array[:, -1:] = np.nan
        return array
    df_all["laplacian"] = df_all["laplacian"].apply(border_array_to_nan)

    # Calculate mu
    df_all["mu"] = [gaussian_topo.mu_helbig_map(topos[index, :, :], 30, verbose=False) for index in range(len(df_all))]
    def border_array_to_nan(array):
        array[:1, :] = np.nan
        array[-1:, :] = np.nan
        array[:, :1] = np.nan
        array[:, -1:] = np.nan
        return array
    df_all["mu"] = df_all["mu"].apply(border_array_to_nan)

    # Wind_direction, alpha, TPI, mu, laplacian, sx, curvature
    def unnesting(df, explode):
        idx = df.index.repeat(df[explode[0]].str.len())
        df1 = pd.concat([
            pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
        df1.index = idx

        return df1.join(df.drop(explode, 1), how='left')

    list_variables = ['topo_name', 'U_test', 'V_test',
           'W_test', 'U_pred', 'V_pred', 'W_pred', 'UV_test', 'UVW_test',
           'alpha_test', 'UV_pred', 'UVW_pred', 'alpha_pred', 'tpi_500',
           'curvature', 'laplacian', 'mu', "sx_300"]
    for variable in list_variables:
        df_all[variable] = df_all[variable].apply(lambda x: np.array(x).flatten())

    test_i = unnesting(df_all, list_variables)

    metrics = Metrics()
    test_i["bias"] = metrics.bias(test_i['UVW_pred'].values, test_i['UVW_test'].values)
    test_i["absolute_error"] = metrics.absolute_error(test_i['UVW_pred'].values, test_i['UVW_test'].values)
    test_i["bias_rel"] = metrics.bias_rel(test_i['UVW_pred'].values, test_i['UVW_test'].values)
    test_i["absolute_error_rel"] = metrics.absolute_error_relative(test_i['UVW_pred'].values, test_i['UVW_test'].values)

    test.append(test_i)

test = pd.concat(test)

"""
# General boxplot
ax = sns.boxplot(data=test, y="bias", showfliers=False)
plt.axis("square")

# Boxplot fct class_sx_300, saved = True
order = ['sx_300 <= q25', 'q25 < sx_300 <= q50', 'q50 < sx_300 <= q75', 'q75 < sx_300']
test = gaussian_topo.classify(test, variable="sx_300", quantile=True)
ax = sns.boxplot(data=test, x="class_sx_300", y="bias", showfliers=False, order=order)

# Boxplot fct class_tpi_500, saved = True
order = ['tpi_500 <= q25', 'q25 < tpi_500 <= q50', 'q50 < tpi_500 <= q75', 'q75 < tpi_500']
test = gaussian_topo.classify(test, variable="tpi_500", quantile=True)
ax = sns.boxplot(data=test, x="class_tpi_500", y="bias", showfliers=False, order=order)

# Boxplot fct mu, saved = True
order = ['mu <= q25', 'q25 < mu <= q50', 'q50 < mu <= q75', 'q75 < mu']
test = gaussian_topo.classify(test, variable="mu", quantile=True)
ax = sns.boxplot(data=test, x="class_mu", y="bias", showfliers=False, order=order)

# Boxplot fct curvature, saved = True
order = ['curvature <= q25', 'q25 < curvature <= q50', 'q50 < curvature <= q75', 'q75 < curvature']
test = gaussian_topo.classify(test, variable="curvature", quantile=True)
ax = sns.boxplot(data=test, x="class_curvature", y="bias", showfliers=False, order=order)

# Boxplot fct degree, saved = True
ax = sns.boxplot(data=test, x="degree", y="bias", showfliers=False)

# Boxplot fct xi, saved = True
ax = sns.boxplot(data=test, x="xi", y="bias", showfliers=False)

# Distribution bias, save = True
sns.displot(test, x="bias", kind="kde"), saved = True
plt.title("Bias distribution on the test dataset (Gaussian topographies)")

# Distribution bias by xi
sns.displot(test, x="bias", kind="kde", hue="xi")
plt.title("Test dataset: Bias distribution by xi")

sns.displot(test, x="bias", kind="kde", hue="degree")
plt.title("Test dataset: Bias distribution by degree")

"""




