import uuid

import matplotlib
matplotlib.use('Agg')
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    sns.set(rc={'figure.figsize': (2 * 11.7, 2 * 8.27)})
    sns.set_style("white")
    sns.set_context('paper', font_scale=1.4)
except ImportError:
    pass

from PRM_predict import create_prm
prm = create_prm(month_prediction=True)

import sys

from downscale.eval.synthetic_topographies import GaussianTopo
from downscale.data_source.observation import Observation
from downscale.data_source.MNT import MNT
from downscale.operators.helbig import DwnscHelbig
from downscale.utils.utils_func import save_figure
from downscale.eval.eval_training import figure_distribution_gaussian
from downscale.visu.visu_gaussian import VisualizationGaussian
from downscale.operators.devine import Devine
from downscale.visu.MidpointNormalize import MidpointNormalize

IGN = MNT(prm=prm)
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)
gaussian_topo = GaussianTopo()

# Figure example_topo_wind_gaussian
config = dict(
    cmap_topo="viridis",
    cmap_arrow="coolwarm",
    midpoint=3,
    scale=35,
    range_idx_to_plot1=[7269, 7262, 7259, 7216, 7210, 7207, 7195, 7194, 7181, 7194, 7195, 7104, 7120, 7131, 7146, 7172, 7179, 7180, 7181, 5010, 5011, 5012, 5013, 5014, 5015, 5016, 5017, 5018, 5019, 5020, 6914, 6984],
    range_idx_to_plot=[500, 5014, 7104],
    n=2,
    vmin=1.5,
    vmax=4.5,
    fontsize=50,
    fontsize_3D=50,
)


#gaussian_topo.figure_example_topo_wind_gaussian(config=config, prm=prm)

# figure tpi_sx
config = dict(
    min_y_dem=16380,
    max_y_dem=26830,
    min_x_dem=26190,
    max_x_dem=32790,
    working_on_a_small_example=False,
    distance_tpi=500,
    resolution_dem=30,
    distance_sx=300,
    angle_sx=270,
    color_real="C2",
    color_gaussian="goldenrod",
    color_station="red",
    svg=False
)

#gaussian_topo.figure_tpi_vs_sx(IGN, BDclim, config, prm)

# Figure laplacian vs mu

config = dict(
    min_y_dem=16380,
    max_y_dem=26830,
    min_x_dem=26190,
    max_x_dem=32790,
    working_on_a_small_example=False,
    laplacian_helbig=False,
    resolution_dem=30,
    color_real="C2",
    color_gaussian="goldenrod",
    color_station="red",
    svg=False
)

#gaussian_topo.figure_laplacian_vs_mu(IGN, BDclim, config, prm)

"""
down_helb = DwnscHelbig()
IGN.data = IGN.data[5000:5300, 7000:7200]
gaussian_topo = GaussianTopo()
topo, wind = gaussian_topo.load_data_all(prm)
topo = topo.reshape((len(topo), 79, 69))
wind = wind.reshape((len(topo), 79, 69, 3))
print("Nb synthetic topographies: ", len(topo))

topo = topo[:100, :, :]
wind = wind[:100, :, :, :]

#
#
# Begin figure 2
#
#

# TPI and Sx on real topographies
tpi_real = down_helb.tpi_map(mnt, 500, resolution=30)
sx_real_1 = down_helb.sx_map(mnt, 30, 300, 270)

tpi_real[:17, :] = np.nan
tpi_real[-17:, :] = np.nan
tpi_real[:, :17] = np.nan
tpi_real[:, -17:] = np.nan

sx_real_1[:10, :] = np.nan
sx_real_1[-10:, :] = np.nan
sx_real_1[:, :10] = np.nan
sx_real_1[:, -10:] = np.nan

tpi_real_flat = tpi_real.flatten()
sx_real_flat_1 = sx_real_1.flatten()
print("TPI and sx real flat calculated")

# TPI and Sx on gaussian topographies
tpi_gau = np.array([down_helb.tpi_map(topo[i], 500, 30) for i in range(len(topo))])
sx_gau = np.array([down_helb.sx_map(topo[i], 30, 300, 270) for i in range(len(topo))])

tpi_gau[:17, :] = np.nan
tpi_gau[-17:, :] = np.nan
tpi_gau[:, :17] = np.nan
tpi_gau[:, -17:] = np.nan

sx_gau[:10, :] = np.nan
sx_gau[-10:, :] = np.nan
sx_gau[:, :10] = np.nan
sx_gau[:, -10:] = np.nan

tpi_gau_flat = tpi_gau.flatten()
sx_gau_flat = sx_gau.flatten()
print("TPI and sx gau flat calculated")

df_topo_gaussian = pd.DataFrame(np.transpose([tpi_gau_flat, sx_gau_flat]), columns=["tpi", "sx"])
df_topo_gaussian["topography"] = "gaussian"
df_topo_real_1 = pd.DataFrame(np.transpose([tpi_real_flat, sx_real_flat_1]), columns=["tpi", "sx"])
df_topo_real_1["topography"] = "real"
df_topo = pd.concat([df_topo_gaussian, df_topo_real_1])
print("Dataframe created")

# Observations
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)
BDclim.update_stations_with_KNN_from_MNT_using_cKDTree(IGN)
idx_x = np.array([np.intp(idx_x) for (idx_x, _) in BDclim.stations[f"index_{prm['name_mnt']}_NN_0_cKDTree_ref_{prm['name_mnt']}"].to_list()])
idx_y = np.array([np.intp(idx_y) for (_, idx_y) in BDclim.stations[f"index_{prm['name_mnt']}_NN_0_cKDTree_ref_{prm['name_mnt']}"].to_list()])
names = BDclim.stations["name"].values
tpi_stations = down_helb.tpi_idx(IGN.data, idx_x=idx_x, idx_y=idx_y, radius=500, resolution=30)
sx_stations = down_helb.sx_idx(IGN.data, idx_x, idx_y, 30, 300, 270, 5, 30)
print("TPI and sx stations calculated")

# Unique values for scatter plot
df_topo1 = df_topo.dropna().drop_duplicates()
tpi_gau_unique = df_topo1["tpi"][df_topo1["topography"] == "gaussian"]
sx_gau_unique = df_topo1["sx"][df_topo1["topography"] == "gaussian"]
tpi_real_unique = df_topo1["tpi"][df_topo1["topography"] == "real"]
sx_real_unique = df_topo1["sx"][df_topo1["topography"] == "real"]
print("TPI and sx unique calculated")

tpi_real_unique = tpi_real_unique.astype(np.float32)
sx_real_unique = sx_real_unique.astype(np.float32)
tpi_gau_unique = tpi_gau_unique.astype(np.float32)
sx_gau_unique = sx_gau_unique.astype(np.float32)
tpi_stations = tpi_stations.astype(np.float32)
sx_stations = sx_stations.astype(np.float32)
df_topo["tpi"] = df_topo["tpi"].astype(np.float32)
df_topo["sx"] = df_topo["sx"].astype(np.float32)

plt.figure(figsize=(15, 15))
df_topo.index = list(range(len(df_topo)))
result = sns.jointplot(data=df_topo.dropna(), x="tpi", y="sx", hue="topography",
                       palette=["C2", "goldenrod"], marker="o", s=3, linewidth=0, edgecolor=None,
                       hue_order=["real", "gaussian"], legend=False, marginal_kws=dict(bw=0.8))
ax = result.ax_joint
ax.scatter(tpi_gau_unique, sx_gau_unique, s=3, alpha=0.75, color="goldenrod")
ax.scatter(tpi_stations, sx_stations, s=7, alpha=1, color='red', zorder=10)
ax.grid(True)
ax = result.ax_marg_x
ax.scatter(tpi_stations, np.zeros_like(tpi_stations), s=5, alpha=1, color='red', zorder=10)
ax.grid(True)
ax = result.ax_marg_y
ax.scatter(np.zeros_like(sx_stations), sx_stations, s=5, alpha=1, color='red', zorder=10)
ax.grid(True)
save_figure("tpi_vs_sx", prm)
print("Second figure TPI_sx created")





#
#
# Begin figure 1
#
#

laplacian_gaussian = down_helb.laplacian_map(topo, 30, helbig=False, verbose=False)
mu_gaussian = down_helb.mu_helbig_map(topo, 30, verbose=False)
laplacian_gaussian_flat = laplacian_gaussian[:, 1:-1, 1:-1].flatten()
mu_gaussian_flat = mu_gaussian[:, 1:-1, 1:-1].flatten()

laplacian_real = down_helb.laplacian_map(mnt, 25, helbig=False, verbose=False)
mu_real = down_helb.mu_helbig_map(mnt, 25, verbose=False)
laplacian_real_flat = laplacian_real[1:-1, 1:-1].flatten()
mu_real_flat = mu_real[1:-1, 1:-1].flatten()

df_topo_gaussian = pd.DataFrame(np.transpose([laplacian_gaussian_flat, mu_gaussian_flat]), columns=["laplacian", "mu"])
df_topo_gaussian["topography"] = "gaussian"
df_topo_real = pd.DataFrame(np.transpose([laplacian_real_flat, mu_real_flat]), columns=["laplacian", "mu"])
df_topo_real["topography"] = "real"
df_topo = pd.concat([df_topo_gaussian, df_topo_real])

df_topo1 = df_topo.drop_duplicates()
lapl_gau_unique = df_topo1["laplacian"][df_topo1["topography"] == "gaussian"]
mu_gau_unique = df_topo1["mu"][df_topo1["topography"] == "gaussian"]
lapl_real_unique = df_topo1["laplacian"][df_topo1["topography"] == "real"]
mu_real_unique = df_topo1["mu"][df_topo1["topography"] == "real"]

BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)
BDclim.update_stations_with_KNN_from_MNT_using_cKDTree(IGN)
idx_x = np.array([np.intp(idx_x) for (idx_x, _) in BDclim.stations[f"index_{prm['name_mnt']}_NN_0_cKDTree_ref_{prm['name_mnt']}"].to_list()])
idx_y = np.array([np.intp(idx_y) for (_, idx_y) in BDclim.stations[f"index_{prm['name_mnt']}_NN_0_cKDTree_ref_{prm['name_mnt']}"].to_list()])
names = BDclim.stations["name"].values
mu_stations = down_helb.mu_helbig_idx(IGN.data, dx=25, idx_x=idx_x, idx_y=idx_y)
laplacian_stations = down_helb.laplacian_idx(IGN.data, dx=25, idx_x=idx_x, idx_y=idx_y, helbig=False)


df_topo.index = list(range(len(df_topo)))
result = sns.jointplot(data=df_topo.dropna(), x="laplacian", y="mu", hue="topography", marker="o", s=5, palette=["C2", "goldenrod"], linewidth=0, edgecolor=None, hue_order=["real", "gaussian"], legend=False)
ax = result.ax_joint
ax.scatter(lapl_gau_unique, mu_gau_unique, s=5, alpha=0.75, color="goldenrod")
ax.scatter(laplacian_stations, mu_stations, s=5, alpha=1, color='red', zorder=10)
ax.grid(True)
ax = result.ax_marg_x
ax.scatter(laplacian_stations, np.zeros_like(laplacian_stations), s=5, alpha=1, color='red', zorder=10)
ax.grid(True)
ax = result.ax_marg_y
ax.scatter(np.zeros_like(mu_stations), mu_stations, s=5, alpha=1, color='red', zorder=10)
ax.grid(True)
save_figure("laplacian_vs_mu", prm)

# Distributions
# for type_plot in ["acceleration", "wind speed", "angular deviation"]:
#    visu.plot_gaussian_distrib(uv, type_plot=type_plot)
"""

"""
# Wind speed, accelerations and angular deviations
uv = gaussian_topo.compute_wind_speed(u, v)
uvw = gaussian_topo.compute_wind_speed(u, v, w)
acceleration = gaussian_topo.wind_speed_ratio(num=wind, den=np.full(wind.shape, 3))
acc_u, acc_v, acc_w = acceleration[:, :, :, 0], acceleration[:, :, :, 1], acceleration[:, :, :, 2]
acc_uv = gaussian_topo.wind_speed_ratio(num=uv, den=np.full(uv.shape, 3))
acc_uvw = gaussian_topo.wind_speed_ratio(num=uvw, den=np.full(uvw.shape, 3))
alpha = np.rad2deg(gaussian_topo.angular_deviation(u, v))
arg = np.unique(np.argwhere(alpha >= 60)[:, 0])
"""

#figure_distribution_gaussian(prm)
