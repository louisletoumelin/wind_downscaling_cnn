import matplotlib

matplotlib.use('Agg')

try:
    import seaborn as sns
    sns.set(rc={'figure.figsize': (2 * 11.7, 2 * 8.27)})
    sns.set_style("white")
    sns.set_context('paper', font_scale=1.4)
except ImportError:
    pass

from .PRM_predict import create_prm
prm = create_prm(month_prediction=True)

from downscale.eval.synthetic_topographies import GaussianTopo



# Load data and reshape
gaussian_topo = GaussianTopo()
topo, wind = gaussian_topo.load_data_all(prm)
topo = topo.reshape((len(topo), 79, 69))
print("Nb synthetic topographies: ", len(topo))
"""
down_helb = DwnscHelbig()
IGN = MNT(prm=prm)
mnt = IGN.data[50:11870, 1600:7400]


#
#
# Begin figure 2
#
#

tpi_real = down_helb.tpi_map(mnt, 500)
sx_real_1 = down_helb.sx_map(mnt, 25, 300, 270)

tpi_real[:17, :] = np.nan
tpi_real[-17:, :] = np.nan
tpi_real[:, :17] = np.nan
tpi_real[:, -17:] = np.nan

sx_real_1[:12, :] = np.nan
sx_real_1[-12:, :] = np.nan
sx_real_1[:, :12] = np.nan
sx_real_1[:, -12:] = np.nan

#sx_real_2 = down_helb.sx_map(mnt, 25, 300, 270)[12:-12, 12:-12]
#sx_real_3 = down_helb.sx_map(mnt, 25, 300, 270)[12:-12, 12:-12]
#sx_real_4 = down_helb.sx_map(mnt, 25, 300, 270)[12:-12, 12:-12]

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

tpi_real_flat = tpi_real.flatten()
sx_real_flat_1 = sx_real_1.flatten()
print("TPI and sx real flat calculated")

tpi_gau_flat = tpi_gau.flatten()
sx_gau_flat = sx_gau.flatten()
print("TPI and sx gau flat calculated")

df_topo_gaussian = pd.DataFrame(np.transpose([tpi_gau_flat, sx_gau_flat]), columns=["tpi", "sx"])
df_topo_gaussian["topography"] = "gaussian"
df_topo_real_1 = pd.DataFrame(np.transpose([tpi_real_flat, sx_real_flat_1]), columns=["tpi", "sx"])
df_topo_real_1["topography"] = "real"
df_topo = pd.concat([df_topo_gaussian, df_topo_real_1])

df_topo1 = df_topo.dropna().drop_duplicates()
tpi_gau_unique = df_topo1["tpi"][df_topo1["topography"] == "gaussian"]
sx_gau_unique = df_topo1["sx"][df_topo1["topography"] == "gaussian"]
tpi_real_unique = df_topo1["tpi"][df_topo1["topography"] == "real"]
sx_real_unique = df_topo1["sx"][df_topo1["topography"] == "real"]
print("TPI and sx unique calculated")

BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)
BDclim.update_stations_with_KNN_from_MNT_using_cKDTree(IGN)
idx_x = np.array([np.intp(idx_x) for (idx_x, _) in BDclim.stations['index_IGN_NN_0_cKDTree_ref_IGN'].to_list()])
idx_y = np.array([np.intp(idx_y) for (_, idx_y) in BDclim.stations['index_IGN_NN_0_cKDTree_ref_IGN'].to_list()])
names = BDclim.stations["name"].values

tpi_stations = down_helb.tpi_idx(IGN.data, idx_x=idx_x, idx_y=idx_y, radius=500, resolution=25)
sx_stations = down_helb.sx_idx(IGN.data, idx_x, idx_y, 25, 300, 270, 5, 30)
print("TPI and sx stations calculated")


plt.figure()
plt.scatter(tpi_real_unique, sx_real_unique, s=5)
plt.scatter(tpi_gau_unique, sx_gau_unique, s=5, alpha=0.75)
plt.scatter(tpi_stations, sx_stations, s=5, alpha=1, color='red', zorder=10)
plt.xlabel("Tpi 500m")
plt.ylabel("Sx 300m")
plt.legend(("Real topography", "Gaussian topography", "Observation stations"), loc='upper right', fontsize=13)
plt.savefig("TPI_sx.png")
plt.savefig("TPI_sx.svg")
"""
"""
tpi_gau_unique = tpi_gau_unique.astype(np.float32)
sx_gau_unique = sx_gau_unique.astype(np.float32)
tpi_stations = tpi_stations.astype(np.float32)
sx_stations = sx_stations.astype(np.float32)
df_topo["tpi"] = df_topo["tpi"].astype(np.float32)
df_topo["sx"] = df_topo["sx"].astype(np.float32)

result = sns.jointplot(data=df_topo.dropna(), x="tpi", y="sx", hue="topography", marker="o", s=3, linewidth=0, edgecolor=None, hue_order=["real", "gaussian"], legend=False, marginal_kws=dict(bw=0.8))
ax = result.ax_joint
ax.scatter(tpi_gau_unique, sx_gau_unique, s=3, alpha=0.75, color="C1")
ax.scatter(tpi_stations, sx_stations, s=7, alpha=1, color='red', zorder=10)
ax = result.ax_marg_x
ax.scatter(tpi_stations, np.zeros_like(tpi_stations), s=5, alpha=1, color='red', zorder=10)
ax = result.ax_marg_y
ax.scatter(np.zeros_like(sx_stations), sx_stations, s=5, alpha=1, color='red', zorder=10)
plt.savefig("tpi_vs_sx3.png")
plt.savefig("tpi_vs_sx3.svg")




#
#
# Begin figure 1
#
#

laplacian_gaussian = down_helb.laplacian_map(topo, 30, helbig=True, verbose=False)
mu_gaussian = down_helb.mu_helbig_map(topo, 30, verbose=False)
laplacian_gaussian_flat = laplacian_gaussian[:, 1:-1, 1:-1].flatten()
mu_gaussian_flat = mu_gaussian[:, 1:-1, 1:-1].flatten()

laplacian_real = down_helb.laplacian_map(mnt, 25, helbig=True, verbose=False)
mu_real = down_helb.mu_helbig_map(mnt, 25, verbose=False)
laplacian_real_flat = laplacian_real[1:-1, 1:-1].flatten()
mu_real_flat = mu_real[1:-1, 1:-1].flatten()

df_topo_gaussian = pd.DataFrame(np.transpose([laplacian_gaussian_flat, mu_gaussian_flat]), columns=["laplacian", "mu"])
df_topo_gaussian["topography"] = "gaussian"
df_topo_real = pd.DataFrame(np.transpose([laplacian_real_flat, mu_real_flat]), columns=["laplacian", "mu"])
df_topo_real["topography"] = "real"
df_topo = pd.concat([df_topo_gaussian, df_topo_real])
#sns.displot(data=df_topo, x="laplacian", hue="topography", kind='kde', fill=True)
#sns.displot(data=df_topo, x="mu", hue="topography", kind='kde', fill=True)

df_topo1 = df_topo.drop_duplicates()
lapl_gau_unique = df_topo1["laplacian"][df_topo1["topography"] == "gaussian"]
mu_gau_unique = df_topo1["mu"][df_topo1["topography"] == "gaussian"]
lapl_real_unique = df_topo1["laplacian"][df_topo1["topography"] == "real"]
mu_real_unique = df_topo1["mu"][df_topo1["topography"] == "real"]

BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)
BDclim.update_stations_with_KNN_from_MNT_using_cKDTree(IGN)
idx_x = np.array([np.intp(idx_x) for (idx_x, _) in BDclim.stations['index_IGN_NN_0_cKDTree_ref_IGN'].to_list()])
idx_y = np.array([np.intp(idx_y) for (_, idx_y) in BDclim.stations['index_IGN_NN_0_cKDTree_ref_IGN'].to_list()])
names = BDclim.stations["name"].values

mu_stations = down_helb.mu_helbig_idx(IGN.data, dx=25, idx_x=idx_x, idx_y=idx_y)
laplacian_stations = down_helb.laplacian_idx(IGN.data, dx=25, idx_x=idx_x, idx_y=idx_y, helbig=True)
"""
"""
#
#
# Scatter plot
#
#
plt.figure()
plt.scatter(lapl_real_unique, mu_real_unique, s=5)
plt.scatter(lapl_gau_unique, mu_gau_unique, s=5, alpha=0.75)
ax = plt.gca()

sns.kdeplot(data=df_topo[df_topo["topography"] == "real"].sample(500_000),
            x="laplacian", y="mu", cut=0, color="midnightblue", levels=9, ax=ax)

plt.scatter(laplacian_stations, mu_stations, s=5, alpha=1, color='red', zorder=10)
plt.xlabel("Laplacian according to Helbig et al. 2017")
plt.ylabel("Mu")
plt.legend(("Real topography in the french Alps", "Gaussian topography", "Observation stations"), loc='upper right', fontsize=13)
plt.axis("square")
#ax = plt.gca()
#for lapl, mu, name in zip(laplacian_stations, mu_stations, names):
#    ax.text(lapl, mu, name, color='red')
"""
"""
#
#
# joinplot
#
#


result = sns.jointplot(data=df_topo, x="laplacian", y="mu", hue="topography", marker="o", s=5, linewidth=0, edgecolor=None, hue_order=["real", "gaussian"], legend=False)
ax = result.ax_joint
ax.scatter(lapl_gau_unique, mu_gau_unique, s=5, alpha=0.75, color="C1")
ax.scatter(laplacian_stations, mu_stations, s=5, alpha=1, color='red', zorder=10)
ax = result.ax_marg_x
ax.scatter(laplacian_stations, np.zeros_like(laplacian_stations), s=5, alpha=1, color='red', zorder=10)
ax = result.ax_marg_y
ax.scatter(np.zeros_like(mu_stations), mu_stations, s=5, alpha=1, color='red', zorder=10)
plt.savefig("laplacian_vs_mu.png")
plt.savefig("laplacian_vs_mu.svg")
"""








"""
wind = wind.reshape((len(wind), 79, 69, 3))
u, v, w = wind[:, :, :, 0], wind[:, :, :, 1], wind[:, :, :, 2]


# Wind speed, accelerations and angular deviations
uv = gaussian_topo.compute_wind_speed(u, v)
uvw = gaussian_topo.compute_wind_speed(u, v, w)
acceleration = gaussian_topo.wind_speed_ratio(num=wind, den=np.full(wind.shape, 3))
acc_u, acc_v, acc_w = acceleration[:, :, :, 0], acceleration[:, :, :, 1], acceleration[:, :, :, 2]
acc_uv = gaussian_topo.wind_speed_ratio(num=uv, den=np.full(uv.shape, 3))
acc_uvw = gaussian_topo.wind_speed_ratio(num=uvw, den=np.full(uvw.shape, 3))
alpha = np.rad2deg(gaussian_topo.angular_deviation(u, v))
arg = np.unique(np.argwhere(alpha >= 60)[:, 0])

p = Processing()
visu = VisualizationGaussian(p)

# 2D maps
visu.plot_gaussian_topo_and_wind_2D(topo, u, v, w, uv, alpha, arg[0])

# Distributions
for type_plot in ["acceleration", "wind speed", "angular deviation"]:
    visu.plot_gaussian_distrib(uv, type_plot=type_plot)

# Distributions by degree or xi
type_plots = ["wind speed", "wind speed", "angular deviation"]
type_of_winds = ["uv", "uvw", None]
for topo_carac in ["degree", "xi"]:
    for type_plot, type_of_wind in zip(type_plots, type_of_winds):
        dict_gaussian_deg_or_xi = gaussian_topo.load_data_by_degree_or_xi(prm,
                                                                          degree_or_xi=topo_carac)
        visu.plot_gaussian_distrib_by_degree_or_xi(dict_gaussian_deg_or_xi,
                                                   degree_or_xi=topo_carac,
                                                   type_plot=type_plots,
                                                   type_of_wind=type_of_winds,
                                                   fill=False, fontsize=20)
"""

