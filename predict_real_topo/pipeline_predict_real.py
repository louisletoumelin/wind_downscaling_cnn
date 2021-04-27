from time import time as t
t_init = t()

import numpy as np
import tensorflow as tf
from line_profiler import LineProfiler

def round(t1, t2):  return (np.round(t2 - t1, 2))


from Processing import Processing
from Visualization import Visualization
from MNT import MNT
from NWP import NWP
from Observation import Observation
from Data_2D import Data_2D
from MidpointNormalize import MidpointNormalize
from Evaluation import Evaluation


"""
Stations
"""
"""
['BARCELONNETTE', 'DIGNE LES BAINS', 'RESTEFOND-NIVOSE',
       'LA MURE-ARGENS', 'ARVIEUX', 'PARPAILLON-NIVOSE', 'EMBRUN',
       'LA FAURIE', 'GAP', 'LA MEIJE-NIVOSE', 'COL AGNEL-NIVOSE',
       'GALIBIER-NIVOSE', 'ORCIERES-NIVOSE', 'RISTOLAS',
       'ST JEAN-ST-NICOLAS', 'TALLARD', "VILLAR D'ARENE",
       'VILLAR ST PANCRACE', 'ASCROS', 'PEIRA CAVA', 'PEONE',
       'MILLEFONTS-NIVOSE', 'CHAPELLE-EN-VER', 'LUS L CROIX HTE',
       'ST ROMAN-DIOIS', 'AIGLETON-NIVOSE', 'CREYS-MALVILLE',
       'LE GUA-NIVOSE', "ALPE-D'HUEZ", 'LA MURE- RADOME',
       'LES ECRINS-NIVOSE', 'GRENOBLE-ST GEOIRS', 'ST HILAIRE-NIVOSE',
       'ST-PIERRE-LES EGAUX', 'GRENOBLE - LVD', 'VILLARD-DE-LANS',
       'CHAMROUSSE', 'ALBERTVILLE JO', 'BONNEVAL-NIVOSE', 'MONT DU CHAT',
       'BELLECOTE-NIVOSE', 'GRANDE PAREI NIVOSE', 'FECLAZ_SAPC',
       'COL-DES-SAISIES', 'ALLANT-NIVOSE', 'LA MASSE',
       'ST MICHEL MAUR_SAPC', 'TIGNES_SAPC', 'LE CHEVRIL-NIVOSE',
       'LES ROCHILLES-NIVOSE', 'LE TOUR', 'AGUIL. DU MIDI',
       'AIGUILLES ROUGES-NIVOSE', 'LE GRAND-BORNAND', 'MEYTHET',
       'LE PLENAY', 'SEYNOD-AREA', 'Col du Lac Blanc', 'Col du Lautaret']
"""

"""
Files
"""

GPU = False
Z0 = True
load_z0 = True
save_z0 = False
peak_valley = True

# Safety
if load_z0:
    save_z0 = False
if save_z0:
    load_z0 = False

# Horovod: initialize Horovod.
if GPU:
    import horovod.tensorflow.keras as hvd
    hvd.init()
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    print(gpus)

# Data path
data_path = "C:/Users/louis/git/wind_downscaling_CNN/Data/1_Raw/"
if GPU:
    data_path = "//scratch/mrmn/letoumelinl/predict_real/"

# CNN path
experience_path = "C:/Users/louis/git/wind_downscaling_CNN/Models/ARPS/"
if GPU:
    experience_path = "//scratch/mrmn/letoumelinl/predict_real/Model/"

# Topography path
topo_path = data_path + "MNT/IGN_25m/ign_L93_25m_alpesIG.tif"
if GPU:
    topo_path = data_path + "MNT/IGN_25m/preprocessed_MNT.nc"

# NWP
AROME_path_1 = data_path + "AROME/FORCING_alp_2017080106_2018080106.nc"
AROME_path_2 = data_path + "AROME/FORCING_alp_2018080106_2019060106.nc"
AROME_path_3 = data_path + "AROME/FORCING_alp_2019060107_2019070106.nc"
path_to_file_npy = data_path + "AROME/AROME_June_2019" if GPU else None
AROME_path = [AROME_path_1, AROME_path_2, AROME_path_3]
selected_path = AROME_path_2

# Z0
path_Z0_2018 = data_path + "AROME/Z0/Z0_alp_2018010100_2018120700.nc"
path_Z0_2019 = data_path + "AROME/Z0/Z0_alp_20190103_20191227.nc"
save_path = data_path + "AROME/Z0/"

# Observation
BDclim_stations_path = data_path + "BDclim/precise_localisation/liste_postes_alps_l93.csv"
if GPU:
    BDclim_stations_path = data_path + "BDclim/pickle_stations.pkl"
BDclim_data_path = data_path + "BDclim/extract_BDClim_et_sta_alp_20171101_20190501.csv"
path_vallot = data_path + "BDclim/Vallot/"
path_saint_sorlin = data_path + "BDclim/Saint-Sorlin/"
path_argentiere = data_path + "BDclim/Argentiere/"

# CNN model
model_experience = "date_16_02_name_simu_FINAL_1_0_model_UNet/"
model_path = experience_path + model_experience

"""
Time
"""

# Date to predict
day_begin = 1
month_begin = 10
year_begin = 2018

day_end = 31
month_end = 10
year_end = 2018

begin = str(year_begin) + "-" + str(month_begin) + "-" + str(day_begin)
end = str(year_end) + "-" + str(month_end) + "-" + str(day_end)

"""
MNT, NWP and observations
"""
# IGN
IGN = MNT(topo_path, "IGN")

# AROME
if Z0:
    AROME = NWP(selected_path, "AROME", begin, end, save_path=save_path, path_Z0_2018=path_Z0_2018, path_Z0_2019=path_Z0_2019, verbose=True, load_z0=load_z0, save=save_z0)
else:
    AROME = NWP(selected_path, "AROME", begin, end, save_path=None, path_Z0_2018=None, path_Z0_2019=None, verbose=True, path_to_file_npy=path_to_file_npy)


# BDclim
BDclim = Observation(BDclim_stations_path, BDclim_data_path, path_vallot, path_saint_sorlin, path_argentiere, begin=begin, end=end, select_date_time_serie=False, vallot=True)
# BDclim.stations_to_gdf(ccrs.epsg(2154), x="X", y="Y")
if not(GPU):
    number_of_neighbors = 4
    BDclim.update_stations_with_KNN_from_NWP(number_of_neighbors, AROME)
    BDclim.update_stations_with_KNN_from_MNT_using_cKDTree(IGN)

"""
Processing, visualization and evaluation
"""

# Processing
p = Processing(BDclim, IGN, AROME, model_path, GPU=GPU, data_path=data_path)

t1 = t()
"""
array_xr = p.predict_UV_with_CNN(BDclim.stations["name"].values,
                                 fast=False,
                                 verbose=True,
                                 plot=False,
                                 Z0_cond=Z0,
                                 peak_valley=peak_valley)
"""

#array_ideal_conditions = p.predict_ideal_conditions(['Col du Lac Blanc'], input_speed=3, input_dir=270)


t2 = t()
print(f'\nPredictions in {round(t1, t2)} seconds')

#t1 = t()
#wind_map, weights, nwp_data_initial, nwp_data, mnt_data = p.predict_map(year_0=2019, month_0=6, day_0=20, hour_0=12, year_1=2019, month_1=6, day_1=20, hour_1=12, dx=10_000, dy=15_000, peak_valley=peak_valley, Z0_cond=Z0)
#t2 = t()
#print(f'\nPredictions in {round(t1, t2)} seconds\n')
#t1 = t()

#wind_map1, coord, nwp_data_initial1, nwp_data1, mnt_data1 = p.predict_map_indexes(year_0=2019, month_0=6, day_0=20, hour_0=10, year_1=2019, month_1=6, day_1=20, hour_1=10, dx=20_000, dy=25_000, peak_valley=peak_valley, Z0_cond=Z0)

"""
lp = LineProfiler()
lp_wrapper = lp(p.predict_map_indexes)
lp_wrapper(year_0=2019, month_0=6, day_0=20, hour_0=15, year_1=2019, month_1=6, day_1=20, hour_1=15, dx=20_000, dy=25_000)
lp.print_stats()
"""
#t2 = t()
#print(f'\nPredictions tensorflow in {round(t1, t2)} seconds\n')

# Visualization
v = Visualization(p)
# v.plot_predictions_2D(array_xr, ['Col du Lac Blanc'])
# v.plot_predictions_3D(array_xr, ['Col du Lac Blanc'])
# v.plot_comparison_topography_MNT_NWP(station_name='Col du Lac Blanc', new_figure=False)

# Evaluation
#e = Evaluation(v, array_xr)


t_end = t()
print(f"\n All prediction in  {round(t_init, t_end)/60} minutes")
#e.plot_time_serie(array_xr, 'Col du Lac Blanc', year=year_begin)
"""
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(mnt_data1[0,:,:])

U = wind_map1[0, :, :, 0]
V = wind_map1[0, :, :, 1]
UV = np.sqrt(U**2+V**2)
plt.figure()
plt.imshow(UV)
plt.contourf(coord[0], coord[1], UV)

plt.figure()
x = mnt_data.x.data
y = mnt_data.y.data
z = UV
plt.contourf(x, y, z)
plt.colorbar()

plt.figure()
x = nwp_data_initial.X_L93.data
y = nwp_data_initial.Y_L93.data
z = nwp_data_initial.Wind.data[0, :, :]
plt.contourf(x, y, z)
plt.colorbar()

plt.figure()
plt.imshow(UV)
plt.axis('equal')
plt.colorbar()

import matplotlib.pyplot as plt
U = wind_map[0, :, :, 0]
V = wind_map[0, :, :, 1]
UV = np.sqrt(U**2+V**2)

#mnt_data = mnt_data[0, :, :]
x_mnt = mnt_data.x
y_mnt = mnt_data.y
XX, YY = np.meshgrid(x_mnt, y_mnt)

nwp_data_initial = nwp_data_initial.assign(theta=lambda x: (np.pi / 180) * (x["Wind_DIR"] % 360))
nwp_data_initial = nwp_data_initial.assign(U=lambda x: -x["Wind"] * np.sin(x["theta"]))
nwp_data_initial = nwp_data_initial.assign(V=lambda x: -x["Wind"] * np.cos(x["theta"]))

x_nwp_initial = nwp_data_initial["X_L93"].data
y_nwp_initial = nwp_data_initial["Y_L93"].data
U_nwp_initial = nwp_data_initial["U"].isel(time=0).data
V_nwp_initial = nwp_data_initial["V"].isel(time=0).data

x_nwp = nwp_data["X_L93"].data
y_nwp = nwp_data["Y_L93"].data
U_nwp = nwp_data["U"].isel(time=0).data
V_nwp = nwp_data["V"].isel(time=0).data

ax = plt.gca()
arrows = ax.quiver(XX, YY, U, V, UV, angles='xy', scale_units='xy', scale=1)
ax.quiver(x_nwp_initial, y_nwp_initial, U_nwp_initial, V_nwp_initial, color='red', angles='xy', scale_units='xy', scale=1)
ax.quiver(x_nwp, y_nwp, U_nwp, V_nwp, color='pink', angles='xy', scale_units='xy', scale=1)

for i in range(5):
    for j in range(5):
        ax.text(x_nwp_initial[i, j] - 100, y_nwp_initial[i, j] + 100, str(np.round(nwp_data_initial.Wind.isel(time=0).data[i, j], 1)) + " m/s", color='red')
        ax.text(x_nwp_initial[i, j] + 100, y_nwp_initial[i, j] - 100, str(np.round(nwp_data_initial.Wind_DIR.isel(time=0).data[i, j])) + '°', color='red')

plt.figure()
plt.contourf(mnt_data.x, mnt_data.y, UV, levels = np.linspace(0, 15, 30), cmap='cividis')
plt.colorbar()

x_station = BDclim.stations['X'].values
y_station = BDclim.stations['Y'].values
x_min_mnt = mnt_data.x.min()
x_max_mnt = mnt_data.x.max()
y_min_mnt = mnt_data.y.min()
y_max_mnt = mnt_data.y.max()
name = BDclim.stations['name'].values
ax = plt.gca()
for i in range(len(x_station)):
    if (x_min_mnt<x_station[i]<x_max_mnt) and (y_min_mnt<y_station[i]<y_max_mnt):
        ax.text(x_station[i], y_station[i], name[i])
        ax.plot(x_station[i], y_station[i], marker='x', color='red')
plt.contour(mnt_data.x, mnt_data.y, mnt_data[0], levels=[500*i for i in range(11)], colors='black', linewidths=1)


import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d  # Warning
fig = plt.figure()
ax = fig.gca(projection='3d')
XX, YY = np.meshgrid(mnt_data.x, mnt_data.y)
ZZ = mnt_data[0]
color_dimension = UV
minn, maxx = np.nanmin(color_dimension), np.nanmax(color_dimension)
norm = colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(cmap="cividis")
m.set_array([])
fcolors = m.to_rgba(color_dimension)
ax.plot_surface(XX, YY, ZZ,
                cmap="cividis",
                facecolors=fcolors,
                linewidth=0,
                shade=False)
cb = plt.colorbar(m)
v = Visualization(p)
v._set_axes_equal(ax)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Wind predictions with CNN")


import matplotlib.pyplot as plt
for hour in range(23):
    wind = np.load(f"hour_{hour}")
    U = wind[0, :, :, 0]
    V = wind[0, :, :, 1]
    UV = np.sqrt(U**2+V**2)
    plt.figure()
    plt.imshow(UV)
    
"""