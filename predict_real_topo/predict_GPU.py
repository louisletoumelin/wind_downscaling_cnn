import numpy as np
from time import time as t

def round(t1, t2):
    return (np.round(t2 - t1, 2))


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

# General
data_path = "C:/Users/louis/git/wind_downscaling_CNN/Data/1_Raw/"
data_path = "//scratch/mrmn/letoumelinl/"
experience_path = "C:/Users/louis/git/wind_downscaling_CNN/Models/ARPS/"
experience_path = "//scratch/mrmn/letoumelinl/ARPS/training_results/"


# Topography
topo_path = data_path + "MNT/IGN_25m/ign_L93_25m_alpesIG.tif"

# NWP
AROME_path_1 = data_path + "AROME/FORCING_alp_2017080106_2018080106.nc"
AROME_path_2 = data_path + "AROME/FORCING_alp_2018080106_2019060106.nc"
AROME_path_3 = data_path + "AROME/FORCING_alp_2019060107_2019070106.nc"
AROME_path = [AROME_path_1, AROME_path_2, AROME_path_3]

# Z0
path_Z0_2018 = data_path + "AROME/Z0/Z0_alp_2018010100_2018120700.nc"
path_Z0_2019 = data_path + "AROME/Z0/Z0_alp_20190103_20191227.nc"

# Observation
BDclim_stations_path = data_path + "BDclim/precise_localisation/liste_postes_alps_l93.csv"
BDclim_data_path = data_path + "BDclim/extract_BDClim_et_sta_alp_20171101_20190501.csv"

# CNN model
model_experience = "date_16_02_name_simu_FINAL_1_0_model_UNet/"
model_path = experience_path + model_experience

"""
Time
"""

# Date to predict
day_begin = 1
month_begin = 6
year_begin = 2019

day_end = 30
month_end = 6
year_end = 2019

begin = str(year_begin) + "-" + str(month_begin) + "-" + str(day_begin)
end = str(year_end) + "-" + str(month_end) + "-" + str(day_end)

"""
MNT, NWP and observations
"""
t0 = t()
# IGN
IGN = MNT(topo_path, "IGN")
t1 = t()
print(f'\nMNT loaded in {round(t0, t1)} seconds')

# AROME
t2 = t()
#AROME = NWP(AROME_path_3, "AROME", begin, end, save_path=None, path_Z0_2018=path_Z0_2018, path_Z0_2019=path_Z0_2019, verbose=True)
AROME = NWP(AROME_path_3, "AROME", begin, end, save_path=None, path_Z0_2018=None, path_Z0_2019=None, verbose=True)
t3 = t()
print(f'\nNWP loaded in {round(t2, t3)} seconds')

# BDclim
t4 = t()
BDclim = Observation(BDclim_stations_path, BDclim_data_path, begin=begin, end=end)
# BDclim.stations_to_gdf(ccrs.epsg(2154), x="X", y="Y")
number_of_neighbors = 4
#BDclim.update_stations_with_KNN_from_NWP(number_of_neighbors, AROME)
#BDclim.update_stations_with_KNN_from_MNT_using_cKDTree(IGN)
t5 = t()
print(f'\nObservations loaded in {round(t4, t5)} seconds')

"""
Processing, visualization and evaluation
"""

# Processing
p = Processing(BDclim, IGN, AROME, model_path)
# ['Col du Lac Blanc', 'Col du Lautaret', 'SEYNOD-AREA', 'LE PLENAY', 'MEYTHET', 'BARCELONNETTE', 'DIGNE LES BAINS', 'RESTEFOND-NIVOSE', 'ST JEAN-ST-NICOLAS', 'TALLARD', "VILLAR D'ARENE", 'LE GUA-NIVOSE', "ALPE-D'HUEZ", 'LA MURE- RADOME'],
t1 = t()

array_xr = p.predict_UV_with_CNN(['Col du Lac Blanc'],
                                 fast=False,
                                 verbose=True,
                                 plot=False,
                                 Z0_cond=False)
t2 = t()
print(f'\nProcessing in {round(t1, t2)} seconds')

t1 = t()
#wind_map, weights, nwp_data_initial, nwp_data, mnt_data, alpha = p.predict_map(year_0=2019, month_0=6, day_0=20, hour_0=15,year_1=2019, month_1=6, day_1=20, hour_1=15, dx=2000, dy=2000)
#wind_map, weights, nwp_data_initial, nwp_data, mnt_data = p.predict_map(year_0=2019, month_0=6, day_0=18, hour_0=10, year_1=2019, month_1=6, day_1=18, hour_1=10, dx=2_000, dy=2_000)
t2 = t()
print(f'\nProcessing in {round(t1, t2)} seconds')

# Visualization
v = Visualization(p)
# v.plot_predictions_2D(array_xr, ['Col du Lac Blanc'])
# v.plot_predictions_3D(array_xr, ['Col du Lac Blanc'])
# v.plot_comparison_topography_MNT_NWP(station_name='Col du Lac Blanc', new_figure=False)

# Evaluation
#e = Evaluation(v, array_xr)
t8 = t()
print(f"\n{round(t0, t8) / 60} minutes")
#e.plot_time_serie(array_xr, 'Col du Lac Blanc', year=year_begin)
"""
import matplotlib.pyplot as plt
U = wind_map[0, :, :, 0]
V = wind_map[0, :, :, 1]
UV = np.sqrt(U**2+V**2)

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
plt.imshow(weights[0])
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
plt.contourf(mnt_data.x, mnt_data.y, UV, levels = list(range(15)), cmap='cividis')
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
"""