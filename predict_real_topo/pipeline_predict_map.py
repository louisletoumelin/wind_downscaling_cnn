from time import time as t

t_init = t()
"""
1h 20km x 20km
CPU: 2min
GPU: 14 sec

24h 50km x 40km
GPU: 20min
"""
import numpy as np
import tensorflow as tf
#from line_profiler import LineProfiler


def round(t1, t2):  return (np.round(t2 - t1, 2))


from Processing import Processing
from Visualization import Visualization
from MNT import MNT
from NWP import NWP
from Observation import Observation
from Data_2D import Data_2D
from MidpointNormalize import MidpointNormalize
from Evaluation import Evaluation
from PRM_predict import create_prm, update_selected_path
from Utils import connect_GPU_to_horovod, select_range

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
       'LE PLENAY', 'SEYNOD-AREA', 'Col du Lac Blanc', 'Col du Lautaret', 'Vallot', 'Saint-Sorlin', 'Argentiere']
"""

"""
Simulation parameters
"""

GPU = False
horovod = False
Z0 = True
load_z0 = True
save_z0 = False
peak_valley = True
launch_predictions = True
select_date_time_serie=True
type_rotation = 'scipy' # 'indexes' or 'scipy'
verbose=True
line_profile=False
memory_profile=False

interp=2
nb_pixels=15
interpolate_final_map=True

dx = 20_000
dy = 25_000
hour_begin = 15
day_begin = 1
month_begin = 10
year_begin = 2018

hour_end = 15
day_end = 1
month_end = 10
year_end = 2018

begin = str(year_begin) + "-" + str(month_begin) + "-" + str(day_begin)
end = str(year_end) + "-" + str(month_end) + "-" + str(day_end)

"""
Utils
"""

# Safety
if load_z0:
    save_z0 = False
if save_z0:
    load_z0 = False

# Initialize horovod and GPU
if GPU: connect_GPU_to_horovod()

# Create prm
prm = create_prm(GPU, end=end, month_prediction=True, Z0=Z0)

"""
MNT, NWP and observations
"""
# IGN
IGN = MNT(prm["topo_path"], name="IGN")

# AROME
AROME = NWP(prm["selected_path"],
            name="AROME",
            begin=begin,
            end=end,
            save_path=prm["save_path"],
            path_Z0_2018=prm["path_Z0_2018"],
            path_Z0_2019=prm["path_Z0_2019"],
            path_to_file_npy=prm["path_to_file_npy"],
            verbose=verbose,
            load_z0=load_z0,
            save=save_z0)

# BDclim
BDclim = Observation(prm["BDclim_stations_path"],
                     prm["BDclim_data_path"],
                     begin=begin,
                     end=end,
                     select_date_time_serie=select_date_time_serie,
                     GPU=GPU)

if not(GPU):
    number_of_neighbors = 4
    BDclim.update_stations_with_KNN_from_NWP(number_of_neighbors, AROME)
    BDclim.update_stations_with_KNN_from_MNT_using_cKDTree(IGN)

"""
Processing, visualization and evaluation
"""

# Processing
p = Processing(obs=BDclim,
               mnt=IGN,
               nwp=AROME,
               model_path=prm['model_path'],
               GPU=GPU,
               data_path=prm['data_path'])
t1 = t()
if launch_predictions:

    predict = p.predict_maps

    ttest=t()
    wind_map, acceleration_all, coords, nwp_data_initial, nwp_data, mnt_data = predict(year_0=year_begin, month_0=month_begin,
                                                                      day_0=day_begin, hour_0=hour_begin,
                                                                      year_1=year_end, month_1=month_end,
                                                                      day_1=day_end, hour_1=hour_end,
                                                                      dx=dx, dy=dy,
                                                                      peak_valley=peak_valley, Z0_cond=Z0,
                                                                      type_rotation=type_rotation,
                                                                      line_profile=line_profile,
                                                                      memory_profile=memory_profile,
                                                                      interp=interp,
                                                                      nb_pixels=nb_pixels,
                                                                      interpolate_final_map=interpolate_final_map)
    ttest1=t()
    print(f'\nDownscaling scipy in {round(ttest, ttest1)} seconds')

t2 = t()
print(f'\nPredictions in {round(t1, t2)} seconds')

"""
lp = LineProfiler()
lp_wrapper = lp(p.predict_map_indexes)
lp_wrapper(year_0=2019, month_0=6, day_0=20, hour_0=15, year_1=2019, month_1=6, day_1=20, hour_1=15, dx=20_000, dy=25_000)
lp.print_stats()
"""

# Visualization
v = Visualization(p)

# Evaluation
if launch_predictions: e = Evaluation(v, array_xr=None)

t_end = t()
print(f"\n All prediction in  {round(t_init, t_end) / 60} minutes")

"""

plt.figure()
plt.imshow(mnt_data[0,:,:])

import matplotlib.pyplot as plt
U = wind_map[0, :, :, 0]
V = wind_map[0, :, :, 1]
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
x_mnt = coords[0]
y_mnt = coords[1]
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