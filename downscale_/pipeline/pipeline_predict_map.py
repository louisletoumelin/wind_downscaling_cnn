from time import time as t

t_init = t()

"""
1h 50km x 40km
CPU: Downscaling with scipy rotation in 77.09 seconds
GPU: Downscaling with scipy rotation in 28.16 seconds

24h 50km x 40km
GPU: Downscaling scipy in 542.96 seconds (9 min)
By rule of three, this give 2 days and 2h for downscaling one year at 1h and 25m resolution
"""

import numpy as np
import xarray as xr
from datetime import datetime

# Create prm
from PRM_predict import create_prm
prm = create_prm(month_prediction=True)

from downscale.operators.devine import Devine
from downscale.data_source.MNT import MNT
from downscale.data_source.NWP import NWP
from downscale.data_source.observation import Observation
from downscale.utils.GPU import connect_GPU_to_horovod
from downscale.visu import Visualization

# Initialize horovod and GPU
connect_GPU_to_horovod() if prm["GPU"] else None

DEM = MNT(prm=prm)

begin = np.datetime64(datetime(prm["year_begin"], prm["month_begin"], prm["day_begin"], prm["hour_begin"]))
end = np.datetime64(datetime(prm["year_end"], prm["month_end"], prm["day_end"], prm["hour_end"] + 1))

AROME = NWP(prm["selected_path"], begin=begin, end=end, prm=prm)
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)
p = Devine(obs=BDclim, mnt=DEM, nwp=AROME, prm=prm)
v = Visualization(p=p, prm=prm)

"""
if not (prm["GPU"]):
    number_of_neighbors = 4
    BDclim.update_stations_with_KNN_from_NWP(number_of_neighbors=number_of_neighbors, nwp=AROME)
    BDclim.update_stations_with_KNN_from_MNT_using_cKDTree(DEM)

# Processing
p = Devine(obs=BDclim, mnt=DEM, nwp=AROME, prm=prm)
"""
"""
# p.update_stations_with_neighbors(mnt=DEM, nwp=AROME, GPU=prm["GPU"], number_of_neighbors=4, interpolated=False)
surfex = xr.open_dataset(prm["path_SURFEX"])

t1 = t()
if prm["launch_predictions"]:
    predict = p.predict_maps
    wind_xr = predict(prm=prm)
    wind_xr = p.compute_speed_and_direction_xarray(xarray_data=wind_xr)
    wind_xr = p.interpolate_mean_K_NN(high_resolution_wind=wind_xr, high_resolution_grid=p.mnt.data_xr,
                                      low_resolution_grid=surfex, length_square=250,
                                      x_name_LR="x", y_name_LR="y", x_name_HR="x", y_name_HR="y",
                                      resolution_HR_x=30, resolution_HR_y=30)
    wind_xr.isel(time=0).U.plot()

print(f'\nPredictions in {round(t1, t())} seconds')

# Visualization
v = Visualization(p)

# Analysis
e = Evaluation(v, array_xr=None) if prm["launch_predictions"] else None

print(f"\n All prediction in  {round(t_init, t()) / 60} minutes")


"""
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
