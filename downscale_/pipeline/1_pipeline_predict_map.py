from time import time as t
import uuid

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
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Create prm
from PRM_predict import create_prm
prm = create_prm(month_prediction=True)

from downscale.operators.devine import Devine
from downscale.data_source.MNT import MNT
from downscale.data_source.NWP import NWP
from downscale.data_source.observation import Observation
from downscale.visu import Visualization

DEM = MNT(prm=prm)

begin = np.datetime64(datetime(prm["year_begin"], prm["month_begin"], prm["day_begin"], prm["hour_begin"]))
end = np.datetime64(datetime(prm["year_end"], prm["month_end"], prm["day_end"], prm["hour_end"] + 1))

AROME = NWP(prm["selected_path"], begin=begin, end=end, prm=prm)
AROME.data_xr = AROME.data_xr.isel(time=[0])
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)
p = Devine(obs=BDclim, mnt=DEM, nwp=AROME, prm=prm)
v = Visualization(p=p, prm=prm)


#if not (prm["GPU"]):
#    number_of_neighbors = 4
#    BDclim.update_stations_with_KNN_from_NWP(number_of_neighbors=number_of_neighbors, nwp=AROME)
#    BDclim.update_stations_with_KNN_from_MNT_using_cKDTree(DEM)

# Processing
p = Devine(obs=BDclim, mnt=DEM, nwp=AROME, prm=prm)


# p.update_stations_with_neighbors(mnt=DEM, nwp=AROME, GPU=prm["GPU"], number_of_neighbors=4, interpolated=False)
#surfex = xr.open_dataset(prm["path_SURFEX"])


if prm["launch_predictions"]:
    predict = p.predict_maps
    t1 = t()
    wind_xr = predict(prm=prm)
    print(f'\nPredictions in {np.round((t() - t1))} seconds')
    wind_xr = p.compute_speed_and_direction_xarray(xarray_data=wind_xr)
    #wind_xr = p.interpolate_mean_K_NN(high_resolution_wind=wind_xr, high_resolution_grid=p.mnt.data_xr,
    #                                  low_resolution_grid=surfex, length_square=250,
    #                                  x_name_LR="x", y_name_LR="y", x_name_HR="x", y_name_HR="y",
    #                                  resolution_HR_x=30, resolution_HR_y=30)

# Visualization
v = Visualization(p)

config = dict(
    scale_arrow=1 / 9,
    width_arrow=3,
    lateral_extent=500,
    idx_cross_section=700,
    scale_cross_section=300,
    width_cross_section=0.001,
    nb_arrows_cross_section=3,
    color_cross_section="grey",
    height_arrows_above_cross_section=500,
    cmap_wind_speed="magma",
    cmap_nwp_arrows="viridis",
    cmap_alpha="coolwarm",
    add_station_arrow=False,
    fontsize=50

)

v.figure_maps_large_domain(wind_xr, config=config, prm=prm)
