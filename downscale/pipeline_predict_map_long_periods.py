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
import pandas as pd
import tensorflow as tf


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

# Create prm
prm = create_prm(month_prediction=True)

# Initialize horovod and GPU
if prm["GPU"]: connect_GPU_to_horovod()

"""
MNT, NWP and observations
"""


# IGN
IGN = MNT(prm["topo_path"], name="IGN")

# AROME
AROME = NWP(prm["selected_path"],
            name="AROME",
            begin=prm["begin"],
            end=prm["end"],
            save_path=prm["save_path"],
            path_Z0_2018=None,
            path_Z0_2019=None,
            path_to_file_npy=prm["path_to_file_npy"],
            verbose=prm["verbose"],
            load_z0=False,
            save=False)

# BDclim
BDclim = Observation(prm["BDclim_stations_path"],
                     prm["BDclim_data_path"],
                     begin=prm["begin"],
                     end=prm["end"],
                     select_date_time_serie=prm["select_date_time_serie"],
                     GPU=prm["GPU"])

if not(prm["GPU"]):
    number_of_neighbors = 4
    BDclim.update_stations_with_KNN_from_NWP(number_of_neighbors, AROME)
    BDclim.update_stations_with_KNN_from_MNT_using_cKDTree(IGN)

"""
Processing, visualization and evaluation
"""

results = {}
results["cnn"] = {}

t1 = t()
if prm["launch_predictions"]:

    # Iterate on weeks
    dates = pd.date_range(start=prm["begin"], end=prm["end"], freq="7D")
    dates_shift = pd.date_range(start=prm["begin"], end=prm["end"], freq="7D").shift()

    for index, (date_begin, date_end) in enumerate(zip(dates, dates_shift)):

        print(f"Begin: {date_begin}")

        # Be sure dates are correct
        date_begin_day = date_begin.day if index == 0 else date_begin.day + 1

        if date_end > pd.to_datetime(prm["end"]):
            date_end = pd.to_datetime(prm["end"])

        print(f"End: {date_end}")

        # AROME
        AROME = NWP(prm["selected_path"],
                    name="AROME",
                    begin=str(date_begin.year) + "-" + str(date_begin.month) + "-" + str(date_begin.day),
                    end=str(date_end.year) + "-" + str(date_end.month) + "-" + str(date_end.day),
                    save_path=prm["save_path"],
                    path_Z0_2018=prm["path_Z0_2018"],
                    path_Z0_2019=prm["path_Z0_2019"],
                    path_to_file_npy=prm["path_to_file_npy"],
                    verbose=prm["verbose"],
                    load_z0=prm["load_z0"],
                    save=prm["save_z0"])

        # Processing
        p = Processing(obs=BDclim,
                       mnt=IGN,
                       nwp=AROME,
                       model_path=prm['model_path'],
                       GPU=prm["GPU"],
                       data_path=prm['data_path'])

        # Select function
        predict = p.predict_maps

        #todo load appropriate NWP files
        wind_map, acceleration, time, stations, _, _ = predict(year_0=date_begin.year,
                                                         month_0=date_begin.month,
                                                         day_0=date_begin_day,
                                                         hour_0=0,
                                                         year_1=date_end.year,
                                                         month_1=date_end.month,
                                                         day_1=date_end.day,
                                                         hour_1=23,
                                                         dx=prm["dx"],
                                                         dy=prm["dy"],
                                                         peak_valley=prm["peak_valley"],
                                                         Z0_cond=prm["Z0"],
                                                         type_rotation = prm["type_rotation"],
                                                         line_profile=prm["line_profile"],
                                                         memory_profile=prm["memory_profile"],
                                                         interp=prm["interp"],
                                                         nb_pixels=prm["nb_pixels"],
                                                         interpolate_final_map=prm["interpolate_final_map"],
                                                         extract_stations_only=prm["extract_stations_only"])
        if index == 0:
            wind_map_all=wind_map
            acceleration_all=acceleration
            time_all=time
        else:
            wind_map_all = np.concatenate((wind_map_all, wind_map))
            acceleration_all = np.concatenate((acceleration_all, acceleration))
            time_all = np.concatenate((time_all, time))

    print(wind_map_all.shape, acceleration_all.shape, time_all.shape)

for index, station in enumerate(stations):
    UV = np.sqrt(wind_map_all[:, index, 0]**2 + wind_map_all[:, index, 1]**2)
    results["cnn"][station] = pd.DataFrame(UV, index=time_all)


t2 = t()
print(f'\nPredictions in {round(t1, t2)} seconds')

# Visualization
v = Visualization(p)

# Evaluation
if prm["launch_predictions"]: e = Evaluation(v, array_xr=None)

t_end = t()
print(f"\n All prediction in  {round(t_init, t_end) / 60} minutes")