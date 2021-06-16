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


from downscale.Operators.Processing import Processing
from downscale.Analysis.Visualization import Visualization
from downscale.Data_family.MNT import MNT
from downscale.Data_family.NWP import NWP
from downscale.Data_family.Observation import Observation
from downscale.Analysis.Evaluation import Evaluation
from PRM_predict import create_prm
from downscale.Utils.GPU import connect_GPU_to_horovod
from downscale.Utils.Utils import round

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


IGN = MNT(prm["topo_path"], name="IGN")
AROME = NWP(prm["selected_path"], name="AROME", begin=prm["begin"], end=prm["end"], prm=prm)
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)

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

        # Be sure dates are correct
        date_begin_day = date_begin.day if index == 0 else date_begin.day + 1
        date_end = pd.to_datetime(prm["end"]) if date_end > pd.to_datetime(prm["end"]) else date_end

        print(f"Begin: {date_begin}")
        print(f"End: {date_end}")

        # AROME
        AROME = NWP(path_to_file=prm["selected_path"],
                    name="AROME",
                    begin=str(date_begin.year) + "-" + str(date_begin.month) + "-" + str(date_begin.day),
                    end=str(date_end.year) + "-" + str(date_end.month) + "-" + str(date_end.day),
                    prm=prm)

        # Processing
        p = Processing(obs=BDclim, mnt=IGN, nwp=AROME, model_path=prm['model_path'], prm=prm)

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
                                                         prm=prm)

        wind_map_all = np.concatenate((wind_map_all, wind_map)) if index != 0 else wind_map
        acceleration_all = np.concatenate((acceleration_all, acceleration)) if index != 0 else acceleration
        time_all = np.concatenate((time_all, time)) if index != 0 else time

    print(wind_map_all.shape, acceleration_all.shape, time_all.shape)

for index, station in enumerate(stations):
    UV = p.wu.compute_wind_speed(wind_map_all[:, index, 0], wind_map_all[:, index, 1])
    results["cnn"][station] = pd.DataFrame(UV, index=time_all)

print(f'\nPredictions in {round(t1, t())} seconds')

# Visualization and evaluation
v = Visualization(p) if prm["launch_predictions"] else None
e = Evaluation(v, array_xr=None) if prm["launch_predictions"] else None

print(f"\n All prediction in  {round(t_init, t()) / 60} minutes")