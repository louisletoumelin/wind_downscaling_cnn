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
from downscale.Utils.Utils import round, select_range_7days_for_long_periods_prediction
from downscale.Utils.prm import update_selected_path_for_long_periods

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

# Create prm, horovod and GPU
prm = create_prm(month_prediction=True)
connect_GPU_to_horovod() if prm["GPU"] else None


IGN = MNT(prm["topo_path"], name="IGN")
AROME = NWP(prm["selected_path"], name="AROME", begin=prm["begin"], end=prm["end"], prm=prm)
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)
p = Processing(obs=BDclim, mnt=IGN, nwp=AROME, model_path=prm['model_path'], prm=prm)
p.update_stations_with_neighbors(mnt=IGN, nwp=AROME, GPU=prm["GPU"], number_of_neighbors=4)

"""
Processing, visualization and evaluation
"""

results = {}
results["cnn"] = {}

t1 = t()
if prm["launch_predictions"]:

    # Iterate on weeks
    begins, ends = select_range_7days_for_long_periods_prediction(begin=prm["begin"], end=prm["end"])

    for index, (begin, end) in enumerate(zip(begins, ends)):

        print(f"Begin: {begin}")
        print(f"End: {end}")

        # Update the name of the file to load
        prm = update_selected_path_for_long_periods(begin, end, prm)

        # Load NWP
        AROME = NWP(path_to_file=prm["selected_path"],
                    name="AROME",
                    begin=str(begin.year) + "-" + str(begin.month) + "-" + str(begin.day),
                    end=str(end.year) + "-" + str(end.month) + "-" + str(end.day),
                    prm=prm)

        # Processing
        p = Processing(obs=BDclim, mnt=IGN, nwp=AROME, model_path=prm['model_path'], prm=prm)
        wind_map, acceleration, time, stations, _, _ = p.predict_maps(year_begin=begin.year,
                                                                      month_begin=begin.month,
                                                                      day_begin=begin.day,
                                                                      hour_begin=begin.hour,
                                                                      year_end=end.year,
                                                                      month_end=end.month,
                                                                      day_end=end.day,
                                                                      hour_end=end.hour,
                                                                      prm=prm)

        wind_map_all = np.concatenate((wind_map_all, wind_map)) if index != 0 else wind_map
        acceleration_all = np.concatenate((acceleration_all, acceleration)) if index != 0 else acceleration
        time_all = np.concatenate((time_all, time)) if index != 0 else time

for index, station in enumerate(stations):
    UV = p.compute_wind_speed(wind_map_all[:, index, 0], wind_map_all[:, index, 1])
    results["cnn"][station] = pd.DataFrame(UV, index=time_all)

print(f'\nPredictions in {round(t1, t())} seconds')

# Visualization and evaluation
v = Visualization(p) if prm["launch_predictions"] else None
e = Evaluation(v, array_xr=None) if prm["launch_predictions"] else None

print(f"\n All prediction in  {round(t_init, t()) / 60} minutes")
