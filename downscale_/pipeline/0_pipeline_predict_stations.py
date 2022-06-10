from time import time as t
from datetime import datetime

import numpy as np
t_init = t()

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass

from PRM_predict import create_prm
prm = create_prm(month_prediction=True)

from downscale.visu.visualization import Visualization
from downscale.operators.devine import Devine
from downscale.eval.eval_from_dict import EvaluationFromDict
from downscale.data_source.MNT import MNT
from downscale.data_source.NWP import NWP
from downscale.data_source.observation import Observation
from downscale.utils.utils_func import save_figure

import matplotlib
#matplotlib.use('Agg')
matplotlib.use('Qt5Agg')

"""
#Stations
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




IGN = MNT(prm=prm)
AROME = NWP(begin=prm["begin"], end=prm["end"], prm=prm)
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)


BDclim.replace_obs_by_QC_obs(prm, replace_old_wind=True, drop_not_valid=False)

use_QC = True
QC = BDclim.time_series
date_selected = np.datetime64(datetime(prm["year_begin"], prm["month_begin"], prm["day_begin"], prm["hour_begin"]))

# Compute nearest neighbor if CPU, load them if GPU
if not prm["GPU"]:
    number_of_neighbors = 4
    BDclim.update_stations_with_KNN_from_NWP(number_of_neighbors=number_of_neighbors, nwp=AROME)
    BDclim.update_stations_with_KNN_from_MNT_using_cKDTree(IGN)

"""
#Processing, visualization and evaluation
"""

# Processing
p = Devine(obs=BDclim, mnt=IGN, nwp=AROME, prm=prm)
p.update_stations_with_neighbors(mnt=IGN, nwp=AROME, GPU=prm["GPU"], number_of_neighbors=4, interpolated=True)

t1 = t()
if prm["launch_predictions"]:

    if prm["stations_to_predict"] == 'all':
        prm["stations_to_predict"] = BDclim.stations["name"].values

    array_xr = p.predict_at_stations(prm["stations_to_predict"], prm=prm)

print(f'\nPredictions in {np.round(t()-t1, 2)} seconds')

v = Visualization(p)
"""
prm["hour_begin"] = 11  # 1 vent d'Ouest
prm["day_begin"] = 7  # 2
prm["month_begin"] = 3  # 8
prm["year_begin"] = 2018  # 2017
date_selected = np.datetime64(datetime(prm["year_begin"], prm["month_begin"], prm["day_begin"], prm["hour_begin"]))
v.plot_predictions_2D(array_xr, ['Col du Lac Blanc'], random_selection=False, date_selected=date_selected, n=2, scale=1/0.04, fontsize=40, vmin=1.38/2, vmax=1.38*2)
save_figure(f"map_example_{date_selected}", prm)

prm["hour_begin"] = 0 # vent de sud est
prm["day_begin"] = 9
prm["month_begin"] = 4
prm["year_begin"] = 2021
date_selected = np.datetime64(datetime(prm["year_begin"], prm["month_begin"], prm["day_begin"], prm["hour_begin"]))
v.plot_predictions_2D(array_xr, ['Col du Lac Blanc'], random_selection=False, date_selected=date_selected, n=2,
                      scale=1/0.04, fontsize=40, vmin=2.43/2, vmax=2.43*2)
save_figure(f"map_example_{date_selected}", prm)


prm["hour_begin"] = 9
prm["day_begin"] = 6
prm["month_begin"] = 4
prm["year_begin"] = 2021
date_selected = np.datetime64(datetime(prm["year_begin"], prm["month_begin"], prm["day_begin"], prm["hour_begin"]))
v.plot_predictions_2D(array_xr, ['Col du Lac Blanc'], random_selection=False, date_selected=date_selected, n=2,
                      scale=1/0.04, fontsize=40, vmin=5.52/2, vmax=5.52*2)
save_figure(f"map_example_{date_selected}", prm)
"""
