from time import time as t

t_init = t()

import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass


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
from Utils import connect_GPU_to_horovod, select_range, check_save_and_load

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
To be modified
"""

GPU = False
horovod = False
Z0 = True
load_z0 = True
save_z0 = False
peak_valley = True
launch_predictions = True
select_date_time_serie = True
verbose = True
stations_to_predict = ['Col du Lac Blanc']
line_profile = False

# Date to predict
day_begin = 1
month_begin = 6
year_begin = 2019

day_end = 5
month_end = 6
year_end = 2019

begin = str(year_begin) + "-" + str(month_begin) + "-" + str(day_begin)
end = str(year_end) + "-" + str(month_end) + "-" + str(day_end)

"""
Utils
"""


# Safety
load_z0, save_z0 = check_save_and_load(load_z0, save_z0)

# Initialize horovod and GPU
if GPU and horovod: connect_GPU_to_horovod()

# Create prm
prm = create_prm(GPU=GPU, Z0=Z0, end=end, month_prediction=True)

"""
MNT, NWP and observations
"""


# IGN
IGN = MNT(prm["topo_path"],
          name="IGN")

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
                     GPU=GPU)#,
                     #path_vallot=prm["path_vallot"],
                     #path_saint_sorlin=prm["path_saint_sorlin"],
                     #path_argentiere=prm["path_argentiere"],
                     #path_Dome_Lac_Blanc=prm["path_Dome_Lac_Blanc"],
                     #path_Col_du_Lac_Blanc=prm["path_Col_du_Lac_Blanc"],
                     #path_Muzelle_Lac_Blanc=prm["path_Muzelle_Lac_Blanc"],
                     #path_Col_de_Porte=prm["path_Col_de_Porte"],
                     #path_Col_du_Lautaret=prm["path_Col_du_Lautaret"],

# Compute nearest neighbor sif CPU, load them if GPU
if not (GPU):
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

    if stations_to_predict == 'all':
        stations_to_predict = BDclim.stations["name"].values

    array_xr = p.predict_at_stations(stations_to_predict,
                                     verbose=True,
                                     Z0_cond=Z0,
                                     peak_valley=peak_valley,
                                     ideal_case=False,
                                     line_profile=line_profile)

t2 = t()
print(f'\nPredictions in {round(t1, t2)} seconds')

# Visualization
v = Visualization(p)
#v.qc_plot_last_flagged(stations=['Vallot', 'Argentiere'])
# v.plot_predictions_2D(array_xr, ['Col du Lac Blanc'])
# v.plot_predictions_3D(array_xr, ['Col du Lac Blanc'])
# v.plot_comparison_topography_MNT_NWP(station_name='Col du Lac Blanc', new_figure=False)

# Evaluation
if launch_predictions: e = Evaluation(v, array_xr)

# e.plot_time_serie(array_xr, 'Col du Lac Blanc', year=year_begin)

t_end = t()
print(f"\n All prediction in  {round(t_init, t_end) / 60} minutes")