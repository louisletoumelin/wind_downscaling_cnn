from time import time as t

t_init = t()

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass


from downscale.operators.devine import Devine
from downscale.visu.visualization import Visualization
from downscale.data_source.MNT import MNT
from downscale.data_source.NWP import NWP
from downscale.data_source.observation import Observation
from downscale.eval.evaluation import Evaluation
from PRM_predict import create_prm
from downscale.utils.GPU import connect_GPU_to_horovod
from downscale.utils.utils_func import round

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


"""
#utils
"""


# Create prm
prm = create_prm(month_prediction=True)

# Initialize horovod and GPU
connect_GPU_to_horovod() if (prm["GPU"] and prm["horovod"]) else None

# Create prm
prm = create_prm(month_prediction=True)

"""
#MNT, NWP and observations
"""


IGN = MNT(prm=prm)
AROME = NWP(begin=prm["begin"], end=prm["end"], prm=prm)
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)

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

print(f'\nPredictions in {round(t1, t())} seconds')

# Visualization
v = Visualization(p)
#v.qc_plot_last_flagged(stations=['Vallot', 'Argentiere'])
# v.plot_predictions_2D(array_xr, ['Col du Lac Blanc'])
# v.plot_predictions_3D(array_xr, ['Col du Lac Blanc'])
# v.plot_comparison_topography_MNT_NWP(station_name='Col du Lac Blanc', new_figure=False)

# Analysis
e = Evaluation(v, array_xr) if prm["launch_predictions"] else None
# e.plot_time_serie(array_xr, 'Col du Lac Blanc', year=year_begin)

print(f"\n All prediction in  {round(t_init, t()) / 60} minutes")