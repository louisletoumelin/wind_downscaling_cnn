from time import time as t

t0 = t()

import numpy as np
import pandas as pd

print("Current directory")
import os
try:
    os.chdir("//home/mrmn/letoumelinl")
except FileNotFoundError:
    pass
print(os.getcwd())

from downscale.Operators.Processing import Processing
from downscale.Analysis.Visualization import Visualization
from downscale.Data_family.MNT import MNT
from downscale.Data_family.NWP import NWP
from downscale.Data_family.Observation import Observation
from downscale.Analysis.Evaluation import Evaluation
from PRM_predict import create_prm
from downscale.Utils.GPU import connect_GPU_to_horovod
from downscale.Utils.Utils import round, select_range_30_days_for_long_periods_prediction
from downscale.Utils.prm import update_selected_path_for_long_periods, select_stations

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
AROME = NWP(prm["AROME_path_2"], name="AROME", begin=prm["begin"], end=prm["begin_after"], prm=prm)
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)
prm = select_stations(prm, BDclim)

p = Processing(obs=BDclim, mnt=IGN, nwp=AROME, model_path=prm['model_path'], prm=prm)
p.update_stations_with_neighbors(mnt=IGN, nwp=AROME, GPU=prm["GPU"], number_of_neighbors=4, interpolated=False)

data_xr_interp = p.interpolate_wind_grid_xarray(AROME.data_xr.isel(time=0),
                                                interp=prm["interp"], method=prm["method"], verbose=prm["verbose"])
AROME.data_xr = data_xr_interp
p.update_stations_with_neighbors(mnt=IGN, nwp=AROME, GPU=prm["GPU"], number_of_neighbors=4, interpolated=True)


"""
Processing, visualization and evaluation
"""

results = {}
results["nwp"] = {}
results["cnn"] = {}
results["obs"] = {}

for station in prm["stations_to_predict"]:
    results["nwp"][station] = []
    results["cnn"][station] = []
    results["obs"][station] = []

if prm["launch_predictions"]:

    # Iterate on weeks
    begins, ends = select_range_30_days_for_long_periods_prediction(begin=prm["begin"], end=prm["end"])

    for index, (begin, end) in enumerate(zip(begins, ends)):

        t1 = t()

        print(f"\nBegin: {begin}")
        print(f"End: {end}\n")
        # Initialize results

        # Update the name of the file to load
        prm = update_selected_path_for_long_periods(begin, end, prm)

        # Load NWP
        AROME = NWP(path_to_file=prm["selected_path"],
                    name="AROME",
                    begin=str(begin.year) + "-" + str(begin.month) + "-" + str(begin.day),
                    end=str(end.year) + "-" + str(end.month) + "-" + str(end.day),
                    prm=prm)
        print("\nSelected path here:")
        print(prm["selected_path"])

        print("\n NWP data here:")
        print(AROME.data_xr)

        # Processing
        p = Processing(obs=BDclim, mnt=IGN, nwp=AROME, model_path=prm['model_path'], prm=prm)

        # Interpolate
        data_xr_interp = p.interpolate_wind_grid_xarray(AROME.data_xr,
                                                        interp=prm["interp"],
                                                        method=prm["method"],
                                                        verbose=prm["verbose"])
        AROME.data_xr = data_xr_interp

        print("\n NWP data here 2:")
        print(AROME.data_xr)

        # Processing with interpolated data
        p = Processing(obs=BDclim, mnt=IGN, nwp=AROME, model_path=prm['model_path'], prm=prm)

        # Predict
        array_xr = p.predict_at_stations(prm["stations_to_predict"], prm=prm)

        # Visualization
        v = Visualization(p)

        # Analysis
        e = Evaluation(v, array_xr)

        # Store nwp, cnn predictions and observations
        for station in prm["stations_to_predict"]:
            nwp, cnn, obs = e._select_dataframe(array_xr,
                                                begin=begin,
                                                end=end,
                                                station_name=station,
                                                variable=prm["variable"],
                                                rolling_mean=None,
                                                rolling_window=None,
                                                interp_str=prm["interp_str"],
                                                extract_around=prm["extract_around"])
            results["nwp"][station].append(nwp)
            results["cnn"][station].append(cnn)
            results["obs"][station].append(obs)


        del p
        del v
        del e
        del array_xr
        del AROME

        time_to_predict_month = np.round(t()-t1, 2)
        print(f"\n Prediction for time between {begin} and {end}:"
              f"\n{time_to_predict_month / 60} minutes")

for station in prm["stations_to_predict"]:
    for metric in ["nwp", "cnn", "obs"]:
        results[metric][station] = pd.concat(results[metric][station])

pd.to_pickle(results, prm["save_path"] + "Results/results.pkl")

time_to_predict_all = np.round(t()-t0, 2)
print(f"\n All prediction in  {time_to_predict_all / 60} minutes")
