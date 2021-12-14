from time import time as t
import pickle

t0 = t()

import numpy as np
import pandas as pd

from PRM_predict import create_prm
prm = create_prm(month_prediction=True)

from downscale.operators.devine import Devine
from downscale.visu.visualization import Visualization
from downscale.data_source.MNT import MNT
from downscale.data_source.NWP import NWP
from downscale.data_source.observation import Observation
from downscale.eval.evaluation import Evaluation
from downscale.utils.GPU import connect_GPU_to_horovod
from downscale.utils.utils_func import select_range_30_days_for_long_periods_prediction
from utils_prm import update_selected_path_for_long_periods, select_stations

connect_GPU_to_horovod() if prm["GPU"] else None

IGN = MNT(prm=prm)
AROME = NWP(prm["AROME_path_1"], begin=prm["begin"], end=prm["begin_after"], prm=prm)
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)
prm = select_stations(prm, BDclim)

p = Devine(obs=BDclim, mnt=IGN, nwp=AROME, prm=prm)
p.update_stations_with_neighbors(mnt=IGN, nwp=AROME, GPU=prm["GPU"], number_of_neighbors=4, interpolated=False)

data_xr_interp = p.interpolate_wind_grid_xarray(AROME.data_xr.isel(time=0),
                                                interp=prm["interp"], method=prm["method"], verbose=prm["verbose"])
AROME.data_xr = data_xr_interp
p.update_stations_with_neighbors(mnt=IGN, nwp=AROME, GPU=prm["GPU"], number_of_neighbors=4, interpolated=True)

"""
Processing, visualization and evaluation
"""

results = {}
for variable in prm["variable"]:
    results[variable] = {}
    for model in ["nwp", "cnn", "obs"]:
        results[variable][model] = {}
        for station in prm["stations_to_predict"]:
            results[variable][model][station] = []

if prm["launch_predictions"]:

    # Iterate on weeks
    begins, ends = select_range_30_days_for_long_periods_prediction(begin=prm["begin"], end=prm["end"], GPU=prm["GPU"])

    for index, (begin, end) in enumerate(zip(begins, ends)):

        t1 = t()

        # Update the name of the file to load
        prm = update_selected_path_for_long_periods(begin, end, prm)

        print(f"\nBegin: {begin}")
        print(f"End: {end}\n")
        print("\nSelected path here:")
        print(prm["selected_path"])

        # Load NWP
        AROME = NWP(begin=begin,
                    end=end,
                    prm=prm)

        # Processing
        p = Devine(obs=BDclim, mnt=IGN, nwp=AROME, prm=prm)

        # Interpolate
        data_xr_interp = p.interpolate_wind_grid_xarray(AROME.data_xr,
                                                        interp=prm["interp"],
                                                        method=prm["method"],
                                                        verbose=prm["verbose"])
        AROME.data_xr = data_xr_interp

        # Processing with interpolated data
        p = Devine(obs=BDclim, mnt=IGN, nwp=AROME, prm=prm)

        # Predict
        array_xr = p.predict_at_stations(prm["stations_to_predict"], prm=prm)

        # Visualization
        v = Visualization(p)

        # Analysis
        e = Evaluation(v, array_xr)

        # Store nwp, cnn predictions and observations
        for variable in prm["variable"]:
            for station in prm["stations_to_predict"]:
                nwp, cnn, obs = e._select_dataframe(array_xr,
                                                    begin=begin,
                                                    end=end,
                                                    station_name=station,
                                                    variable=variable,
                                                    rolling_mean=None,
                                                    rolling_window=None,
                                                    interp_str=prm["interp_str"],
                                                    interp_str_nwp="",
                                                    extract_around=prm["extract_around"])
                results[variable]["nwp"][station].append(nwp)
                results[variable]["cnn"][station].append(cnn)
                results[variable]["obs"][station].append(obs)

        del p
        del v
        del e
        del array_xr
        del AROME

        time_to_predict_month = np.round(t() - t1, 2)
        print(f"\n Prediction for time between {begin} and {end}:"
              f"\n{time_to_predict_month / 60} minutes")

for variable in prm["variable"]:
    for station in prm["stations_to_predict"]:
        for metric in ["nwp", "cnn", "obs"]:
            results[variable][metric][station] = pd.concat(results[variable][metric][station])

path = "//scratch/mrmn/letoumelinl/predict_real/Results/" if prm["GPU"] else ''

# Save results
with open(path + prm["results_name"] + '.pickle', 'wb') as handle:
    pickle.dump(results, handle)

# Save prm
pd.DataFrame.from_dict(data=prm, orient='index').to_csv(path + "prm_" + prm['results_name'] + ".csv", header=False)

time_to_predict_all = np.round(t() - t0, 2)
print(f"\n All prediction in  {time_to_predict_all / 60} minutes")
