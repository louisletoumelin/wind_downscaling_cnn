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

import xarray as xr

from downscale.Operators.Processing import Processing
from downscale.Analysis.Visualization import Visualization
from downscale.Data_family.MNT import MNT
from downscale.Data_family.NWP import NWP
from downscale.Analysis.Evaluation import Evaluation
from PRM_predict import create_prm
from downscale.Utils.GPU import connect_GPU_to_horovod, check_connection_GPU
from downscale.Utils.Utils import round, select_range_7days_for_long_periods_prediction, select_range_30_days_for_long_periods_prediction
from downscale.Utils.prm import update_selected_path_for_long_periods

# Create prm, horovod and GPU
prm = create_prm(month_prediction=True)
connect_GPU_to_horovod() if prm["GPU"] else None
check_connection_GPU(prm) if prm["GPU"] else None

IGN = MNT(prm=prm)
AROME = NWP(begin=prm["begin"], end=prm["end"], prm=prm)
p = Processing(mnt=IGN, nwp=AROME, prm=prm)

"""
Processing, visualization and evaluation
"""

datasets = []

t1 = t()
if prm["launch_predictions"]:

    # Iterate on weeks
    begins, ends = select_range_7days_for_long_periods_prediction(begin=prm["begin"], end=prm["end"])
    surfex = xr.open_dataset(prm["path_SURFEX"])

    for index, (begin, end) in enumerate(zip(begins, ends)):

        print(f"\n\nBegin: {begin}")
        print(f"End: {end}")

        # Update the name of the file to load
        prm = update_selected_path_for_long_periods(begin, end, prm)

        # Load NWP
        AROME = NWP(begin=str(begin.year) + "-" + str(begin.month) + "-" + str(begin.day),
                    end=str(end.year) + "-" + str(end.month) + "-" + str(end.day),
                    prm=prm)

        # Processing
        p = Processing(mnt=IGN, nwp=AROME, prm=prm)
        wind_xr = p.predict_maps(prm=prm)
        wind_xr = wind_xr.interp_like(surfex, method="linear")
        datasets.append(wind_xr)

        print(f"\n Prediction between {begin} and {end} in {round(t1, t()) / 60} minutes")

    wind_xr = xr.concat(datasets, dim="time")
    wind_xr.to_netcdf(prm["path_save_prediction_on_surfex_grid"] + prm["results_name"] + ".nc")

    # Visualization and evaluation
    v = Visualization(p)
    e = Evaluation(v)

    print(f"\n All prediction in {round(t_init, t()) / 60} minutes")

