from time import time as t
import pandas as pd

t_init = t()

# Create prm
from PRM_predict import create_prm
prm = create_prm(month_prediction=True)

from downscale.data_source.observation import Observation
from downscale.data_source.MNT import MNT
from downscale.data_source.NWP import NWP
from downscale.operators.devine import Devine
from downscale.visu.visualization import Visualization
from downscale.eval.evaluation import Evaluation
from downscale.utils.utils_func import round

# BDclim
#IGN = MNT(prm=prm)
#AROME = NWP(begin=prm["begin"], end=prm["end"], prm=prm)
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)
#p = Devine(mnt=IGN, nwp=AROME, prm=prm)
#v = Visualization(p)
#e = Evaluation(v)

# Quality control
if prm["launch_predictions"]:
    BDclim.qc(prm)
    BDclim.time_series.to_pickle(prm["QC_pkl"])
    BDclim.replace_obs_by_QC_obs(prm=prm, drop_not_valid=True)
    BDclim.time_series.to_pickle(prm["QC_pkl_no_qc"])
else:
    BDclim.time_series = pd.read_pickle(prm["QC_pkl"])

print(f"\n All prediction in  {round(t_init, t()) / 60} minutes")


def replace_obs_by_QC_obs(time_series_qc_all, replace_old_wind=True, drop_not_valid=True,
                          remove_Dome_and_Vallot_suspicious=False, verbose=True):
    print(time_series_qc_all.head())
    if replace_old_wind:
        # Speed
        wind_corrected_not_nan = ~time_series_qc_all["wind_corrected"].isna()
        time_series_qc_all.loc[wind_corrected_not_nan, 'vw10m(m/s)'] = time_series_qc_all.loc[wind_corrected_not_nan, "wind_corrected"]
        time_series_qc_all['wind_corrected'] = time_series_qc_all['vw10m(m/s)']
        time_series_qc_all['UV'] = time_series_qc_all['vw10m(m/s)']

        # Direction
        time_series_qc_all['winddir(deg)'] = time_series_qc_all["UV_DIR"]
        time_series_qc_all["Wind_DIR"] = time_series_qc_all["UV_DIR"]
        print("__old wind name replaced by corrected wind name") if verbose else None

    time_series_qc = time_series_qc_all

    if drop_not_valid:
        time_series_qc = apply_qc_filters_to_time_series(time_series_qc)
        assert len(time_series_qc) != len(time_series_qc_all)
    else:
        time_series_qc = apply_qc_filters_to_time_series(time_series_qc, create_qc=True)

    return time_series_qc