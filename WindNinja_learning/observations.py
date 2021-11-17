import numpy as np
import pandas as pd


def load_observations(path_to_file, list_variables_str = ['index_AROME_NN_0_ref_AROME']):
    BDclim = pd.read_csv(path_to_file)
    BDclim[list_variables_str] = BDclim[list_variables_str].apply(lambda x: x.apply(eval))
    return BDclim


def select_idx_station_in_NWP_grid(stations, station_name, prm={}):
    idx_str = f"index_{prm['nwp_name']}_NN_0_ref_{prm['nwp_name']}"
    y_idx_nwp, x_idx_nwp = stations[idx_str][stations["name"] == station_name].values[0]
    return np.int16(y_idx_nwp), np.int16(x_idx_nwp)


def lower_station_name(station):
    return station.lower().replace("'", "_").replace("-", "_").replace(" ", "_")