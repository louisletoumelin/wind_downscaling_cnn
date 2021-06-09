import numpy as np
import matplotlib.pyplot as plt
from time import time as t

from MNT import MNT
from NWP import NWP
from topo_utils import *
from PRM_predict import create_prm, update_selected_path

# Create prm
prm = create_prm(month_prediction=True)

# IGN
IGN = MNT(prm["topo_path"], name="IGN")

# AROME
AROME = NWP(prm["selected_path"],
            name="AROME",
            begin=prm["begin"],
            end=prm["end"],
            save_path=prm["save_path"],
            path_Z0_2018=prm["path_Z0_2018"],
            path_Z0_2019=prm["path_Z0_2019"],
            path_to_file_npy=prm["path_to_file_npy"],
            verbose=prm["verbose"],
            load_z0=prm["load_z0"],
            save=prm["save_z0"])

AROME.data_xr = AROME.data_xr.assign_coords(x=("xx", AROME.data_xr.X_L93.data[0,:]))
AROME.data_xr = AROME.data_xr.assign_coords(y=("yy", AROME.data_xr.Y_L93.data[:,0]))
AROME.data_xr = AROME.data_xr.drop(("xx", "yy"), dim=None)
AROME.data_xr = AROME.data_xr["Wind"]
AROME.data_xr = AROME.data_xr.rename({"xx": "x", "yy":"y"})


IGN.data_xr = IGN.data_xr.astype(np.float32)
IGN.data_xr = IGN.data_xr.sel(x=slice(800_000, 950_000), y=slice(6_500_000, 6_450_000))


AROME_interpolated = AROME.data_xr.interp_like(IGN.data_xr, method="nearest")

mnt = IGN.data_xr.data[0, :, :]

downscale = Dwnsc_helbig()

t0 = t()
output_downscale = downscale.downscale_helbig(mnt, idx_x=None, idx_y=None, type="map", librairie="numba")
t1 = t()
print(t1-t0)

result = AROME_interpolated.data[:, 100:-100, 100:-100] * output_downscale.reshape((1, output_downscale.shape[0], output_downscale.shape[1]))
print(result.shape)
