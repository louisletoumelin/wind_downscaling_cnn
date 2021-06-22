import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 2_000_000

import numpy as np
from time import time as t

from downscale.Data_family.MNT import MNT
from downscale.Data_family.NWP import NWP
from downscale.Operators.topo_utils import Dwnsc_helbig, Sgp_helbig, Topo_utils
from PRM_predict import create_prm

# Create prm
prm = create_prm(month_prediction=True)

# IGN
IGN = MNT(prm["topo_path"], name="IGN")

# AROME
AROME = NWP(prm["selected_path"], name="AROME",begin=prm["begin"], end=prm["end"], prm=prm)

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
#mu_mnt = downscale.mu_helbig_map(mnt, dx=25)
#laplacian_mnt = downscale.laplacian_map(mnt, dx=25)

t0 = t()
output_downscale = downscale.downscale_helbig(mnt, idx_x=None, idx_y=None, type="map", library="numba")
print(t()-t0)

result = AROME_interpolated.data * output_downscale.reshape((1, output_downscale.shape[0], output_downscale.shape[1]))
print(result.shape)