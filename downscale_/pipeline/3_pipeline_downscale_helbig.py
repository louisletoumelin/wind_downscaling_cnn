import matplotlib as mpl
#mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 2_000_000
from time import time as t
import numpy as np

# Create prm
from PRM_predict import create_prm
prm = create_prm(month_prediction=True)

from downscale.data_source.MNT import MNT
from downscale.data_source.NWP import NWP
from downscale.data_source.observation import Observation
from downscale.operators.helbig import DwnscHelbig
from downscale.utils.context_managers import timer_context
from downscale.visu.visualization import Visualization


IGN = MNT(prm=prm)
AROME = NWP(prm["selected_path"], begin=prm["begin"], end=prm["end"], prm=prm)
AROME.data_xr = AROME.data_xr.isel(time=0)
AROME.convert_to_mnt_format(extract_wind=True)
AROME_interpolated = AROME.data_xr.interp_like(IGN.data_xr, method="nearest")
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)

#IGN.data_xr = IGN.data_xr.sel(x=slice(900_000, 950_000), y=slice(6_450_000, 6_500_000))


mnt = IGN.data_xr.data[0, :, :]

d = DwnscHelbig()
# mu_mnt = d.mu_helbig_map(mnt, dx=25)
# laplacian_mnt = d.laplacian_map(mnt, dx=25)

with timer_context("downscale Helbig et al. 2017", level="", unit="minute", verbose=True):
    output_downscale = d.downscale_helbig(mnt, idx_x=None, idx_y=None, type_input="map", library="numba")

#result = AROME_interpolated.data * output_downscale.reshape((1, output_downscale.shape[0], output_downscale.shape[1]))
#print(result.shape)



"""
import matplotlib.pyplot as plt

mnt_small = IGN.data[330:12_300, 420:7_200]
x_min = np.nanmin(IGN.data_xr.x.data[420:7_200])
y_max = np.nanmax(IGN.data_xr.y.data[330:12_300])

stations = BDclim.stations
resolution_x = 25
resolution_y = 25
idx_x, idx_y = IGN.find_nearest_MNT_index(stations["X"], stations["Y"])

mu_mnt_idx = downscale_1.mu_helbig_idx(mnt, dx=25, idx_x=idx_x, idx_y=idx_y)
laplacian_mnt_idx = downscale_1.laplacian_idx(mnt, dx=25, idx_x=idx_x, idx_y=idx_y)
x_dsc_topo_idx = downscale_1.x_dsc_topo_helbig(mnt, dx=25, idx_x=idx_x, idx_y=idx_y, type_input="indexes")

downscale_1 = DwnscHelbig()

mu_mnt = downscale_1.mu_helbig_map(mnt_small, dx=25)
laplacian_mnt = downscale_1.laplacian_map(mnt_small, dx=25)
x_dsc_topo = downscale_1.x_dsc_topo_helbig(mnt_small)

plt.figure()
plt.imshow(mnt_small)
plt.colorbar()
plt.figure()
plt.imshow(laplacian_mnt, vmin=-0.5, vmax=0.5)
plt.colorbar()
plt.title("Laplacian")
plt.figure()
plt.imshow(mu_mnt)
plt.colorbar()
plt.title("Mu")
plt.figure()
plt.imshow(x_dsc_topo)
plt.colorbar()
plt.title("x_dsc_topo")

plt.figure()
plt.scatter(laplacian_mnt, mu_mnt, c=x_dsc_topo, s=3)
plt.scatter(laplacian_mnt_idx, mu_mnt_idx, c="red", s=3)
plt.xlabel("Laplacian")
plt.ylabel("Mu")
plt.colorbar()
plt.savefig("laplacian_mu2.png")
"""