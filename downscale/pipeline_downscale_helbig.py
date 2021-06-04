import numpy as np
import matplotlib.pyplot as plt

from MNT import MNT
from topo_utils import *
from PRM_predict import create_prm, update_selected_path

# Create prm
prm = create_prm(month_prediction=True)

# IGN
IGN = MNT(prm["topo_path"], name="IGN")

mnt = IGN.data
mnt_small = mnt[6000:8200, 4000:6000]

subgrid = Sgp_helbig()
map = subgrid.subgrid_map(mnt_small)
plt.figure()
plt.imshow(map)

downscale = Dwnsc_helbig()
result = downscale.x_dsc_topo_helbig_map(mnt_small, 25)
plt.figure()
plt.imshow(result)