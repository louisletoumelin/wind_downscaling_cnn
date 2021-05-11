import numpy as np
import pandas as pd
import xarray as xr
from time import time as t

from Data_2D import Data_2D

try:
    import dask

    _dask = True
except:
    _dask = False


class MNT(Data_2D):
    _dask = _dask

    def __init__(self, path_to_file, name=None, resolution_x=25, resolution_y=25):
        t0 = t()

        # Inherit from Data
        super().__init__(path_to_file, name)

        # Load MNT with xr.open_rasterio or xr.open_dataset
        self.load_mnt_files(path_to_file)

        # Corners of MNT
        self.x_min = np.min(self.data_xr.x.data)
        self.y_max = np.max(self.data_xr.y.data)
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y

        # Name
        self.name = name

        t1 = t()
        print(f"\nMNT created in {np.round(t1-t0, 2)} seconds\n")

    def load_mnt_files(self, path_to_file, verbose=True):
        if _dask:
            self.data_xr = xr.open_rasterio(path_to_file).astype(np.float32, copy=False)
            self.data = self.data_xr.values[0, :, :]
            if verbose: print("__Used xr.open_rasterio to open MNT")
        else:
            self.data_xr = xr.open_dataset(path_to_file).astype(np.float32, copy=False)
            self.data = self.data_xr.__xarray_dataarray_variable__.data[0, :, :]
            if verbose: print("__Used xr.open_dataset to open MNT")

    def find_nearest_MNT_index(self, x, y,
                               look_for_corners=True, xmin_MNT=None, ymax_MNT=None,
                               look_for_resolution=True, resolution_x=25, resolution_y=25):

        if look_for_corners:
            xmin_MNT = self.x_min
            ymax_MNT = self.y_max

        if look_for_resolution:
            resolution_x = self.resolution_x
            resolution_y = self.resolution_y

        index_x_MNT = (x - xmin_MNT) // resolution_x
        index_y_MNT = (ymax_MNT - y) // resolution_y
        return (index_x_MNT, index_y_MNT)

    @property
    def shape(self):
        return (self.data.shape)

