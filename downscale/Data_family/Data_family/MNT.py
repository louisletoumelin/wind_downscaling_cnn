import numpy as np
import xarray as xr
from time import time as t

from downscale.Data_family.Data_2D import Data_2D

try:
    import dask
    _dask = True
except ModuleNotFoundError:
    _dask = False

try:
    import rasterio
    _rasterio = True
except ModuleNotFoundError:
    _rasterio = False


class MNT(Data_2D):
    _dask = _dask
    _rasterio = _rasterio

    def __init__(self, path_to_file=None, name=None, resolution_x=25, resolution_y=25, prm=None):
        print("\nBegin MNT creation")
        t0 = t()

        path_to_file = prm["topo_path"] if prm["topo_path"] is not None else path_to_file
        name = prm["name_mnt"] if prm["name_mnt"] is not None else name
        resolution_x = prm["resolution_mnt_x"] if prm["resolution_mnt_x"] is not None else resolution_x
        resolution_y = prm["resolution_mnt_y"] if prm["resolution_mnt_y"] is not None else resolution_y

        # Inherit from Data
        super().__init__(path_to_file, name)

        # Load MNT with xr.open_rasterio or xr.open_dataset
        self._mnt_loaded = False
        self.load_mnt_files(path_to_file, chunks=None) if path_to_file is not None else None

        # Corners of MNT
        if self._mnt_loaded:
            self.get_mnt_characteristics(resolution_x=resolution_x, resolution_y=resolution_y, name=name,
                                        path_to_file=path_to_file, verbose=True)

        print(f"MNT created in {np.round(t()-t0, 2)} seconds\n")

    def get_mnt_characteristics(self, resolution_x=None, resolution_y=None, name=None, verbose=False, path_to_file=None):

        self.x_min = np.min(self.data_xr.x.data)
        self.y_max = np.max(self.data_xr.y.data)
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.name = name

        if verbose:
            print(f"____MNT characteristics:")
            print(f"________________________ Resolution_x :{resolution_x}")
            print(f"________________________ Resolution_y :{resolution_y}")
            print(f"________________________ Name :{name}")
            print(f"________________________ File found at :{path_to_file}")

    def load_mnt_files(self, path_to_file, verbose=True, chunks=None):
        if _rasterio:
            if not _dask:
                chunks = None
            self.data_xr = xr.open_rasterio(path_to_file, chunks=chunks).astype(np.float32, copy=False)
            self.data = self.data_xr.values[0, :, :]
            print(f"__Used xr.open_rasterio to open MNT and {chunks} chunks") if verbose else None
            self._mnt_loaded = True
        else:
            self.data_xr = xr.open_dataset(path_to_file).astype(np.float32, copy=False)
            self.data = self.data_xr.__xarray_dataarray_variable__.data[0, :, :]
            print("__Used xr.open_dataset to open MNT") if verbose else None
            self._mnt_loaded = True

    def find_nearest_MNT_index(self, x, y,
                               look_for_corners=True, xmin_MNT=None, ymax_MNT=None,
                               look_for_resolution=True, resolution_x=25, resolution_y=25):

        if look_for_corners:
            xmin_MNT = self.x_min
            ymax_MNT = self.y_max

        if look_for_resolution:
            resolution_x = self.resolution_x
            resolution_y = self.resolution_y

        index_x_MNT = np.intp((x - xmin_MNT) // resolution_x)
        index_y_MNT = np.intp((ymax_MNT - y) // resolution_y)

        return index_x_MNT, index_y_MNT

    def _get_mnt_data_and_shape(self, mnt_data):
        """
        This function takes as input a mnt and returns data, coordinates and shape
        """

        if self._dask:
            shape_x_mnt, shape_y_mnt = mnt_data.data.shape[1:]
            mnt_data_x = mnt_data.x.data.astype(np.float32)
            mnt_data_y = mnt_data.y.data.astype(np.float32)
            mnt_data = mnt_data.data.astype(np.float32)
        else:
            mnt_data_x = mnt_data.x.data.astype(np.float32)
            mnt_data_y = mnt_data.y.data.astype(np.float32)
            mnt_data = mnt_data.__xarray_dataarray_variable__.data.astype(np.float32)
            shape_x_mnt, shape_y_mnt = mnt_data[0, :, :].shape
        return mnt_data, mnt_data_x, mnt_data_y, shape_x_mnt, shape_y_mnt

    @property
    def shape(self):
        return self.data.shape

