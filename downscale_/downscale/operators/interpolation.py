from time import time as t

import numpy as np
import xarray as xr

try:
    from numba import jit, prange, float64, float32, int32, int64

    _numba = True
except ModuleNotFoundError:
    _numba = False

from downscale.data_source.MNT import MNT
from downscale.utils.decorators import print_func_executed_decorator, timer_decorator, check_type_kwargs_inputs
from downscale.utils.utils_func import change_dtype_if_required
from downscale.operators.wind_utils import Wind_utils


class Interpolation(MNT, Wind_utils):

    def __init__(self):
        super().__init__()

    @staticmethod
    @print_func_executed_decorator("interpolate xarray grid", level_begin="____", level_end="____")
    @timer_decorator("interpolate xarray grid", unit="second", level=". . . . ")
    def interpolate_xarray_grid(xarray_data=None, interp=None, name_x='xx', name_y='yy', method='linear'):
        """
        Interpolate a regular grid on an xarray dataframe using multilinear interpolation.
        The interp parameters control the upsampling.

        New dimensions = old dimension * interp


        Parameters
        ----------
        xarray_data : xarray dataframe
        name_x: str
            name of the variable containing x coordinates
        name_y: str
            name of the variable containing y coordinates
        interp : int
            Interpolation parameters. New dimensions = old dimension * interp

        Returns
        -------
        xarray_data : xarray dataframe
            Xarray DataFrame interpolated
        """
        new_x = np.linspace(xarray_data[name_x].min().data, xarray_data[name_x].max().data,
                            xarray_data.dims[name_x] * interp)
        new_y = np.linspace(xarray_data[name_y].min().data, xarray_data[name_y].max().data,
                            xarray_data.dims[name_y] * interp)
        xarray_data = xarray_data.interp(xx=new_x, yy=new_y, method=method)
        if name_x in xarray_data.keys():
            xarray_data = xarray_data.drop(name_x)
        if name_y in xarray_data.keys():
            xarray_data = xarray_data.drop(name_y)
        return xarray_data

    @print_func_executed_decorator("interpolating xarray", level_begin="\n__", level_end="__")
    @timer_decorator("interpolating xarray", unit="second", level=". . ")
    def interpolate_wind_grid_xarray(self, nwp_data, interp=2, method='linear', u_name="U", v_name="V", verbose=True):

        # Calculate U_nwp and V_nwp
        if u_name not in nwp_data.keys() and v_name not in nwp_data.keys():
            nwp_data = self.horizontal_wind_component(library="xarray", xarray_data=nwp_data, unit_direction="degree")

        # Drop variables
        if "Wind" in nwp_data.keys() and "Wind_DIR" in nwp_data.keys():
            nwp_data = nwp_data.drop_vars(["Wind", "Wind_DIR"])

        # Interpolate AROME
        nwp_data = self.interpolate_xarray_grid(xarray_data=nwp_data, interp=interp, method=method)
        nwp_data = self.compute_speed_and_direction_xarray(xarray_data=nwp_data)

        return nwp_data

    @print_func_executed_decorator("final interpolation", level_begin="\n__", level_end="__")
    @timer_decorator("final interpolation", unit="second", level=". . ")
    def interpolate_nan_in_wind_map(self, wind_map, library='numba', verbose=True):
        if library == 'numba' and _numba:
            jit_int = jit([float32[:, :, :, :](float32[:, :, :, :])], nopython=True)(self._interpolate_nans_in_wind_map)
            result = jit_int(wind_map)
            print("____Library: Numba") if verbose else None
        else:
            result = self._interpolate_nans_in_wind_map(wind_map)
            print("____Library: Numpy") if verbose else None
        return result

    @staticmethod
    def _interpolate_nans_in_wind_map(wind_map):
        for time_step in range(wind_map.shape[0]):
            for component in range(wind_map.shape[3]):
                # Select component to interpolate
                wind_component = wind_map[time_step, :, :, component]
                nan_indexes = np.argwhere(np.isnan(wind_component))
                for y, x in nan_indexes:
                    right = (y, x + 1)
                    left = (y, x - 1)
                    up = (y + 1, x)
                    down = (y - 1, x)
                    neighbors_indexes = [right, left, up, down]
                    neighbors = np.array([wind_component[index] for index in neighbors_indexes])
                    if np.isnan(neighbors).sum() <= 3:
                        wind_map[time_step, y, x, component] = np.mean(neighbors[~np.isnan(neighbors)])
        return wind_map

    @staticmethod
    def _interpolate_mean_KNN_num(array_LR_wind, U, V, W, nb_y_px, nb_x_px, indexes_x, indexes_y, nb_time_steps, nb_pixels):
        for idx_y in range(nb_y_px):
            for idx_x in range(nb_x_px):
                x = indexes_x[idx_x]
                y = indexes_y[idx_y]
                for idx_wind_component, wind_component in enumerate([U, V, W]):
                    for time in range(nb_time_steps):
                        array_LR_wind[time, idx_y, idx_x, idx_wind_component] = np.mean(
                            wind_component[time, y - nb_pixels:y + nb_pixels, x - nb_pixels:x + nb_pixels])
        return array_LR_wind

    def _interpolate_mean_KNN(self, array_LR_wind, U, V, W, nb_y_px, nb_x_px, indexes_x, indexes_y, nb_time_steps, nb_pixels,
                              library="numba", verbose=True):
        if library == 'numba' and _numba:
            jit_int = jit([float32[:, :, :, :]
                           (float32[:, :, :, :], float32[:, :, :], float32[:, :, :], float32[:, :, :],
                            int32, int32, int32[:], int32[:],
                            int32, int32)], nopython=True)(self._interpolate_mean_KNN_num)
            result = jit_int(array_LR_wind, U, V, W, nb_y_px, nb_x_px, indexes_x, indexes_y, nb_time_steps, nb_pixels)
            print("____Library: Numba") if verbose else None
        else:
            result = self._interpolate_mean_KNN_num(array_LR_wind, U, V, W, nb_y_px, nb_x_px,
                                                    indexes_x, indexes_y, nb_time_steps, nb_pixels)
            print("____Library: Numpy") if verbose else None
        return result

    @print_func_executed_decorator("interpolation mean KNN", level_begin="\n__", level_end="__")
    @timer_decorator("interpolation mean KNN", unit="second", level=". . . . ")
    @check_type_kwargs_inputs({"high_resolution_wind": [xr.core.dataarray.DataArray, xr.core.dataset.Dataset],
                               "low_resolution_grid": [xr.core.dataarray.DataArray, xr.core.dataset.Dataset]})
    def interpolate_mean_K_NN(self, high_resolution_wind=None, high_resolution_grid=None,
                                low_resolution_grid=None, length_square=None,
                                x_name_LR="x", y_name_LR="y", x_name_HR="x", y_name_HR="y",
                                resolution_HR_x=None, resolution_HR_y=None, library="numba", verbose=True):

        assert resolution_HR_x == resolution_HR_y

        min_mnt = np.nanmin(high_resolution_grid[x_name_HR].values)
        max_mnt = np.nanmax(high_resolution_grid[y_name_HR].values)

        indexes_x, indexes_y = self.find_nearest_MNT_index(low_resolution_grid[x_name_LR].values,
                                                           low_resolution_grid[y_name_LR].values,
                                                           look_for_corners=False,
                                                           xmin_MNT=min_mnt,
                                                           ymax_MNT=max_mnt,
                                                           look_for_resolution=False,
                                                           resolution_x=resolution_HR_x)

        # Prepare variables
        indexes_x = change_dtype_if_required(indexes_x, np.int32)                                    #int32[:]
        indexes_y = change_dtype_if_required(indexes_y, np.int32)                                    #int32[:]
        nb_time_steps = np.int32(high_resolution_wind.Wind.shape[0])                                 #int32
        nb_px_y = np.int32(low_resolution_grid.ZS.shape[0])                                          #int32
        nb_px_x = np.int32(low_resolution_grid.ZS.shape[1])                                          #int32
        shape_low_resolution_wind = (nb_time_steps, nb_px_y, nb_px_x, 3)
        array_LR_wind = np.empty(shape_low_resolution_wind).astype(np.float32)                       #float32[:,:,:,:]

        U = high_resolution_wind["U"].values.astype(np.float32)                                      #float32[:,:,:]
        V = high_resolution_wind["V"].values.astype(np.float32)                                      #float32[:,:,:]
        W = high_resolution_wind["W"].values.astype(np.float32)                                      #float32[:,:,:]

        nb_pixels = np.int32(length_square // (2 * resolution_HR_x))                                 #int32

        # Interpolate
        array_LR_wind = self._interpolate_mean_KNN(array_LR_wind, U, V, W, nb_px_y, nb_px_x, indexes_x,
                                                   indexes_y, nb_time_steps, nb_pixels,
                                                   library=library, verbose=verbose)

        # Save results in xarray
        low_resolution_grid = low_resolution_grid.assign(U=(("time", y_name_LR, x_name_LR), array_LR_wind[:, :, :, 0]))
        low_resolution_grid = low_resolution_grid.assign(V=(("time", y_name_LR, x_name_LR), array_LR_wind[:, :, :, 1]))
        low_resolution_grid = low_resolution_grid.assign(W=(("time", y_name_LR, x_name_LR), array_LR_wind[:, :, :, 2]))

        return low_resolution_grid
