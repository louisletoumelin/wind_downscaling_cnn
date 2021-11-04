from time import time as t

import numpy as np
import xarray as xr

try:
    from numba import jit, prange, float64, float32, int32, int64

    _numba = True
except ModuleNotFoundError:
    _numba = False

from downscale.Data_family.MNT import MNT
from downscale.Utils.Decorators import print_func_executed_decorator, timer_decorator, check_type_kwargs_inputs
from downscale.Operators.wind_utils import Wind_utils


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
        return xarray_data

    @print_func_executed_decorator("interpolating xarray", level_begin="\n__", level_end="__")
    @timer_decorator("interpolating xarray", unit="second", level=". . ")
    def interpolate_wind_grid_xarray(self, nwp_data, interp=2, method='linear', verbose=True):

        if verbose:
            t0 = t()

        # Calculate U_nwp and V_nwp
        nwp_data = self.horizontal_wind_component(library="xarray", xarray_data=nwp_data, unit_direction="degree")

        # Drop variables
        nwp_data = nwp_data.drop_vars(["Wind", "Wind_DIR"])

        # Interpolate AROME
        nwp_data = self.interpolate_xarray_grid(xarray_data=nwp_data, interp=interp, method=method)
        nwp_data = self.compute_speed_and_direction_xarray(xarray_data=nwp_data)

        return nwp_data

    @print_func_executed_decorator("final interpolation", level_begin="\n__", level_end="__")
    @timer_decorator("final interpolation", unit="second", level=". . ")
    def interpolate_final_result(self, wind_map, library='numba', verbose=True):
        if library == 'numba' and _numba:
            jit_int = jit([float32[:, :, :, :](float32[:, :, :, :])], nopython=True)(self._interpolate_array)
            result = jit_int(wind_map)
            print("____Library: Numba") if verbose else None
        else:
            result = self._interpolate_array(wind_map)
            print("____Library: Numpy") if verbose else None
        return result

    @staticmethod
    def _interpolate_array(wind_map):
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

    @print_func_executed_decorator("interpolation mean KNN", level_begin="\n__", level_end="__")
    @timer_decorator("interpolation mean KNN", unit="second", level=". . . . ")
    @check_type_kwargs_inputs({"high_resolution_wind": [xr.core.dataarray.DataArray, xr.core.dataset.Dataset],
                               "low_resolution_grid": [xr.core.dataarray.DataArray, xr.core.dataset.Dataset]})
    def interpolation_mean_K_NN(self, high_resolution_wind=None, high_resolution_grid=None,
                                low_resolution_grid=None, length_square=None,
                                x_name_LR="x", y_name_LR="y", x_name_HR="x", y_name_HR="y",
                                resolution_HR_x=None, resolution_HR_y=None):

        assert resolution_HR_x == resolution_HR_y

        min_mnt = np.nanmin(high_resolution_grid[x_name_HR].values)
        max_mnt = np.nanmax(high_resolution_grid[y_name_HR].values)

        indexes_x, indexes_y = self.find_nearest_MNT_index(low_resolution_grid[x_name_LR],
                                                           low_resolution_grid[y_name_LR],
                                                           look_for_corners=False,
                                                           xmin_MNT=min_mnt,
                                                           ymax_MNT=max_mnt,
                                                           look_for_resolution=False,
                                                           resolution_x=resolution_HR_x)

        nb_time_steps = high_resolution_wind.Wind.shape[0]
        nb_y_px = low_resolution_grid.ZS.shape[0]
        nb_x_px = low_resolution_grid.ZS.shape[1]
        shape_low_resolution_wind = (nb_time_steps, nb_y_px, nb_x_px, 3)
        array_LR_wind = np.empty(shape_low_resolution_wind)

        U = high_resolution_wind["U"]
        V = high_resolution_wind["V"]
        W = high_resolution_wind["W"]

        nb_pixels = length_square // (2 * resolution_HR_x)
        for idx_y in range(nb_y_px):
            for idx_x in range(nb_x_px):
                for idx_wind_component, wind_component in enumerate([U, V, W]):
                    for time in range(nb_time_steps):
                        x = indexes_x[idx_x]
                        y = indexes_y[idx_y]
                        array_LR_wind[time, idx_y, idx_x, idx_wind_component] = np.mean(
                            wind_component.values[time, y - nb_pixels:y + nb_pixels, x - nb_pixels:x + nb_pixels])

        low_resolution_grid = low_resolution_grid.assign(U=(("time", y_name_LR, x_name_LR), array_LR_wind[:, :, :, 0]))
        low_resolution_grid = low_resolution_grid.assign(V=(("time", y_name_LR, x_name_LR), array_LR_wind[:, :, :, 1]))
        low_resolution_grid = low_resolution_grid.assign(W=(("time", y_name_LR, x_name_LR), array_LR_wind[:, :, :, 2]))

        return low_resolution_grid
