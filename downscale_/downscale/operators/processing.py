# Packages: numpy, pandas, xarray, scipy, tensorflow, matplotlib
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
import datetime
from time import time as t

# Local imports
from downscale.utils.GPU import environment_GPU
from downscale.utils.decorators import print_func_executed_decorator, timer_decorator, check_type_kwargs_inputs
from downscale.operators.rotation import Rotation
from downscale.operators.helbig import DwnscHelbig
from downscale.operators.micro_met import MicroMet
from downscale.operators.interpolation import Interpolation
from downscale.operators.generators import Generators

# try importing optional modules
try:
    from numba import jit, prange, float64, float32, int32, int64

    _numba = True
except ModuleNotFoundError:
    _numba = False

try:
    import numexpr as ne

    _numexpr = True
except ModuleNotFoundError:
    _numexpr = False


class Processing(DwnscHelbig, MicroMet, Rotation, Interpolation, Generators):
    n_rows, n_col = 79, 69

    def __init__(self, obs=None, mnt=None, nwp=None, prm=None):

        super().__init__()

        self.observation = obs if obs is not None else None
        self.mnt = mnt if mnt is not None else None
        self.nwp = nwp if nwp is not None else None
        self.model_path = prm.get("model_path")

        self.is_updated_with_topo_characteristics = False
        if prm.get("GPU") is not None:
            environment_GPU(GPU=prm["GPU"])

    def update_station_with_topo_characteristics(self):
        self.update_stations_with_laplacian()
        self.update_stations_with_tpi(radius=2000)
        self.update_stations_with_tpi(radius=500)
        self.update_stations_with_mu()
        self.update_stations_with_curvature()
        self.is_updated_with_topo_characteristics = True

    def update_stations_with_laplacian(self):

        stations = self.observation.stations
        mnt_name = self.mnt.name

        idx_x = stations[f"index_{mnt_name}_NN_0_cKDTree_ref_{mnt_name}"].str[0].values
        idx_y = stations[f"index_{mnt_name}_NN_0_cKDTree_ref_{mnt_name}"].str[1].values

        idx_x = idx_x.astype(np.int32)
        idx_y = idx_y.astype(np.int32)

        laplacians = self.laplacian_idx(self.mnt.data, idx_x, idx_y, self.mnt.resolution_x, helbig=False)

        self.observation.stations["laplacian"] = laplacians

    def update_stations_with_sx(self, sx_direction):

        stations = self.observation.stations
        mnt_name = self.mnt.name

        idx_x = stations[f"index_{mnt_name}_NN_0_cKDTree_ref_{mnt_name}"].str[0].values
        idx_y = stations[f"index_{mnt_name}_NN_0_cKDTree_ref_{mnt_name}"].str[1].values

        idx_x = idx_x.astype(np.int32)
        idx_y = idx_y.astype(np.int32)
        sx = self.sx_idx(self.mnt.data, idx_x, idx_y, cellsize=30, dmax=300, in_wind=sx_direction, wind_inc=5, wind_width=30)

        self.observation.stations["sx_300"] = sx

    def update_stations_with_tpi(self, radius=2000):

        stations = self.observation.stations
        mnt_name = self.mnt.name

        idx_x = stations[f"index_{mnt_name}_NN_0_cKDTree_ref_{mnt_name}"].str[0].values
        idx_y = stations[f"index_{mnt_name}_NN_0_cKDTree_ref_{mnt_name}"].str[1].values

        idx_x = idx_x.astype(np.int32)
        idx_y = idx_y.astype(np.int32)

        tpi = self.tpi_idx(self.mnt.data, idx_x, idx_y, radius, resolution=self.mnt.resolution_x)

        self.observation.stations[f"tpi_{str(int(radius))}"] = tpi

    def update_stations_with_mu(self):

        stations = self.observation.stations
        mnt_name = self.mnt.name

        idx_x = stations[f"index_{mnt_name}_NN_0_cKDTree_ref_{mnt_name}"].str[0].values
        idx_y = stations[f"index_{mnt_name}_NN_0_cKDTree_ref_{mnt_name}"].str[1].values

        idx_x = idx_x.astype(np.int32)
        idx_y = idx_y.astype(np.int32)

        mu = self.mu_helbig_idx(self.mnt.data, self.mnt.resolution_x, idx_x, idx_y)

        self.observation.stations["mu"] = mu

    def update_stations_with_curvature(self):

        stations = self.observation.stations
        mnt_name = self.mnt.name

        idx_x = stations[f"index_{mnt_name}_NN_0_cKDTree_ref_{mnt_name}"].str[0].values
        idx_y = stations[f"index_{mnt_name}_NN_0_cKDTree_ref_{mnt_name}"].str[1].values

        idx_x = idx_x.astype(np.int32)
        idx_y = idx_y.astype(np.int32)

        curvature = self.curvature_idx(self.mnt.data, idx_x, idx_y, method="fast", scale=False)

        self.observation.stations["curvature"] = curvature

    def update_stations_with_neighbors(self, mnt=None, nwp=None, GPU=False, number_of_neighbors=4, interpolated=False):

        if not GPU and not interpolated:
            mnt = self.mnt if mnt is None else mnt
            nwp = self.nwp if nwp is None else nwp
            self.observation.update_stations_with_KNN_from_NWP(nwp=nwp,
                                                               number_of_neighbors=number_of_neighbors,
                                                               interpolated=interpolated)
            self.observation.update_stations_with_KNN_from_MNT_using_cKDTree(mnt,
                                                                             number_of_neighbors=number_of_neighbors)

        if not GPU and interpolated:
            self.observation.update_stations_with_KNN_from_NWP(nwp=nwp,
                                                               number_of_neighbors=number_of_neighbors,
                                                               interpolated=interpolated)

            self.observation.update_stations_with_KNN_of_NWP_in_MNT_using_cKDTree(mnt, nwp,
                                                                                  number_of_neighbors=4,
                                                                                  interpolated=interpolated)

    @print_func_executed_decorator("extract_variable_from_nwp_at_station", level_begin="\n____", level_end="____")
    @timer_decorator("extract_variable_from_nwp_at_station", unit="second", level=". . . . ")
    def _extract_variable_from_nwp_at_station(self,
                                              station_name,
                                              variable_to_extract=["time", "wind_speed", "wind_direction", "Z0",
                                                                   "Z0REL", "ZS"],
                                              interp_str="",
                                              only_array=True,
                                              verbose=False):
        """
        Extract numpy arrays of specified variable at a specific station.
        Needs the station name. Return a list of arrays containing variable time series at the station.

        Parameters
        ----------
        station_name : string
            The considered station
        variable_to_extract : list
            List of variables to extract

        Returns
        -------
        results : List
            List containing numpy arrays (the variables to extract)
        """

        # Select station
        nwp_name = self.nwp.name
        stations = self.observation.stations
        idx_str = f"index_{nwp_name}_NN_0{interp_str}_ref_{nwp_name}{interp_str}"

        y_idx_nwp, x_idx_nwp = stations[idx_str][stations["name"] == station_name].values[0]
        y_idx_nwp, x_idx_nwp = np.int16(y_idx_nwp), np.int16(x_idx_nwp)

        # Select NWP data
        nwp_instance = self.nwp.data_xr

        results = [0 for k in range(len(variable_to_extract))]
        if "time" in variable_to_extract:
            time_index = nwp_instance.time.data
            results[variable_to_extract.index("time")] = time_index

        if "wind_speed" in variable_to_extract:
            if only_array:
                wind_speed = nwp_instance.Wind.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
            else:
                wind_speed = wind_speed = nwp_instance.Wind.isel(xx=x_idx_nwp, yy=y_idx_nwp)
            results[variable_to_extract.index("wind_speed")] = wind_speed

        if "wind_direction" in variable_to_extract:
            if only_array:
                wind_dir = nwp_instance.Wind_DIR.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
            else:
                wind_dir = nwp_instance.Wind_DIR.isel(xx=x_idx_nwp, yy=y_idx_nwp)
            results[variable_to_extract.index("wind_direction")] = wind_dir

        if "Z0" in variable_to_extract:
            if only_array:
                Z0 = nwp_instance.Z0.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
            else:
                Z0 = nwp_instance.Z0.isel(xx=x_idx_nwp, yy=y_idx_nwp)
            results[variable_to_extract.index("Z0")] = Z0

        if "Z0REL" in variable_to_extract:
            if only_array:
                Z0REL = nwp_instance.Z0REL.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
            else:
                Z0REL = nwp_instance.Z0REL.isel(xx=x_idx_nwp, yy=y_idx_nwp)
            results[variable_to_extract.index("Z0REL")] = Z0REL

        if "ZS" in variable_to_extract:
            if only_array:
                ZS = nwp_instance.ZS.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
            else:
                ZS = nwp_instance.ZS.isel(xx=x_idx_nwp, yy=y_idx_nwp)
            results[variable_to_extract.index("ZS")] = ZS

        print(f"Selected time series for pixel at station: {station_name}") if verbose else None

        results = results[0] if len(results) == 1 else results
        return results

    @print_func_executed_decorator("select timeframe NWP", level_begin="____", level_end="____")
    @timer_decorator("select timeframe NWP", unit="second", level=". . . . ")
    def _select_timeframe_nwp(self, begin=None, end=None, ideal_case=False, nwp=None, verbose=True):
        """
        Selects a timeframe for NWP.

        Parameters
        ----------
        begin : date, optional
            Date to begin slicing. Exemple: '2019-6-1' (Default: None)
        end : date, optional
            Date to end slicing. Exemple: '2019-6-30' (Default: None)
        ideal_case : boolean, optional
            If True, end == begin + 1 day.
            Used for ideal cases where we force wind speed and direction independently from NWP.

        Returns
        -------
        Updates nwp instance.
        """

        # Select timeframe
        if ideal_case:
            begin = nwp.begin
            year, month, day = np.int16(begin.split('-'))
            end = str(year) + "-" + str(month) + "-" + str(day + 1)

        self.nwp.select_timeframe(begin=begin, end=end)

        print(f"________Begin: {begin}. "
              f"\n________End: {end}") if verbose else None

    @print_func_executed_decorator("select large domain with coordinates", level_begin="____", level_end="____")
    @timer_decorator("select large domain with coordinates", unit="second", level=". . . . ")
    def _select_large_domain_with_coordinates(self, type_input="MNT",
                                              coords=None, epsg_coords="2154", dx_mnt=2_000, x_mnt="x", y_mnt="y",
                                              dx_nwp=100, x_nwp="X_L93", y_nwp="Y_L93", verbose=True):
        t0 = t() if verbose else None

        if epsg_coords != "2154":
            raise NotImplementedError("Please specify coordinates in L93 coordinate system")

        x_min, y_min, x_max, y_max = coords

        assert x_min < x_max
        assert y_min < y_max

        if type_input == "MNT":
            data_xr = self.mnt.data_xr
            x = x_mnt
            y = y_mnt
            dx = dx_mnt
        elif type_input == "NWP":
            data_xr = self.nwp.data_xr
            x = x_nwp
            y = y_nwp
            dx = dx_nwp
        else:
            raise NotImplementedError("We only support domain selection with coordinates and station name")

        x_min = x_min - dx
        x_max = x_max + dx
        y_min = y_min - dx
        y_max = y_max + dx

        assert x_min > np.min(data_xr[x])
        assert y_min > np.min(data_xr[y])
        assert x_max < np.max(data_xr[x])
        assert y_max < np.max(data_xr[y])

        mask = (x_min <= data_xr[x]) & (data_xr[x] <= x_max) & (y_min <= data_xr[y]) & (data_xr[y] <= y_max)
        data_xr = data_xr.where(mask, drop=True)

        if verbose:
            print(f"____Selected domain {type_input}. "
                  f"\n________Margin: {dx} m. "
                  f"\n________Final length: {np.nanmax(data_xr[x].values) - np.nanmin(data_xr[x].values)} m. "
                  f"\n________Final height {np.nanmax(data_xr[y].values) - np.nanmin(data_xr[y].values)} m")

        return data_xr

    @print_func_executed_decorator("Select large domain around station", level_begin="____", level_end="____")
    @timer_decorator("Select large domain around station", unit="second", level=". . . . ")
    def _select_large_domain_around_station(self, station_name, dx, dy, type_input="NWP", additional_dx_mnt=None,
                                            verbose=True):
        """

        This function operate on NWP or MNT and select and area around a station specified by its name

        Parameters
        ----------
        station_name : str
            Station name
        dx : float
            The domain length is 2*dx
        dy : float
            The domain length is 2*dx
        type_input: str
            "NWP" or "MNT"
        additional_dx_mnt: float
            Additional length to the side of the domain (Default: True)

            Usually used when extracting MNT data. As CNN predictions are performed on maps but the information is only
            stored on specific pixels near the center, we require more MNT data than the original domain size.


        Returns
        -------
        result : list
            Xarray DataFrame on the specified domain
        """
        t0 = t() if verbose else None

        stations = self.observation.stations
        if type_input == "NWP":
            nwp_name = self.nwp.name
            nwp_x, nwp_y = stations[f"{nwp_name}_NN_0"][stations["name"] == station_name].values[0]
            x_min = nwp_x - dx
            x_max = nwp_x + dx
            y_min = nwp_y - dy
            y_max = nwp_y + dy

            data_xr = self.nwp.data_xr
            mask = (x_min <= data_xr.X_L93) & (data_xr.X_L93 <= x_max) & (y_min <= data_xr.Y_L93) & (
                    data_xr.Y_L93 <= y_max)
            data_xr = data_xr.where(mask, drop=True)

        elif type_input == "MNT":

            # MNT domain must be larger than NWP domain as we extract MNT data around NWP data
            if additional_dx_mnt is not None:
                dx = dx + additional_dx_mnt
                dy = dy + additional_dx_mnt

            mnt_name = self.mnt.name
            mnt_x, mnt_y = stations[f"{mnt_name}_NN_0_cKDTree"][stations["name"] == station_name].values[0]

            x_min = mnt_x - dx
            x_max = mnt_x + dx
            y_min = mnt_y - dy
            y_max = mnt_y + dy

            data_xr = self.mnt.data_xr
            mask = (x_min <= data_xr.x) & (data_xr.x <= x_max) & (y_min <= data_xr.y) & (data_xr.y <= y_max)
            data_xr = data_xr.where(mask, drop=True)

        else:
            raise NotImplementedError

        print(f"____Time to select large domain around station: {np.round(t()-t0)} seconds")
        return data_xr

    @print_func_executed_decorator("prepare time and domain NWP", level_begin="\n__", level_end="__")
    @timer_decorator("prepare time and domain NWP", unit="second", level=". . ")
    def prepare_time_and_domain_nwp(self, year_0, month_0, day_0, hour_0, year_1, month_1, day_1, hour_1,
                                    select_area="station", station_name=None, dx=None, dy=None,
                                    coords=None, epsg_coords="2154",
                                    additional_dx_mnt=None, x_nwp="X_L93", y_nwp="Y_L93",
                                    additional_dx_nwp=100, verbose=True):
        if verbose:
            print(f"____Area selected using: {select_area}")

        # Select timeframe
        begin = datetime.datetime(year_0, month_0, day_0, hour_0)
        end = datetime.datetime(year_1, month_1, day_1, hour_1)
        self._select_timeframe_nwp(begin=begin, end=end)

        # Select area
        if select_area == "station":
            nwp_data = self._select_large_domain_around_station(station_name, dx, dy,
                                                                type_input="NWP",
                                                                additional_dx_mnt=additional_dx_mnt)
        elif select_area == "coord":
            nwp_data = self._select_large_domain_with_coordinates(coords=coords,
                                                                  epsg_coords=epsg_coords,
                                                                  type_input="NWP",
                                                                  x_nwp=x_nwp,
                                                                  y_nwp=y_nwp,
                                                                  dx_nwp=additional_dx_nwp)
        else:
            raise NotImplementedError

        nwp_data = nwp_data.isel(xx=slice(1, -1))

        nwp_data = nwp_data.isel(yy=slice(1, -1))

        assert nwp_data.isnull().sum() == 0

        return nwp_data

    @print_func_executed_decorator("get NWP characteristics", level_begin="\n__", level_end="__")
    @timer_decorator("get NWP characteristics", unit="second", level=". . ")
    def get_characteristics_nwp(self, nwp_data, verbose=True):

        t0 = t() if verbose else None

        times = nwp_data.time.data.astype(np.float32)
        nb_time_step = len(times)
        nwp_x_l93 = nwp_data.X_L93.data.astype(np.float32)
        nwp_y_l93 = nwp_data.Y_L93.data.astype(np.float32)
        nb_px_nwp_y, nb_px_nwp_x = nwp_x_l93.shape

        if verbose:
            print(f"____Selection NWP characteristics: {np.round(t()-t0)} seconds")

        return times, nb_time_step, nwp_x_l93, nwp_y_l93, nb_px_nwp_y, nb_px_nwp_x

    @print_func_executed_decorator("get MNT characteristics", level_begin="\n__", level_end="__")
    @timer_decorator("get MNT characteristics", unit="second", level=". . ")
    def get_characteristics_mnt(self, mnt_data, verbose=True):

        xmin_mnt = np.nanmin(mnt_data.x.data)
        ymax_mnt = np.nanmax(mnt_data.y.data)
        resolution_x = self.mnt.resolution_x
        resolution_y = self.mnt.resolution_y

        return xmin_mnt, ymax_mnt, resolution_x, resolution_y

    @staticmethod
    @print_func_executed_decorator("extracting variables from xarray", level_begin="\n__", level_end="__")
    @timer_decorator("extracting variables from xarray", unit="second", level=". . ")
    def extract_from_xarray_to_numpy(array, list_variables, verbose=True):
        return (array[variable].data.astype(np.float32) for variable in list_variables)

    @print_func_executed_decorator("Look for stations in the domain", level_begin="\n__", level_end="__")
    @timer_decorator("Look for stations in the domain", unit="second", level=". . ")
    def find_stations_in_domain(self, mnt_data_x=None, mnt_data_y=None, verbose=True):

        # DataFrame with stations
        stations = self.observation.stations

        # Filters
        filter_x_min = mnt_data_x.min() < stations['X']
        filter_x_max = stations['X'] < mnt_data_x.max()
        filter_y_min = mnt_data_y.min() < stations['Y']
        filter_y_max = stations['Y'] < mnt_data_y.max()

        # Stations in the domain
        stations = stations[filter_x_min & filter_x_max & filter_y_min & filter_y_max]

        if verbose:
            print(f"__Stations found in the domain: \n {stations['name'].values}")
        return stations['X'], stations['Y'], stations['name']
