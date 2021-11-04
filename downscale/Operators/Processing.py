# Packages: numpy, pandas, xarray, scipy, tensorflow, matplotlib
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.colors as colors
import random
import datetime
from time import time as t

# Local imports
from downscale.Utils.GPU import environment_GPU
from downscale.Utils.Utils import assert_equal_shapes, reshape_list_array, print_statistical_description_array
from downscale.Utils.Decorators import print_func_executed_decorator, timer_decorator
from downscale.Operators.Rotation import Rotation
from downscale.Operators.wind_utils import Wind_utils
from downscale.Operators.topo_utils import Topo_utils
from downscale.Operators.Helbig import DwnscHelbig
from downscale.Operators.Micro_Met import MicroMet

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

try:
    from shapely.geometry import Point

    _shapely_geometry = True
except ModuleNotFoundError:
    _shapely_geometry = False

try:
    import geopandas as gpd

    _geopandas = True
except ModuleNotFoundError:
    _geopandas = False

try:
    import dask

    _dask = True
except ModuleNotFoundError:
    _dask = False


# Custom Metrics : NRMSE
def nrmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred))) / (K.max(y_pred) - K.min(y_pred))


# Custom Metrics : RMSE
def root_mse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


# noinspection PyAttributeOutsideInit,PyUnboundLocalVariable
class Processing(Wind_utils, DwnscHelbig, MicroMet, Rotation):
    n_rows, n_col = 79, 69
    _geopandas = _geopandas
    _shapely_geometry = _shapely_geometry

    def __init__(self, obs=None, mnt=None, nwp=None, prm=None, GPU=False):

        super().__init__()

        self.observation = obs if obs is not None else None
        self.mnt = mnt if mnt is not None else None
        self.nwp = nwp if nwp is not None else None
        self.model_path = prm["model_path"] if (prm is not None and prm["model_path"] is not None) else None
        self.data_path = prm['data_path'] if (prm is not None and prm['data_path'] is not None) else None
        self.is_updated_with_topo_characteristics = False
        environment_GPU(GPU=prm["GPU"]) if (prm is not None and prm["GPU"] is not None) else None

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

    @staticmethod
    def _select_time_serie_from_array_xr(array_xr, station_name='Col du Lac Blanc', variable='UV', center=True):
        """
        Extract a time serie from array_xr and returns a numpy array.

        Used to extract values at the center of the CNN domain.

        Parameters
        ----------
        array_xr : xarray dataframe
            Contains the predictions
        station_name : string, optional
            The considered station (Default: 'Col du Lac Blanc')
        variable : string, optional
            The variable to extract (Default: 'UV')

        Returns
        -------
        prediction : 1-D ndarray
            Considered variable.
        time : 1-D ndarray
            Time of predictions.
        """
        if center:
            x = 34
            y = 39
        prediction = array_xr[variable].sel(station=station_name).isel(x=x, y=y).data
        time = array_xr.sel(station=station_name).isel(x=x, y=y).time.data
        return prediction, time

    @print_func_executed_decorator("load model", level_begin="\n__", level_end="__")
    @timer_decorator("load model", unit="second", level=". . . . ")
    def load_cnn(self, dependencies=False, verbose=True):
        """
        Load a CNN, and its dependencies.

        Parameters
        ----------
        dependencies : boolean, optional
            CNN dependencies (Default: False')

        Returns
        -------
        model : Tensorflow model
            Return the CNN, stored in "self.model"
        """

        # todo load dependencies
        if dependencies:

            def root_mse(y_true, y_pred):
                return K.sqrt(K.mean(K.square(y_true - y_pred)))

            dependencies = {'root_mse': root_mse}
            #model = load_model(self.model_path+"model_weights.h5", custom_objects=dependencies)
            model = load_model(self.model_path, custom_objects=dependencies)

            print("____Dependencies: True") if verbose else None
        else:
            #model = load_model(self.model_path+"model_weights.h5")
            print("____Dependencies: False") if verbose else None

        self.model = model

    def _load_norm_prm(self):
        """
        Load normalization parameters: mean and std.

        Returns
        -------
        mean : numpy array
        std : numpy array
        """

        # todo load dependencies
        dict_norm = pd.read_csv(self.model_path + "dict_norm.csv")
        mean = dict_norm["0"].iloc[0]
        std = dict_norm["0"].iloc[1]
        return mean, std

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

    @timer_decorator("initialize arrays", unit="second", level=". . . . ")
    def _initialize_arrays(self, predict='stations_month', nb_station=None, nb_sim=None):
        """
        Utility function used to initialize arrays in _predict_at_stations
        """
        if predict == 'stations_month':
            topo = np.empty((nb_station, nb_sim, self.n_rows, self.n_col, 1), dtype=np.float32)
            wind_speed_all = np.empty((nb_station, nb_sim), dtype=np.float32)
            wind_dir_all = np.empty((nb_station, nb_sim), dtype=np.float32)
            Z0_all = np.empty((nb_station, nb_sim), dtype=np.float32)
            Z0REL_all = np.empty((nb_station, nb_sim), dtype=np.float32)
            ZS_all = np.empty((nb_station, nb_sim), dtype=np.uint16)
            peak_valley_height = np.empty(nb_station, dtype=np.float32)
            mean_height = np.empty(nb_station, dtype=np.float32)
            max_height = np.empty(nb_station, dtype=np.float32)
            all_topo_HD = np.empty((nb_station, self.n_rows, self.n_col), dtype=np.uint16)
            all_topo_x_small_l93 = np.empty((nb_station, self.n_col), dtype=np.float32)
            all_topo_y_small_l93 = np.empty((nb_station, self.n_rows), dtype=np.float32)
            ten_m_array = 10 * np.ones((nb_station, nb_sim), dtype=np.float32)
            three_m_array = 3 * np.ones((nb_station, nb_sim), dtype=np.float32)
            list_arrays_1 = [topo, wind_speed_all, wind_dir_all, Z0_all, Z0REL_all, ZS_all, peak_valley_height,
                             mean_height]
            list_arrays_2 = [all_topo_HD, all_topo_x_small_l93, all_topo_y_small_l93, ten_m_array, three_m_array,
                             max_height]
            list_return = list_arrays_1 + list_arrays_2
        return list_return

    @timer_decorator("modify wind speed observations", unit="second", level=". . ")
    def modify_wind_speed_observation(self, prm, z_out_height=10, wind_speed='vw10m(m/s)',
                                      snow_height_str="HTN(cm)", Z0_snow=0.001, Z0_bare_ground=0.05):

        stations = self.observation.stations
        time_series = self.observation.time_series
        height_sensor = self.read_height_sensor(prm["height_sensor_path"])
        time_series["wind_corrected"] = np.nan

        for station in stations["name"].values:

            filter_station = height_sensor["name"] == station
            sensor_height = height_sensor["height"][filter_station].values[0]

            # Select observations at the station
            filter_station = time_series["name"] == station
            obs_station = time_series[filter_station]

            # Load wind speed from observations
            UV = obs_station[wind_speed].values

            Z0 = np.full_like(UV, Z0_bare_ground)

            z_in = np.full_like(UV, sensor_height)
            z_out = np.full_like(UV, z_out_height)

            if station in prm["list_no_HTN"]:

                if sensor_height != 10:

                    # Compute logarithmic adjustment
                    z_in_verbose = str(np.round(np.mean(z_in)))
                    z_out_verbose = str(z_out_height)
                    wind_corrected = self.apply_log_profile(z_in, z_out, UV, Z0, z_in_verbose=z_in_verbose,
                                                            z_out_verbose=z_out_verbose)

                else:

                    wind_corrected = UV

            else:

                snow_height = obs_station[snow_height_str].values / 100
                Z0 = np.where(snow_height > 0.02, Z0_snow, Z0)

                # Compute logarithmic adjustment
                z_in = z_in - snow_height
                z_in_verbose = "multiple heights depending on snow height"
                z_out_verbose = str(z_out_height)
                wind_corrected = self.apply_log_profile(z_in, z_out, UV, Z0, z_in_verbose=z_in_verbose,
                                                        z_out_verbose=z_out_verbose)

            filter_time = time_series.index.isin(obs_station.index)
            time_series["wind_corrected"][filter_station & filter_time] = wind_corrected

        self.observation.time_series = time_series

    @staticmethod
    def read_height_sensor(path):
        try:
            return pd.read_pickle(path)
        except:
            return pd.read_csv(path)

    def predict_at_stations(self, stations_name, line_profile=None, prm=None):
        """
        This function is used to select predictions at stations, the line profiled version or the memory profiled version
        """

        if line_profile:
            from line_profiler import LineProfiler
            lp = LineProfiler()
            lp_wrapper = lp(self._predict_at_stations)
            lp_wrapper(stations_name, **prm)
            lp.print_stats()
        else:
            array_xr = self._predict_at_stations(stations_name, **prm)
            return array_xr

    @staticmethod
    def select_height_for_exposed_wind_speed(height=None, zs=None, peak_valley=None):
        if peak_valley:
            return height
        else:
            return zs

    @staticmethod
    @print_func_executed_decorator("get closer from learning conditions", level_begin="\n__", level_end="__")
    @timer_decorator("get closer from learning conditions", unit="second", level="..")
    def get_closer_from_learning_conditions(topo, mean_height, std, P95=530, P05=-527, axis=(1, 2)):

        max_alt_deviation = np.nanmax(topo.squeeze() - mean_height, axis=axis)
        min_alt_deviation = np.nanmin(topo.squeeze() - mean_height, axis=axis)

        alpha_max = np.abs(max_alt_deviation / P95)
        alpha_min = np.abs(min_alt_deviation / P05)
        alpha_i = np.nanmax((alpha_max, alpha_min))

        alpha = np.where(max_alt_deviation > P95,
                         np.where(min_alt_deviation < P05, alpha_i, alpha_max),
                         np.where(min_alt_deviation < P05, alpha_min, 1))
        print("____Quantile 0.5 alpha", np.quantile(alpha, 0.5))
        print("____Quantile 0.8 alpha", np.quantile(alpha, 0.8))
        print("____Quantile 0.9 alpha", np.quantile(alpha, 0.9))
        print("____Max alpha", np.nanmax(alpha))

        return alpha * std

    def _predict_at_stations(self, stations_name, **kwargs):
        """
        Wind downscaling operated at observation stations sites only.

        17 min on CPU
        3.7 minutes on GPU

        Parameters
        ----------
        stations_name : list of strings
            List containing station names
        Z0 : boolean
            To expose wind speed
        peak_valley : boolean
            Use mean peak valley height to expose wind. An other option is to use mean height.
        log_profile_* : boolean(s)
            If True, apply a log profile inside the function to adapt to calculation heights.
        ideal_case: boolean
            If True, run an ideal case during one day where the input speed and direction are specified by the user
        input_speed: float
            Input wind speed specified by the user for ideal cases (Default: 3 [m/s])
        input_dir: float
            Input wind direction specified by the user for ideal cases (Default: 270 [°], wind coming from the West)

        Returns
        -------
        array_xr : xarray DataFrame
            Result dataframe containing wind components, speeds, wind directions, accelerations and input data


        Exemple
        -------
        array_xr = p._predict_at_stations(['Col du Lac Blanc',
                             verbose=True,
                             Z0=True,
                             peak_valley=True,
                             ideal_case=False)
        """

        verbose = kwargs.get("verbose")
        Z0 = kwargs.get("Z0")
        peak_valley = kwargs.get("peak_valley")
        log_profile_to_h_2 = kwargs.get("log_profile_to_h_2")
        log_profile_from_h_2 = kwargs.get("log_profile_from_h_2")
        log_profile_10m_to_3m = kwargs.get("log_profile_10m_to_3m")
        log_profile_3m_to_10m = kwargs.get("log_profile_3m_to_10m")
        ideal_case = kwargs.get("ideal_case")
        input_speed = kwargs.get("input_speed")
        input_dir = kwargs.get("input_dir")
        interp_str = kwargs.get("interp_str")
        centered_on_interpolated = kwargs.get("centered_on_interpolated")
        scaling_function = kwargs.get("scaling_function")
        scale_at_10m = kwargs.get("scale_at_10m")
        scale_at_max_altitude = kwargs.get("scale_at_max_altitude")
        get_closer_learning_condition = kwargs.get("get_closer_learning_condition")

        # Select timeframe
        self._select_timeframe_nwp(ideal_case=ideal_case, verbose=True)

        # Simulation parameters
        time_xr = self._extract_variable_from_nwp_at_station(random.choice(stations_name), variable_to_extract=["time"],
                                                             interp_str=interp_str)
        nb_sim = len(time_xr)
        nb_station = len(stations_name)

        # initialize arrays
        topo, wind_speed_all, wind_dir_all, Z0_all, Z0REL_all, ZS_all, \
        peak_valley_height, mean_height, all_topo_HD, all_topo_x_small_l93, all_topo_y_small_l93, ten_m_array, \
        three_m_array, max_height = self._initialize_arrays(predict='stations_month', nb_station=nb_station,
                                                            nb_sim=nb_sim)

        # Indexes
        nb_pixel = 70  # min = 116/2
        y_offset_left = nb_pixel - 39
        y_offset_right = nb_pixel + 40
        x_offset_left = nb_pixel - 34
        x_offset_right = nb_pixel + 35

        for idx_station, single_station in enumerate(stations_name):

            print(f"\nBegin downscaling at {single_station}")

            # Select nwp pixel
            if Z0:
                wind_dir_all[idx_station, :], wind_speed_all[idx_station, :], Z0_all[idx_station, :], \
                Z0REL_all[idx_station, :], ZS_all[idx_station, :] = self._extract_variable_from_nwp_at_station(
                    single_station,
                    variable_to_extract=["wind_direction", "wind_speed", "Z0", "Z0REL", "ZS"],
                    interp_str=interp_str)
            else:
                wind_dir_all[idx_station, :], wind_speed_all[idx_station, :], \
                ZS_all[idx_station, :] = self._extract_variable_from_nwp_at_station(
                    single_station,
                    variable_to_extract=["wind_direction", "wind_speed", "ZS"],
                    interp_str=interp_str)

            # For ideal case, we define the input speed and direction
            if ideal_case:
                wind_speed_all[idx_station, :], \
                wind_dir_all[idx_station, :] = self._scale_wind_for_ideal_case(wind_speed_all[idx_station, :],
                                                                               wind_dir_all[idx_station, :],
                                                                               input_speed,
                                                                               input_dir)

            # Extract topography
            extract_around = "station" if not centered_on_interpolated else "nwp_neighbor_interp"
            nwp = None if not centered_on_interpolated else self.nwp
            topo_HD, topo_x_l93, topo_y_l93 = self.observation.extract_MNT(self.mnt,
                                                                           nb_pixel,
                                                                           nb_pixel,
                                                                           station=single_station,
                                                                           nwp=nwp,
                                                                           extract_around=extract_around)

            # Rotate topographies
            topo[idx_station, :, :, :, 0] = self.select_rotation(data=topo_HD,
                                                                 wind_dir=wind_dir_all[idx_station, :],
                                                                 clockwise=False)[:, y_offset_left:y_offset_right,
                                            x_offset_left:x_offset_right]

            # Store results
            all_topo_HD[idx_station, :, :] = topo_HD[y_offset_left:y_offset_right, x_offset_left:x_offset_right]
            all_topo_x_small_l93[idx_station, :] = topo_x_l93[x_offset_left:x_offset_right]
            all_topo_y_small_l93[idx_station, :] = topo_y_l93[y_offset_left:y_offset_right]
            peak_valley_height[idx_station] = np.int32(2 * np.nanstd(all_topo_HD[idx_station, :, :]))
            mean_height[idx_station] = np.int32(np.nanmean(all_topo_HD[idx_station, :, :]))
            max_height[idx_station] = np.int32(np.nanmax(all_topo_HD[idx_station, :, :]))

        # Exposed wind
        if Z0:
            peak_valley_height = peak_valley_height.reshape((nb_station, 1))
            max_height = max_height.reshape((nb_station, 1))
            Z0_all = np.where(Z0_all == 0, 1 * 10 ^ (-8), Z0_all)
            if scale_at_10m:
                zs = ten_m_array
            elif scale_at_max_altitude:
                zs = max_height
            else:
                zs = ZS_all
            height = self.select_height_for_exposed_wind_speed(height=peak_valley_height,
                                                               zs=zs,
                                                               peak_valley=peak_valley)
            del zs
            del max_height
            wind1 = np.copy(wind_speed_all)

            # Log profile
            if log_profile_to_h_2:
                wind_speed_all = self.apply_log_profile(z_in=ten_m_array, z_out=height / 2, wind_in=wind_speed_all,
                                                        z0=Z0_all,
                                                        verbose=verbose, z_in_verbose="10m", z_out_verbose="height/2")
            a1 = self.wind_speed_ratio(num=wind_speed_all, den=wind1)
            wind2 = np.copy(wind_speed_all)

            # Expose wind
            if scale_at_10m or scale_at_max_altitude:
                z_out = height
            else:
                z_out = height / 2
            exp_Wind, acceleration_factor = self.exposed_wind_speed(wind_speed=wind_speed_all,
                                                                    z_out=z_out,
                                                                    z0=Z0_all,
                                                                    z0_rel=Z0REL_all)
            # exp_Wind = wind_speed_all
            # acceleration_factor = exp_Wind*0+1

            a2 = self.wind_speed_ratio(num=exp_Wind, den=wind2)
            del wind2
            wind3 = np.copy(exp_Wind)

            # Log profile
            if log_profile_from_h_2:
                exp_Wind = self.apply_log_profile(z_in=height / 2, z_out=three_m_array, wind_in=exp_Wind, z0=Z0_all,
                                                  verbose=verbose, z_in_verbose="height/2", z_out_verbose="3m")
            a3 = self.wind_speed_ratio(num=exp_Wind, den=wind3)
            del wind3

        # Normalize
        _, std = self._load_norm_prm()
        mean_height = mean_height.reshape((nb_station, 1, 1, 1))
        if get_closer_learning_condition:
            std = self.get_closer_from_learning_conditions(topo, mean_height, std, axis=(2, 3))
            std = std.reshape((nb_station, nb_sim, 1, 1, 1))
        else:
            std = std.reshape((1, 1, 1, 1, 1))

        mean_height = mean_height.reshape((nb_station, 1, 1, 1, 1))
        topo = self.normalize_topo(topo, mean_height, std)

        # Reshape for tensorflow
        topo = topo.reshape((nb_station * nb_sim, self.n_rows, self.n_col, 1))

        """
        Warning: change dependencies here
        """
        # Load model
        self.load_cnn(dependencies=True)

        # Predictions
        prediction = self.cnn_prediction(topo)

        # Acceleration NWP to CNN
        UVW_int = self.compute_wind_speed(U=prediction[:, :, :, 0], V=prediction[:, :, :, 1], W=prediction[:, :, :, 2])
        acceleration_CNN = self.wind_speed_ratio(num=UVW_int, den=3 * np.ones(UVW_int.shape))

        # Reshape predictions for analysis
        prediction = prediction.reshape((nb_station, nb_sim, self.n_rows, self.n_col, 3))

        # Reshape for broadcasting
        wind_speed_all = wind_speed_all.reshape((nb_station, nb_sim, 1, 1, 1))
        if Z0:
            exp_Wind, acceleration_factor, ten_m_array, three_m_array, Z0_all = reshape_list_array(
                list_array=[exp_Wind, acceleration_factor, ten_m_array, three_m_array, Z0_all],
                shape=(nb_station, nb_sim, 1, 1, 1))
            peak_valley_height = peak_valley_height.reshape((nb_station, 1, 1, 1, 1))
        wind_dir_all = wind_dir_all.reshape((nb_station, nb_sim, 1, 1))

        # Wind speed scaling
        scaling_wind = exp_Wind if Z0 else wind_speed_all
        prediction = self.wind_speed_scaling(scaling_wind, prediction, type_scaling=scaling_function)

        # Copy wind variable
        wind4 = np.copy(prediction)

        if log_profile_3m_to_10m:
            # Apply log profile: 3m => 10m
            prediction = self.apply_log_profile(z_in=ten_m_array, z_out=three_m_array, wind_in=prediction, z0=Z0_all,
                                                verbose=verbose, z_in_verbose="3m", z_out_verbose="10m")
        # Acceleration a4
        a4 = self.wind_speed_ratio(num=prediction, den=wind4)
        del wind4

        # Wind computations
        U_old = prediction[:, :, :, :, 0].view()  # Expressed in the rotated coord. system [m/s]
        V_old = prediction[:, :, :, :, 1].view()  # Expressed in the rotated coord. system [m/s]
        W_old = prediction[:, :, :, :, 2].view()  # Good coord. but not on the right pixel [m/s]

        # Recalculate with respect to original coordinates
        UV = self.compute_wind_speed(U=U_old, V=V_old, W=None)  # Good coord. but not on the right pixel [m/s]
        alpha = self.angular_deviation(U_old, V_old,
                                       unit_output="radian")  # Expressed in the rotated coord. system [radian]
        UV_DIR = self.direction_from_alpha(wind_dir_all, alpha,
                                           unit_direction="degree",
                                           unit_alpha="radian",
                                           unit_output="radian")  # Good coord. but not on the right pixel [radian]

        # Verification of shapes
        assert_equal_shapes([U_old, V_old, W_old, UV, alpha, UV_DIR], (nb_station, nb_sim, self.n_rows, self.n_col))

        # Calculate U and V along initial axis
        # Good coord. but not on the right pixel [m/s]
        prediction[:, :, :, :, 0], prediction[:, :, :, :, 1] = self.horizontal_wind_component(UV=UV,
                                                                                              UV_DIR=UV_DIR,
                                                                                              unit_direction="radian",
                                                                                              verbose=True)
        del UV_DIR

        # Rotate clockwise to put the wind value on the right topography pixel
        prediction = np.moveaxis(prediction, -1, 2)
        wind_dir_all = wind_dir_all.reshape((nb_station, nb_sim, 1))
        prediction = self.select_rotation(data=prediction[:, :, :, :, :],
                                          wind_dir=wind_dir_all[:, :, :],
                                          clockwise=True,
                                          verbose=False)
        prediction = np.moveaxis(prediction, 2, -1)

        U = prediction[:, :, :, :, 0].view()
        V = prediction[:, :, :, :, 1].view()
        W = prediction[:, :, :, :, 2].view()

        # Compute wind direction
        UV_DIR = self.direction_from_u_and_v(U, V, unit_output="degree")  # Good axis and pixel [degree]

        # UVW
        UVW = self.compute_wind_speed(U=U, V=V, W=W)

        # Acceleration NWP to CNN
        acceleration_all = self.wind_speed_ratio(num=UVW,
                                                 den=wind1.reshape((nb_station, nb_sim, 1, 1))) if Z0 else np.full_like(
            UVW, np.nan)

        # Reshape after broadcasting
        wind_speed_all, wind_dir_all, Z0_all = reshape_list_array(list_array=[wind_speed_all, wind_dir_all, Z0_all],
                                                                  shape=(nb_station, nb_sim))
        if Z0:
            exp_Wind, acceleration_factor, a1, a2, a3 = reshape_list_array(
                list_array=[exp_Wind, acceleration_factor, a1, a2, a3],
                shape=(nb_station, nb_sim))
            a4, acceleration_CNN = reshape_list_array(list_array=[np.max(a4, axis=4), acceleration_CNN],
                                                      shape=(nb_station, nb_sim, self.n_rows, self.n_col))
            peak_valley_height = peak_valley_height.reshape(nb_station)

        # Verification of shapes
        assert_equal_shapes([U, V, W, UV_DIR], (nb_station, nb_sim, self.n_rows, self.n_col))
        assert_equal_shapes([wind_speed_all, wind_dir_all], (nb_station, nb_sim))

        # Store results
        print('__Start creating array') if verbose else None
        array_xr = xr.Dataset(data_vars={"U": (["station", "time", "y", "x"], U),
                                         "V": (["station", "time", "y", "x"], V),
                                         "W": (["station", "time", "y", "x"], W),
                                         "UV": (["station", "time", "y", "x"], self.compute_wind_speed(U=U, V=V)),
                                         "UVW": (["station", "time", "y", "x"], self.compute_wind_speed(U=U, V=V, W=W)),
                                         "UV_DIR_deg": (["station", "time", "y", "x"], UV_DIR),
                                         "alpha_deg": (["station", "time", "y", "x"],
                                                       wind_dir_all.reshape((nb_station, nb_sim, 1, 1)) - UV_DIR),
                                         "NWP_wind_speed": (["station", "time"], wind_speed_all),
                                         "NWP_wind_DIR": (["station", "time"], wind_dir_all),
                                         "ZS_mnt": (["station", "y", "x"], all_topo_HD,),
                                         "peak_valley_height": (["station"], peak_valley_height),
                                         "XX": (["station", "x"], all_topo_x_small_l93,),
                                         "YY": (["station", "y"], all_topo_y_small_l93,),
                                         "exp_Wind": (
                                             ["station", "time"],
                                             exp_Wind if Z0 else np.zeros((nb_station, nb_sim)),),
                                         "acceleration_factor": (
                                             ["station", "time"],
                                             acceleration_factor if Z0 else np.zeros((nb_station, nb_sim)),),
                                         "a1": (
                                             ["station", "time"],
                                             a1 if Z0 else np.zeros((nb_station, nb_sim)),),
                                         "a2": (
                                             ["station", "time"],
                                             a2 if Z0 else np.zeros((nb_station, nb_sim)),),
                                         "a3": (
                                             ["station", "time"],
                                             a3 if Z0 else np.zeros((nb_station, nb_sim)),),
                                         "a4": (
                                             ["station", "time", "y", "x"],
                                             a4 if Z0 else np.zeros(
                                                 (nb_station, nb_sim, self.n_rows, self.n_col)),),
                                         "acceleration_all": (
                                             ["station", "time", "y", "x"],
                                             acceleration_all if Z0 else np.zeros(
                                                 (nb_station, nb_sim, self.n_rows, self.n_col)),),
                                         "acceleration_CNN": (
                                             ["station", "time", "y", "x"],
                                             acceleration_CNN if Z0 else np.zeros(
                                                 (nb_station, nb_sim, self.n_rows, self.n_col)),),
                                         "Z0": (
                                             ["station", "time"],
                                             Z0_all if Z0 else np.zeros((nb_station, nb_sim)),),
                                         "Z0REL": (["station", "time"],
                                                   Z0REL_all if Z0 else np.zeros((nb_station, nb_sim)),),
                                         },

                              coords={"station": np.array(stations_name),
                                      "time": np.array(time_xr),
                                      "x": np.array(list(range(self.n_col))),
                                      "y": np.array(list(range(self.n_rows)))})
        print('__Creating array done') if verbose else None

        return array_xr

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

    @staticmethod
    def _iterate_to_replace_pixels(wind, mnt_map, idx_y_mnt, idx_x_mnt, save_pixels, y_offset_left, y_offset_right,
                                   x_offset_left, x_offset_right):

        for j in prange(wind.shape[1]):
            for i in prange(wind.shape[2]):
                mnt_y_left = idx_y_mnt[j, i] - save_pixels
                mnt_y_right = idx_y_mnt[j, i] + save_pixels + 1
                mnt_x_left = idx_x_mnt[j, i] - save_pixels
                mnt_x_right = idx_x_mnt[j, i] + save_pixels + 1
                for time in prange(wind.shape[0]):
                    for dimension in prange(wind.shape[5]):
                        mnt_map[time, mnt_y_left:mnt_y_right, mnt_x_left:mnt_x_right, dimension] = wind[time, j, i,
                                                                                        y_offset_left:y_offset_right,
                                                                                        x_offset_left:x_offset_right, dimension]
        return mnt_map

    def _replace_pixels_on_map(self, mnt_map=None, wind=None, idx_x_mnt=None, idx_y_mnt=None, library='num',
                               y_offset_left=None, y_offset_right=None, x_offset_left=None, x_offset_right=None,
                               save_pixels=None, verbose=True):

        if library == 'numba' and _numba:
            jit_rpl_px = jit([float32[:, :, :, :](float32[:, :, :, :, :, :], float32[:, :, :, :], int64[:, :],
                                                  int64[:, :], int64, int64, int64, int64, int64)],
                             nopython=True, parallel=True)(self._iterate_to_replace_pixels)
            result = jit_rpl_px(wind, mnt_map, idx_y_mnt, idx_x_mnt, save_pixels, y_offset_left, y_offset_right,
                                x_offset_left, x_offset_right)
            print("____Library: Numba") if verbose else None

        else:
            result = self._iterate_to_replace_pixels(wind, mnt_map, idx_y_mnt, idx_x_mnt, save_pixels, y_offset_left,
                                                     y_offset_right, x_offset_left, x_offset_right)
            print("____Library: Numpy") if verbose else None

        return result

    @print_func_executed_decorator("replace pixels on maps", level_begin="\n__", level_end="__")
    @timer_decorator("replace pixels on maps", unit="second", level=". . ")
    def replace_pixels_on_map(self, mnt_map=None, wind=None, idx_x_mnt=None, idx_y_mnt=None, type_rotation=None,
                              wind_speed_nwp=None, save_pixels=15, acceleration=False,
                              library='numba', verbose=True):

        x_center, y_center = self._select_center_to_replace_pixels(type_rotation)

        y_offset_left = y_center - save_pixels
        y_offset_right = y_center + save_pixels + 1
        x_offset_left = x_center - save_pixels
        x_offset_right = x_center + save_pixels + 1

        mnt_map = self._replace_pixels_on_map(mnt_map=mnt_map, wind=wind, idx_x_mnt=idx_x_mnt, idx_y_mnt=idx_y_mnt,
                                              library=library, save_pixels=save_pixels,
                                              y_offset_left=y_offset_left, y_offset_right=y_offset_right,
                                              x_offset_left=x_offset_left, x_offset_right=x_offset_right)
        if acceleration:
            for j in range(wind.shape[1]):
                for i in range(wind.shape[2]):
                    mnt_y_left = idx_y_mnt[j, i] - save_pixels
                    mnt_y_right = idx_y_mnt[j, i] + save_pixels + 1
                    mnt_x_left = idx_x_mnt[j, i] - save_pixels
                    mnt_x_right = idx_x_mnt[j, i] + save_pixels + 1
                    acceleration_all = np.empty(mnt_map.shape[:-1])
                    UV = self.compute_wind_speed(
                        wind[:, j, i, y_offset_left:y_offset_right, x_offset_left:x_offset_right, 0],
                        wind[:, j, i, y_offset_left:y_offset_right, x_offset_left:x_offset_right, 1],
                        verbose=False)

                    acceleration_all[:, mnt_y_left:mnt_y_right, mnt_x_left:mnt_x_right] = UV / wind_speed_nwp[:, j,
                                                                                               i].reshape(
                        (wind_speed_nwp.shape[0], 1, 1))
            return mnt_map, acceleration_all
        else:
            return mnt_map, np.array([])

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

    @print_func_executed_decorator("Horizontal stacking of topographies", level_begin="\n__", level_end="__")
    @timer_decorator("Horizontal stacking of topographies", unit="second", level=". . ")
    def hstack_topo(self, topo_i, idx_x_mnt, idx_y_mnt, mnt_data, nb_pixel, library='numba', verbose=True):

        if library == 'numba' and _numba:
            jit_hstack = jit(
                [float32[:, :, :, :](float32[:, :, :, :], int64[:, :], int64[:, :], float32[:, :, :], int64)],
                nopython=True)(self._hstack_topo)
            result = jit_hstack(topo_i, idx_x_mnt, idx_y_mnt, mnt_data, nb_pixel)
            print("____Library: Numba") if verbose else None

        else:
            result = self._hstack_topo(topo_i, idx_x_mnt, idx_y_mnt, mnt_data, nb_pixel)
            print("____Library: Numba") if verbose else None

        return result

    @staticmethod
    def _hstack_topo(topo_i, idx_x_mnt, idx_y_mnt, mnt_data, nb_pixel):
        for j in range(topo_i.shape[0]):
            for i in range(topo_i.shape[1]):
                y = idx_y_mnt[j, i]
                x = idx_x_mnt[j, i]
                topo_i[j, i, :, :] = mnt_data[0, y - nb_pixel:y + nb_pixel, x - nb_pixel:x + nb_pixel]
        return topo_i

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

    @print_func_executed_decorator("Extract indexes of downscaled wind at stations", level_begin="\n__", level_end="__")
    @timer_decorator("Extract indexes of downscaled wind at stations", unit="second", level=". . ")
    def extract_downscaled_wind_at_stations(self,
                                            mnt_data_x=None, mnt_data_y=None, xmin_mnt=None, ymax_mnt=None,
                                            resolution_x=None, resolution_y=None, look_for_resolution=False,
                                            look_for_corners=False, verbose=True):

        x_stations, y_stations, stations = self.find_stations_in_domain(mnt_data_x=mnt_data_x, mnt_data_y=mnt_data_y)
        idx_x_stations, idx_y_stations = self.mnt.find_nearest_MNT_index(x_stations,
                                                                         y_stations,
                                                                         look_for_corners=look_for_corners,
                                                                         xmin_MNT=xmin_mnt,
                                                                         ymax_MNT=ymax_mnt,
                                                                         look_for_resolution=look_for_resolution,
                                                                         resolution_x=resolution_x,
                                                                         resolution_y=resolution_y)

        return idx_x_stations, idx_y_stations, stations.values

    @print_func_executed_decorator("cnn prediction", level_begin="\n__", level_end="__")
    @timer_decorator("cnn prediction", unit="minute", level="..")
    def cnn_prediction(self, input_topo, verbose=True):
        #print(f"____Input shape before CNN prediction: {input_topo.shape}") if verbose else None
        return self.model.predict(input_topo)

    @print_func_executed_decorator("cnn prediction map", level_begin="\n__", level_end="__")
    @timer_decorator("cnn prediction map", unit="second", level="..")
    def cnn_prediction_map(self, input_, GPU=False):

        device = '/gpu:0' if GPU else '/cpu:0'

        prediction = []

        topo_generator = self.create_generator(input_)
        for batch_1 in topo_generator:

            callable_generator = self.create_callable_generator(batch_1)
            batch_1 = tf.data.Dataset.from_generator(callable_generator, tf.float32).batch(2**10)

            with tf.device(device):
                batch_1 = self.model.predict(batch_1)

            with tf.device('/cpu:0'):
                prediction.append(batch_1)
                del batch_1
                del callable_generator

        print("____Using tf.data.Dataset.from_generator")
        prediction = np.concatenate(prediction).astype(np.float32)
        print(f"____shape prediction: {np.shape(prediction)}")

        return prediction

    @staticmethod
    def create_generator(array):
        nb_elements = array.shape[0]
        list_inputs = np.linspace(0, nb_elements, 10)
        for i, input_ in enumerate(list_inputs):
            if input_ != list_inputs[-1]:
                begin = np.intp(list_inputs[i])
                end = np.intp(list_inputs[i+1])
                yield array[begin:end, :]

    @staticmethod
    def split_array(array):
        nb_elements = array.shape[0]
        return array[:nb_elements//2, :], array[nb_elements//2:, :]

    @staticmethod
    def create_callable_generator(array):
        def create_generator_2():
            for i, _ in enumerate(array):
                yield array[i, :]
        return create_generator_2

    def select_z_expose(self, topo, expose, shape=None):

        if expose == "peak valley":
            z_expose = self.mean_peak_valley(topo) / 2
        elif expose == "10m":
            z_expose = np.full(shape, 10)
        elif expose == "max altitude":
            z_expose = np.int32(np.nanmax(topo, axis=(2, 3)))
        else:
            z_expose = None

        assert z_expose.shape == shape

        return z_expose

    def _rotation_wrapper_for_topo_in_map_predictions(self, topo, angle, type_rotation, GPU=None,
                                                      nb_time_step=None, nb_px_nwp_y=None, nb_px_nwp_x=None):

        if type_rotation == 'indexes':
            topo_rot = np.empty((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79 * 69)).astype(np.float32)
            all_mat = np.load(self.data_path + "MNT/indexes_rot.npy").astype(np.int32).reshape(360, 79 * 69, 2)
            # all_mat=(360, 5451, 2), topo_rot=(1, 21, 21, 5451), topo_i=(21, 21, 140, 140), angle=(1, 21, 21)
            topo_rot = self.select_rotation(all_mat=all_mat, topo_rot=topo_rot, topo_i=topo, angles=angle,
                                            type_rotation='topo_indexes', library='numba')
        elif type_rotation == 'scipy':
            nb_pixel = 70  # min = 116/2
            y_left = nb_pixel - 39
            y_right = nb_pixel + 40
            x_left = nb_pixel - 34
            x_right = nb_pixel + 35
            topo = topo.reshape((1, nb_px_nwp_y, nb_px_nwp_x, 140, 140))
            angle = angle.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x))
            topo_rot = self.select_rotation(data=topo,
                                            wind_dir=angle,
                                            type_rotation=type_rotation,
                                            clockwise=False,
                                            GPU=GPU).squeeze()[:, :, :, y_left:y_right, x_left:x_right]
        elif type_rotation == "tfa":
            nb_pixel = 70  # min = 116/2
            y_left = nb_pixel - 39
            y_right = nb_pixel + 40
            x_left = nb_pixel - 34
            x_right = nb_pixel + 35
            topo = topo.reshape((1, nb_px_nwp_y * nb_px_nwp_x, 140, 140, 1))
            angle = angle.reshape((nb_time_step, nb_px_nwp_y * nb_px_nwp_x))
            device = '/gpu:0' if GPU else '/cpu:0'
            with tf.device(device):
                topo_rot = self.select_rotation(data=topo,
                                                wind_dir=angle,
                                                type_rotation=type_rotation,
                                                clockwise=False,
                                                GPU=GPU).squeeze()[:, :, y_left:y_right, x_left:x_right]
        return topo_rot

    def _rotation_wrapper_for_wind_in_predict_maps(self, wind, angle, type_rotation,
                                                   GPU=None, nb_time_step=None,
                                                   nb_px_nwp_y=None, nb_px_nwp_x=None):
        if type_rotation == 'indexes':
            all_mat = np.load(self.data_path + "MNT/indexes_rot.npy").astype(np.int32).reshape(360, 79 * 69, 2)
            wind_large = np.full((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 140, 140, 3), np.nan, dtype=np.float32)
            wind_large = self.select_rotation(all_mat=all_mat, wind_large=wind_large, wind=wind, angles=angle,
                                              type_rotation='wind_indexes', library='numba')

        elif type_rotation == 'scipy':
            wind = np.moveaxis(wind, -1, 3)
            wind = wind.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 3, 79, 69))
            angle = angle.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1))
            wind_large = self.select_rotation(data=wind, wind_dir=angle, clockwise=True,
                                              type_rotation=type_rotation, GPU=GPU)
            wind_large = np.moveaxis(wind_large, 3, -1)

        elif type_rotation == "tfa":
            wind = wind.reshape((nb_time_step, nb_px_nwp_y*nb_px_nwp_x, 79, 69, 3))
            angle = angle.reshape((nb_time_step, nb_px_nwp_y*nb_px_nwp_x))
            wind_large = self.select_rotation(data=wind, wind_dir=angle, clockwise=True,
                                              type_rotation=type_rotation, GPU=GPU)
            wind_large = wind_large.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79, 69, 3))

        return wind_large

    @staticmethod
    def _select_center_to_replace_pixels(type_rotation):
        if type_rotation == "indexes":
            y_center = 70
            x_center = 70
        elif type_rotation in ["scipy", "tfa"]:
            y_center = 39
            x_center = 34
        return x_center, y_center

    @print_func_executed_decorator("interpolation mean KNN", level_begin="\n__", level_end="__")
    @timer_decorator("interpolation mean KNN", unit="second", level=". . . . ")
    def interpolation_mean_K_NN(self, high_resolution_wind, high_resolution_grid, low_resolution_grid, length_square,
                                x_name_LR="x", y_name_LR="y", x_name_HR="x", y_name_HR="y",
                                resolution_HR_x=None, resolution_HR_y=None):

        accepted_inputs = [xr.core.dataarray.DataArray, xr.core.dataset.Dataset]
        assert type(high_resolution_grid) in accepted_inputs
        assert type(low_resolution_grid) in accepted_inputs
        assert resolution_HR_x == resolution_HR_y

        resolution_x = self.mnt.resolution_x if resolution_HR_x is None else resolution_HR_x

        min_mnt = np.nanmin(high_resolution_grid[x_name_HR].values)
        max_mnt = np.nanmax(high_resolution_grid[y_name_HR].values)

        indexes_x, indexes_y = self.mnt.find_nearest_MNT_index(low_resolution_grid[x_name_LR],
                                                               low_resolution_grid[y_name_LR],
                                                               look_for_corners=False,
                                                               xmin_MNT=min_mnt,
                                                               ymax_MNT=max_mnt,
                                                               look_for_resolution=False,
                                                               resolution_x=resolution_x)

        nb_time_steps = high_resolution_wind.Wind.shape[0]
        nb_y_px = low_resolution_grid.ZS.shape[0]
        nb_x_px = low_resolution_grid.ZS.shape[1]
        shape_low_resolution_wind = (nb_time_steps, nb_y_px, nb_x_px, 3)
        array_LR_wind = np.empty(shape_low_resolution_wind)

        U = high_resolution_wind["U"]
        V = high_resolution_wind["V"]
        W = high_resolution_wind["W"]

        nb_pixels = length_square//(2*resolution_x)
        for idx_y in range(nb_y_px):
            for idx_x in range(nb_x_px):
                for idx_wind_component, wind_component in enumerate([U, V, W]):
                    for time in range(nb_time_steps):
                        x = indexes_x[idx_x]
                        y = indexes_y[idx_y]
                        array_LR_wind[time, idx_y, idx_x, idx_wind_component] = np.mean(wind_component.values[time, y-nb_pixels:y+nb_pixels, x-nb_pixels:x+nb_pixels])

        low_resolution_grid = low_resolution_grid.assign(U=(("time", y_name_LR, x_name_LR), array_LR_wind[:, :, :, 0]))
        low_resolution_grid = low_resolution_grid.assign(V=(("time", y_name_LR, x_name_LR), array_LR_wind[:, :, :, 1]))
        low_resolution_grid = low_resolution_grid.assign(W=(("time", y_name_LR, x_name_LR), array_LR_wind[:, :, :, 2]))

        return low_resolution_grid


    @print_func_executed_decorator("Predict maps", level_begin="\n", level_end="")
    @timer_decorator("Predict maps", unit="minute", level="")
    def _predict_maps(self, **kwargs):

        station_name = kwargs.get("station_name")
        dx = kwargs.get("dx")
        dy = kwargs.get("dy")
        interp = kwargs.get("interp")
        year_begin = kwargs.get("year_begin")
        month_begin = kwargs.get("month_begin")
        day_begin = kwargs.get("day_begin")
        hour_begin = kwargs.get("hour_begin")
        year_end = kwargs.get("year_end")
        month_end = kwargs.get("month_end")
        day_end = kwargs.get("day_end")
        hour_end = kwargs.get("hour_end")
        Z0 = kwargs.get("Z0")
        verbose = kwargs.get("verbose")
        peak_valley = kwargs.get("peak_valley")
        method = kwargs.get("method")
        type_rotation = kwargs.get("type_rotation")
        log_profile_to_h_2 = kwargs.get("log_profile_to_h_2")
        log_profile_from_h_2 = kwargs.get("log_profile_from_h_2")
        log_profile_10m_to_3m = kwargs.get("log_profile_10m_to_3m")
        log_profile_3m_to_10m = kwargs.get("log_profile_3m_to_10m")
        nb_pixels = kwargs.get("nb_pixels")
        interpolate_final_map = kwargs.get("interpolate_final_map")
        centered_on_interpolated = kwargs.get("centered_on_interpolated")
        scaling_function = kwargs.get("scaling_function")
        expose = kwargs.get("expose")
        interpolate_nwp = kwargs.get("interpolate_nwp")
        #scale_at_10m = kwargs.get("scale_at_10m")
        #scale_at_max_altitude = kwargs.get("scale_at_max_altitude")

        get_closer_learning_condition = kwargs.get("get_closer_learning_condition")
        coords_domain_for_map_prediction = kwargs.get("coords_domain_for_map_prediction")
        select_area = kwargs.get("select_area")
        GPU = kwargs.get("GPU")
        ten_m_array = None
        three_m_array = None

        # Select NWP data
        nwp_data = self.prepare_time_and_domain_nwp(year_begin, month_begin, day_begin, hour_begin,
                                                    year_end, month_end, day_end, hour_end,
                                                    select_area=select_area,
                                                    station_name=station_name, dx=dx, dy=dy,
                                                    coords=coords_domain_for_map_prediction,
                                                    additional_dx_nwp=4_000)

        # Interpolate NWP
        if interpolate_nwp:
            nwp_data = self.interpolate_wind_grid_xarray(nwp_data, interp=interp, method=method, verbose=verbose)
        times_to_save = nwp_data.time.values
        times, nb_time_step, nwp_x_l93, nwp_y_l93, nb_px_nwp_y, nb_px_nwp_x = self.get_characteristics_nwp(nwp_data)

        variables = ["Wind", "Wind_DIR", "Z0", "Z0REL", "ZS"] if Z0 else ["Wind", "Wind_DIR"]
        if Z0:
            wind_speed_nwp, wind_DIR_nwp, Z0_nwp, Z0REL_nwp, ZS_nwp = self.extract_from_xarray_to_numpy(nwp_data,
                                                                                                        variables)
        else:
            wind_speed_nwp, wind_DIR_nwp = self.extract_from_xarray_to_numpy(nwp_data, variables)

        # Select MNT data
        if centered_on_interpolated:
            mnt_data = self._select_large_domain_with_coordinates(type_input="MNT",
                                                                  coords=coords_domain_for_map_prediction,
                                                                  epsg_coords="2154",
                                                                  dx_mnt=4_000,
                                                                  x_mnt="x",
                                                                  y_mnt="y")
        else:
            mnt_data = self._select_large_domain_around_station(station_name, dx, dy,
                                                                type_input="MNT",
                                                                additional_dx_mnt=4_000)
        mnt_data = self.mnt.data_xr
        # Select MNT data characteristics
        xmin_mnt, ymax_mnt, resolution_x, resolution_y = self.get_characteristics_mnt(mnt_data)
        mnt_data, mnt_data_x, mnt_data_y, shape_x_mnt, shape_y_mnt = self.mnt._get_mnt_data_and_shape(mnt_data)
        coord = [mnt_data_x, mnt_data_y]

        # Initialize wind map
        wind_map = np.zeros((nb_time_step, shape_x_mnt, shape_y_mnt, 3), dtype=np.float32)

        # Constants
        nb_pixel = 70

        # Select indexes MNT
        idx_x_mnt, idx_y_mnt = self.mnt.find_nearest_MNT_index(nwp_x_l93[:, :],
                                                               nwp_y_l93[:, :],
                                                               look_for_corners=False,
                                                               xmin_MNT=xmin_mnt,
                                                               ymax_MNT=ymax_mnt,
                                                               look_for_resolution=False,
                                                               resolution_x=resolution_x,
                                                               resolution_y=resolution_y)

        # Create a vector containing all topographies around NWP pixels
        topo_i = np.empty((nb_px_nwp_y, nb_px_nwp_x, 140, 140)).astype(np.float32)
        topo_i = self.hstack_topo(topo_i, idx_x_mnt, idx_y_mnt, mnt_data, nb_pixel, library='numba')

        # Wind direction
        angle = np.where(wind_DIR_nwp > 0, np.int32(wind_DIR_nwp - 1), np.int32(359))

        # Rotate topography
        topo_rot = self._rotation_wrapper_for_topo_in_map_predictions(topo_i, angle, type_rotation,
                                                                      GPU=GPU,
                                                                      nb_time_step=nb_time_step,
                                                                      nb_px_nwp_y=nb_px_nwp_y,
                                                                      nb_px_nwp_x=nb_px_nwp_x)

        topo_rot = topo_rot.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79, 69)).astype(np.float32)
        mean_height = np.nanmean(topo_rot, axis=(3, 4)).astype(np.float32)

        # Normalize
        _, std = self._load_norm_prm()
        mean_height = mean_height.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1))
        if get_closer_learning_condition:
            std = self.get_closer_from_learning_conditions(topo_rot, mean_height, std, axis=(3, 4))
        std = std.reshape((1, 1, 1, 1, 1)).astype(dtype=np.float32, copy=False)
        topo_rot = self.normalize_topo(topo_rot, mean_height, std).astype(dtype=np.float32, copy=False)

        del std
        del mean_height

        # Load model
        self.load_cnn(dependencies=True)

        # Predictions
        topo_rot = topo_rot.reshape((nb_time_step * nb_px_nwp_x * nb_px_nwp_y, self.n_rows, self.n_col, 1))
        prediction = self.cnn_prediction_map(topo_rot, GPU=GPU)

        # Reshape predictions for analysis and broadcasting
        prediction = prediction.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, self.n_rows, self.n_col, 3)).astype(
            np.float32, copy=False)
        wind_speed_nwp = wind_speed_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1)).astype(np.float32,
                                                                                                          copy=False)
        wind_DIR_nwp = wind_DIR_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1)).astype(np.float32,
                                                                                                   copy=False)

        #acceleration_cnn = self.wind_speed_ratio(num=prediction, den=3)
        #print_statistical_description_array(acceleration_cnn, name="Acceleration CNN")

        # Exposed wind speed
        if expose is not None:

            z_expose = self.select_z_expose(topo_i, expose, shape=(nb_px_nwp_y, nb_px_nwp_x))

            # Reshape for broadcasting
            Z0_nwp = Z0_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            Z0REL_nwp = Z0REL_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            ZS_nwp = ZS_nwp.reshape((1, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            z_expose = z_expose.reshape((1, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))

            # Apply log profile: 10m => h/2 or 10m => Zs
            if log_profile_to_h_2:
                ten_m_array = np.full_like(ZS_nwp, 10) if ten_m_array is None else ten_m_array
                wind_speed_nwp = self.apply_log_profile(z_in=ten_m_array, z_out=z_expose,
                                                        wind_in=wind_speed_nwp, z0=Z0_nwp,
                                                        verbose=verbose, z_in_verbose="10m", z_out_verbose=expose)

            # Expose wind speed
            exp_Wind, acceleration_factor = self.exposed_wind_speed(wind_speed=wind_speed_nwp,
                                                                    z_out=z_expose,
                                                                    z0=Z0_nwp,
                                                                    z0_rel=Z0REL_nwp,
                                                                    verbose=True)

            del Z0REL_nwp

            if log_profile_from_h_2:
                # Apply log profile: h/2 => 10m or Zs => 10m
                ten_m_array = np.full_like(ZS_nwp, 10) if ten_m_array is None else ten_m_array
                exp_Wind = self.apply_log_profile(z_in=z_expose, z_out=ten_m_array,
                                                  wind_in=exp_Wind, z0=Z0_nwp,
                                                  verbose=verbose, z_in_verbose=expose, z_out_verbose="10m")

            # Wind speed scaling
            scaling_wind = exp_Wind.view()
        else:
            scaling_wind = wind_speed_nwp.view()

        prediction = self.wind_speed_scaling(scaling_wind, prediction, type_scaling=scaling_function)

        if log_profile_3m_to_10m:
            # Apply log profile: 3m => 10m
            three_m_array = np.full_like(ZS_nwp, 10) if three_m_array is None else three_m_array
            ten_m_array = np.full_like(ZS_nwp, 10) if ten_m_array is None else ten_m_array
            prediction = self.apply_log_profile(z_in=three_m_array, z_out=ten_m_array, wind_in=prediction, z0=Z0_nwp,
                                                verbose=verbose, z_in_verbose="3m", z_out_verbose="10m")

        del ZS_nwp

        # Wind computations
        U_old = prediction[:, :, :, :, :, 0].view(dtype=np.float32)  # Expressed in the rotated coord. system [m/s]
        V_old = prediction[:, :, :, :, :, 1].view(dtype=np.float32)  # Expressed in the rotated coord. system [m/s]
        # W_old = prediction[:, :, :, :, :, 2].view(dtype=np.float32)  # Good coord. but not on the right pixel [m/s]

        # Recalculate with respect to original coordinates
        UV = self.compute_wind_speed(U=U_old, V=V_old, W=None)  # Good coord. but not on the right pixel [m/s]
        alpha = self.angular_deviation(U_old, V_old,
                                       unit_output="radian")  # Expressed in the rotated coord. system [radian]
        UV_DIR = self.direction_from_alpha(wind_DIR_nwp, alpha,
                                           unit_direction="degree",
                                           unit_alpha="radian",
                                           unit_output="radian")  # Good coord. but not on the right pixel [radian]

        del Z0_nwp
        del alpha

        # Reshape wind speed
        wind_speed_nwp = wind_speed_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x))

        # Calculate U and V along initial axis
        # Good coord. but not on the right pixel [m/s]
        prediction[:, :, :, :, :, 0], prediction[:, :, :, :, :, 1] = self.horizontal_wind_component(UV=UV,
                                                                                                    UV_DIR=UV_DIR,
                                                                                                    unit_direction="radian")
        del UV_DIR
        del UV

        # Reduce size matrix of indexes
        prediction = prediction.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79 * 69, 3))

        # Rotate back
        wind_large = self._rotation_wrapper_for_wind_in_predict_maps(prediction, angle, type_rotation,
                                                                     GPU=GPU, nb_time_step=nb_time_step,
                                                                     nb_px_nwp_y=nb_px_nwp_y, nb_px_nwp_x=nb_px_nwp_x)
        # Replace pixels on map
        wind_map, acceleration_all = self.replace_pixels_on_map(mnt_map=wind_map, wind=wind_large,
                                                                idx_x_mnt=idx_x_mnt, idx_y_mnt=idx_y_mnt,
                                                                type_rotation=type_rotation,
                                                                wind_speed_nwp=wind_speed_nwp, save_pixels=nb_pixels,
                                                                acceleration=True, library="numba")

        del wind_large
        del angle
        del prediction

        if interpolate_final_map:
            wind_map = self.interpolate_final_result(wind_map, library="numba")

        if centered_on_interpolated:

            wind_map = wind_map.reshape((nb_time_step, shape_x_mnt, shape_y_mnt, 3))
            wind_xr = xr.Dataset(
                data_vars=dict(
                    U=(["time", "y", "x"], wind_map[:, :, :, 0]),
                    V=(["time", "y", "x"], wind_map[:, :, :, 1]),
                    W=(["time", "y", "x"], wind_map[:, :, :, 2])),
                coords=dict(
                    x=(["x"], coord[0]),
                    y=(["y"], coord[1]),
                    time=times_to_save))

            return wind_xr

        return wind_map, acceleration_all, coord, _, nwp_data, mnt_data

    def predict_maps(self, line_profile=False, memory_profile=False, prm=None):
        """
        This function is used to select map predictions, the line profiled version or the memory profiled version
        """

        line_profile = prm["line_profile"] if prm["line_profile"] is not None else line_profile
        memory_profile = prm["memory_profile"] if prm["memory_profile"] is not None else memory_profile

        if line_profile:
            from line_profiler import LineProfiler
            lp = LineProfiler()
            lp_wrapper = lp(self._predict_maps)
            lp_wrapper(**prm)
            lp.print_stats()
        elif memory_profile:
            from memory_profiler import profile
            mp = profile(self._predict_maps)
            mp(**prm)
        else:
            array_xr = self._predict_maps(**prm)
            return array_xr



