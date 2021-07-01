# Packages: numpy, pandas, xarray, scipy, tensorflow, matplotlib
import numpy as np
import pandas as pd
import xarray as xr
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.colors as colors
import random
import datetime

# Local imports
from downscale.Utils.GPU import environment_GPU
from downscale.Utils.Utils import assert_equal_shapes, reshape_list_array
from downscale.Operators.Rotation import Rotation
from downscale.Operators.wind_utils import Wind_utils
from downscale.Operators.topo_utils import Topo_utils

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
class Processing(Wind_utils, Topo_utils, Rotation):
    n_rows, n_col = 79, 69
    _geopandas = _geopandas
    _shapely_geometry = _shapely_geometry

    def __init__(self, obs=None, mnt=None, nwp=None, model_path=None, prm=None):

        super().__init__()

        GPU = prm["GPU"]
        data_path = prm['data_path']

        self.observation = obs
        self.mnt = mnt
        self.nwp = nwp
        self.model_path = model_path
        self.data_path = data_path
        environment_GPU(GPU=GPU)

    def _extract_variable_from_nwp_at_station(self,
                                              station_name,
                                              variable_to_extract=["time", "wind_speed", "wind_direction", "Z0",
                                                                   "Z0REL", "ZS"],
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
        y_idx_nwp, x_idx_nwp = stations[f"index_{nwp_name}_NN_0"][stations["name"] == station_name].values[0]
        y_idx_nwp, x_idx_nwp = np.int16(y_idx_nwp), np.int16(x_idx_nwp)

        # Select NWP data
        nwp_instance = self.nwp.data_xr

        results = [0 for k in range(len(variable_to_extract))]
        if "time" in variable_to_extract:
            time_index = nwp_instance.time.data
            results[variable_to_extract.index("time")] = time_index

        if "wind_speed" in variable_to_extract:
            wind_speed = nwp_instance.Wind.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
            results[variable_to_extract.index("wind_speed")] = wind_speed

        if "wind_direction" in variable_to_extract:
            wind_dir = nwp_instance.Wind_DIR.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
            results[variable_to_extract.index("wind_direction")] = wind_dir

        if "Z0" in variable_to_extract:
            Z0 = nwp_instance.Z0.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
            results[variable_to_extract.index("Z0")] = Z0

        if "Z0REL" in variable_to_extract:
            Z0REL = nwp_instance.Z0REL.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
            results[variable_to_extract.index("Z0REL")] = Z0REL

        if "ZS" in variable_to_extract:
            ZS = nwp_instance.ZS.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
            results[variable_to_extract.index("ZS")] = ZS

        print(f"Selected time series for pixel at station: {station_name}") if verbose else None

        results = results[0] if len(results) == 1 else results
        return (results)

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

    def load_model(self, dependencies=False):
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
            model = load_model(self.model_path + "fold_0.h5", custom_objects=dependencies)
        else:
            model = load_model(self.model_path + "fold_0.h5")

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

    def _select_timeframe_nwp(self, begin=None, end=None, ideal_case=False, verbose=True):
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
            begin = self.nwp.begin
            year, month, day = np.int16(begin.split('-'))
            end = str(year) + "-" + str(month) + "-" + str(day + 1)

        self.nwp.select_timeframe(begin=begin, end=end)

        print("__NWP time window selected") if verbose else None

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
            all_topo_HD = np.empty((nb_station, self.n_rows, self.n_col), dtype=np.uint16)
            all_topo_x_small_l93 = np.empty((nb_station, self.n_col), dtype=np.float32)
            all_topo_y_small_l93 = np.empty((nb_station, self.n_rows), dtype=np.float32)
            ten_m_array = 10 * np.ones((nb_station, nb_sim), dtype=np.float32)
            three_m_array = 3 * np.ones((nb_station, nb_sim), dtype=np.float32)
            list_arrays_1 = [topo, wind_speed_all, wind_dir_all, Z0_all, Z0REL_all, ZS_all, peak_valley_height,
                             mean_height]
            list_arrays_2 = [all_topo_HD, all_topo_x_small_l93, all_topo_y_small_l93, ten_m_array, three_m_array]
            list_return = list_arrays_1 + list_arrays_2
        return list_return

    def predict_at_stations(self, stations_name, line_profile=None, **kwargs):
        """
        This function is used to select predictions at stations, the line profiled version or the memory profiled version
        """

        if line_profile:
            from line_profiler import LineProfiler
            lp = LineProfiler()
            lp_wrapper = lp(self._predict_at_stations)
            lp_wrapper(stations_name, **kwargs)
            lp.print_stats()
        else:
            array_xr = self._predict_at_stations(stations_name, **kwargs)
            return array_xr

    @staticmethod
    def select_height_for_exposed_wind_speed(height=None, zs=None, peak_valley=None):
        if peak_valley:
            return height
        else:
            return zs

    @staticmethod
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

    def _predict_at_stations(self, stations_name, verbose=True, Z0=True, peak_valley=True,
                             log_profile_to_h_2=False, log_profile_from_h_2=False, log_profile_10m_to_3m=False,
                             ideal_case=False, input_speed=3, input_dir=270, **kwargs):
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

        # Select timeframe
        self._select_timeframe_nwp(ideal_case=ideal_case, verbose=True)

        # Simulation parameters
        time_xr = self._extract_variable_from_nwp_at_station(random.choice(stations_name), variable_to_extract=["time"])
        nb_sim = len(time_xr)
        nb_station = len(stations_name)

        # initialize arrays
        topo, wind_speed_all, wind_dir_all, Z0_all, Z0REL_all, ZS_all, \
        peak_valley_height, mean_height, all_topo_HD, all_topo_x_small_l93, all_topo_y_small_l93, ten_m_array, \
        three_m_array = self._initialize_arrays(predict='stations_month', nb_station=nb_station, nb_sim=nb_sim)

        # Indexes
        nb_pixel = 70  # min = 116/2
        y_offset_left = nb_pixel - 39
        y_offset_right = nb_pixel + 40
        x_offset_left = nb_pixel - 34
        x_offset_right = nb_pixel + 35

        for idx_station, single_station in enumerate(stations_name):

            print(f"\nBegin downscaling at {single_station}")

            # Select nwp pixel
            wind_dir_all[idx_station, :], wind_speed_all[idx_station, :], Z0_all[idx_station, :], \
            Z0REL_all[idx_station, :], ZS_all[idx_station, :] = self._extract_variable_from_nwp_at_station(
                single_station,
                variable_to_extract=["wind_direction", "wind_speed", "Z0", "Z0REL", "ZS"])

            # For ideal case, we define the input speed and direction
            if ideal_case:
                wind_speed_all[idx_station, :], \
                wind_dir_all[idx_station, :] = self._scale_wind_for_ideal_case(wind_speed_all[idx_station, :],
                                                                               wind_dir_all[idx_station, :],
                                                                               input_speed,
                                                                               input_dir)

            # Extract topography
            topo_HD, topo_x_l93, topo_y_l93 = self.observation.extract_MNT_around_station(single_station,
                                                                                          self.mnt,
                                                                                          nb_pixel,
                                                                                          nb_pixel)

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

        # Exposed wind
        if Z0:
            peak_valley_height = peak_valley_height.reshape((nb_station, 1))
            Z0_all = np.where(Z0_all == 0, 1 * 10 ^ (-8), Z0_all)
            height = self.select_height_for_exposed_wind_speed(height=peak_valley_height,
                                                               zs=ZS_all,
                                                               peak_valley=peak_valley)
            wind1 = np.copy(wind_speed_all)

            # Log profile
            if log_profile_to_h_2:
                wind_speed_all = self.apply_log_profile(z_in=ten_m_array, z_out=height / 2, wind_in=wind_speed_all,
                                                        z0=Z0_all,
                                                        verbose=verbose, z_in_verbose="10m", z_out_verbose="height/2")
            a1 = self.wind_speed_ratio(num=wind_speed_all, den=wind1)
            wind2 = np.copy(wind_speed_all)

            # Expose wind
            exp_Wind, acceleration_factor = self.exposed_wind_speed(wind_speed=wind_speed_all,
                                                                    z_out=height / 2,
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
        std = self.get_closer_from_learning_conditions(topo, mean_height, std, axis=(2, 3))
        mean_height = mean_height.reshape((nb_station, 1, 1, 1, 1))
        std = std.reshape((nb_station, nb_sim, 1, 1, 1))
        topo = self.normalize_topo(topo, mean_height, std)

        # Reshape for tensorflow
        topo = topo.reshape((nb_station * nb_sim, self.n_rows, self.n_col, 1))
        print('__Reshaped tensorflow done') if verbose else None

        """
        Warning: change dependencies here
        """
        # Load model
        self.load_model(dependencies=True)

        # Predictions
        prediction = self.model.predict(topo)
        print('__Prediction done') if verbose else None

        # Acceleration NWP to CNN
        UVW_int = self.compute_wind_speed(U=prediction[:, :, :, 0], V=prediction[:, :, :, 1], W=prediction[:, :, :, 2])
        acceleration_CNN = self.wind_speed_ratio(num=UVW_int, den=3 * np.ones(UVW_int.shape))

        # Reshape predictions for analysis
        prediction = prediction.reshape((nb_station, nb_sim, self.n_rows, self.n_col, 3))
        print(f"__Prediction reshaped: {prediction.shape}") if verbose else None

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
        prediction = self.wind_speed_scaling(scaling_wind, prediction, linear=True)

        # Copy wind variable
        wind4 = np.copy(prediction)

        if log_profile_10m_to_3m:
            # Apply log profile: 3m => 10m
            prediction = self.apply_log_profile(z_in=three_m_array, z_out=ten_m_array, wind_in=prediction, z0=Z0_all,
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
        alpha = self.angular_deviation(U_old, V_old)  # Expressed in the rotated coord. system [radian]
        UV_DIR = self.direction_from_alpha(wind_dir_all, alpha)  # Good coord. but not on the right pixel [radian]

        # Verification of shapes
        assert_equal_shapes([U_old, V_old, W_old, UV, alpha, UV_DIR], (nb_station, nb_sim, self.n_rows, self.n_col))

        # Calculate U and V along initial axis
        # Good coord. but not on the right pixel [m/s]
        prediction[:, :, :, :, 0], prediction[:, :, :, :, 1] = self.horizontal_wind_component(UV=UV,
                                                                                              UV_DIR=UV_DIR,
                                                                                              verbose=True)
        del UV_DIR

        # Rotate clockwise to put the wind value on the right topography pixel
        print('__Start rotating to initial position') if verbose else None
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

        print('__Wind prediction rotated for initial topography') if verbose else None

        # Compute wind direction
        UV_DIR = self.direction_from_u_and_v(U, V)  # Good axis and pixel [degree]

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

        print('__Reshape final predictions done') if verbose else None

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

    def _select_large_domain_around_station(self, station_name, dx, dy, type_input="NWP", additional_dx_mnt=None):
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

        return data_xr

    @staticmethod
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

    def prepare_time_and_domain_nwp(self, year_0, month_0, day_0, hour_0, year_1, month_1, day_1, hour_1,
                                    station_name=None, dx=None, dy=None, additional_dx_mnt=None, verbose=True):

        begin = datetime.datetime(year_0, month_0, day_0, hour_0)
        end = datetime.datetime(year_1, month_1, day_1, hour_1)
        self._select_timeframe_nwp(begin=begin, end=end)

        nwp_data = self._select_large_domain_around_station(station_name, dx, dy, type_input="NWP",
                                                            additional_dx_mnt=additional_dx_mnt)
        if verbose: print("__Prepare time and domain NWP")
        return nwp_data

    def interpolate_wind_grid_xarray(self, nwp_data, interp=3, method='linear', verbose=True):

        # Calculate U_nwp and V_nwp
        nwp_data = self.horizontal_wind_component(working_with_xarray=True, xarray_data=nwp_data)

        # Drop variables
        nwp_data = nwp_data.drop_vars(["Wind", "Wind_DIR"])

        # Interpolate AROME
        nwp_data = self.interpolate_xarray_grid(xarray_data=nwp_data, interp=interp, method=method)
        nwp_data = self.compute_wind_speed(computing='xarray', xarray_data=nwp_data)

        return nwp_data

    def get_caracteristics_nwp(self, nwp_data):
        times = nwp_data.time.data.astype(np.float32)
        nb_time_step = len(times)
        nwp_x_l93 = nwp_data.X_L93.data.astype(np.float32)
        nwp_y_l93 = nwp_data.Y_L93.data.astype(np.float32)
        nb_px_nwp_y, nb_px_nwp_x = nwp_x_l93.shape
        return times, nb_time_step, nwp_x_l93, nwp_y_l93, nb_px_nwp_y, nb_px_nwp_x

    def get_caracteristics_mnt(self, mnt_data, verbose=True):

        xmin_mnt = np.nanmin(mnt_data.x.data)
        ymax_mnt = np.nanmax(mnt_data.y.data)
        resolution_x = self.mnt.resolution_x
        resolution_y = self.mnt.resolution_y

        print("__Selected NWP caracteristics") if verbose else None

        return xmin_mnt, ymax_mnt, resolution_x, resolution_y

    @staticmethod
    def extract_from_xarray_to_numpy(array, list_variables, verbose=True):
        print("__Variables extracted from xarray data") if verbose else None
        return (array[variable].data.astype(np.float32) for variable in list_variables)

    @staticmethod
    def _iterate_to_replace_pixels(wind, mnt_map, idx_y_mnt, idx_x_mnt, save_pixels, y_offset_left, y_offset_right,
                                   x_offset_left, x_offset_right):
        for j in range(wind.shape[1]):
            for i in range(wind.shape[2]):
                mnt_y_left = idx_y_mnt[j, i] - save_pixels
                mnt_y_right = idx_y_mnt[j, i] + save_pixels + 1
                mnt_x_left = idx_x_mnt[j, i] - save_pixels
                mnt_x_right = idx_x_mnt[j, i] + save_pixels + 1

                mnt_map[:, mnt_y_left:mnt_y_right, mnt_x_left:mnt_x_right, :] = wind[:, j, i,
                                                                                y_offset_left:y_offset_right,
                                                                                x_offset_left:x_offset_right, :]
        return mnt_map

    def _replace_pixels_on_map(self, mnt_map=None, wind=None, idx_x_mnt=None, idx_y_mnt=None, library='num',
                               y_offset_left=None, y_offset_right=None, x_offset_left=None, x_offset_right=None,
                               save_pixels=None, verbose=True):

        if library == 'numba' and _numba:
            jit_rpl_px = jit([float32[:, :, :, :](float32[:, :, :, :, :, :], float32[:, :, :, :], int64[:, :],
                                                  int64[:, :], int64, int64, int64, int64, int64)],
                             nopython=True)(self._iterate_to_replace_pixels)
            result = jit_rpl_px(wind, mnt_map, idx_y_mnt, idx_x_mnt, save_pixels, y_offset_left, y_offset_right,
                                x_offset_left, x_offset_right)
            print("____Used numba to replace pixels") if verbose else None

        else:
            result = self._iterate_to_replace_pixels(wind, mnt_map, idx_y_mnt, idx_x_mnt, save_pixels, y_offset_left,
                                                     y_offset_right, x_offset_left, x_offset_right)
            print("____Used numpy to replace pixels") if verbose else None

        return result

    def replace_pixels_on_map(self, mnt_map=None, wind=None, idx_x_mnt=None, idx_y_mnt=None,
                              x_center=None, y_center=None, wind_speed_nwp=None, save_pixels=15, acceleration=False,
                              library='numba', verbose=True):

        y_offset_left = y_center - save_pixels
        y_offset_right = y_center + save_pixels + 1
        x_offset_left = x_center - save_pixels
        x_offset_right = x_center + save_pixels + 1

        print("__Replaced pixels on map") if verbose else None

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

    def interpolate_final_result(self, wind_map, library='numba', verbose=True):
        if library == 'numba' and _numba:
            jit_int = jit([float32[:, :, :, :](float32[:, :, :, :])], nopython=True)(self._interpolate_array)
            result = jit_int(wind_map)
            print("____Used numba to perform final interpolation") if verbose else None
        else:
            result = self._interpolate_array(wind_map)
            print("____Used numpy to perform final interpolation") if verbose else None
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

    def hstack_topo(self, topo_i, idx_x_mnt, idx_y_mnt, mnt_data, nb_pixel, library='numba', verbose=True):

        if library == 'numba' and _numba:
            jit_hstack = jit(
                [float32[:, :, :, :](float32[:, :, :, :], int64[:, :], int64[:, :], float32[:, :, :], int64)],
                nopython=True)(self._hstack_topo)
            result = jit_hstack(topo_i, idx_x_mnt, idx_y_mnt, mnt_data, nb_pixel)
            print("____Used numba to stack horizontally topographies") if verbose else None

        else:
            result = self._hstack_topo(topo_i, idx_x_mnt, idx_y_mnt, mnt_data, nb_pixel)
            print("____Used numpy to stack horizontally topographies") if verbose else None

        return result

    @staticmethod
    def _hstack_topo(topo_i, idx_x_mnt, idx_y_mnt, mnt_data, nb_pixel):
        for j in range(topo_i.shape[0]):
            for i in range(topo_i.shape[1]):
                y = idx_y_mnt[j, i]
                x = idx_x_mnt[j, i]
                topo_i[j, i, :, :] = mnt_data[0, y - nb_pixel:y + nb_pixel, x - nb_pixel:x + nb_pixel]
        return topo_i

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
        print("__Extracted indexes at stations") if verbose else None

        return idx_x_stations, idx_y_stations, stations.values

    # todo save indexes second rotation
    def _predict_maps(self, station_name='Col du Lac Blanc', x_0=None, y_0=None, dx=10_000, dy=10_000, interp=3,
                      year_begin=None, month_begin=None, day_begin=None, hour_begin=None,
                      year_end=None, month_end=None, day_end=None, hour_end=None,
                      Z0=False, verbose=True, peak_valley=True, method='linear', type_rotation='indexes',
                      log_profile_to_h_2=False, log_profile_from_h_2=False, log_profile_10m_to_3m=False,
                      nb_pixels=15, interpolate_final_map=True, extract_stations_only=False, **kwargs):

        # Select NWP data
        nwp_data = self.prepare_time_and_domain_nwp(year_begin, month_begin, day_begin, hour_begin,
                                                    year_end, month_end, day_end, hour_end,
                                                    station_name=station_name, dx=dx, dy=dy)
        nwp_data_initial = nwp_data.copy(deep=False)
        nwp_data = self.interpolate_wind_grid_xarray(nwp_data, interp=interp, method=method, verbose=verbose)
        times, nb_time_step, nwp_x_l93, nwp_y_l93, nb_px_nwp_y, nb_px_nwp_x = self.get_caracteristics_nwp(nwp_data)
        variables = ["Wind", "Wind_DIR", "Z0", "Z0REL", "ZS"] if Z0 else ["Wind", "Wind_DIR"]
        if Z0:
            wind_speed_nwp, wind_DIR_nwp, Z0_nwp, Z0REL_nwp, ZS_nwp = self.extract_from_xarray_to_numpy(nwp_data,
                                                                                                        variables)
        else:
            wind_speed_nwp, wind_DIR_nwp = self.extract_from_xarray_to_numpy(nwp_data, variables)

        # Select MNT data
        mnt_data = self._select_large_domain_around_station(station_name, dx, dy, type_input="MNT",
                                                            additional_dx_mnt=2_000)
        xmin_mnt, ymax_mnt, resolution_x, resolution_y = self.get_caracteristics_mnt(mnt_data)
        mnt_data, mnt_data_x, mnt_data_y, shape_x_mnt, shape_y_mnt = self.mnt._get_mnt_data_and_shape(mnt_data)
        coord = [mnt_data_x, mnt_data_y]

        # Initialize wind map
        wind_map = np.zeros((nb_time_step, shape_x_mnt, shape_y_mnt, 3), dtype=np.float32)
        peak_valley_height = np.empty((nb_px_nwp_y, nb_px_nwp_x), dtype=np.float32)
        mean_height = np.empty((nb_px_nwp_y, nb_px_nwp_x), dtype=np.float32)

        # Constants
        nb_pixel = 70

        # Load pre_rotated indexes
        all_mat = np.load(self.data_path + "MNT/indexes_rot.npy").astype(np.int32).reshape(360, 79 * 69, 2)

        # Select indexes MNT
        idx_x_mnt, idx_y_mnt = self.mnt.find_nearest_MNT_index(nwp_x_l93[:, :],
                                                               nwp_y_l93[:, :],
                                                               look_for_corners=False,
                                                               xmin_MNT=xmin_mnt,
                                                               ymax_MNT=ymax_mnt,
                                                               look_for_resolution=False,
                                                               resolution_x=resolution_x,
                                                               resolution_y=resolution_y)

        # Large topo
        topo_i = np.empty((nb_px_nwp_y, nb_px_nwp_x, 140, 140)).astype(np.float32)
        topo_i = self.hstack_topo(topo_i, idx_x_mnt, idx_y_mnt, mnt_data, nb_pixel, library='numba')

        # Mean peak_valley altitude
        peak_valley_height[:, :] = self.mean_peak_valley(topo_i, )
        mean_height[:, :] = np.int32(np.mean(topo_i))

        # Wind direction
        angle = np.where(wind_DIR_nwp > 0, np.int32(wind_DIR_nwp - 1), np.int32(359))

        # Rotate topography
        if type_rotation == 'indexes':
            topo_rot = np.empty((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79 * 69)).astype(np.float32)
            # all_mat=(360, 5451, 2), topo_rot=(1, 21, 21, 5451), topo_i=(21, 21, 140, 140), angle=(1, 21, 21)
            topo_rot = self.select_rotation(all_mat=all_mat, topo_rot=topo_rot, topo_i=topo_i, angles=angle,
                                            type_rotation='topo_indexes', library='numba')
        if type_rotation == 'scipy':
            nb_pixel = 70  # min = 116/2
            y_left = nb_pixel - 39
            y_right = nb_pixel + 40
            x_left = nb_pixel - 34
            x_right = nb_pixel + 35
            topo_i = topo_i.reshape((nb_px_nwp_y, nb_px_nwp_x, 140, 140))
            topo_rot = self.select_rotation(data=topo_i,
                                            wind_dir=angle,
                                            clockwise=False)[:, :, :, y_left:y_right, x_left:x_right]
        topo_rot = topo_rot.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79, 69))

        # Normalize
        _, std = self._load_norm_prm()
        mean_height = mean_height.reshape((1, nb_px_nwp_y, nb_px_nwp_x, 1, 1))
        std = self.get_closer_from_learning_conditions(topo_rot, mean_height, std, axis=(3, 4))
        std = std.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1)).astype(dtype=np.float32, copy=False)
        topo_rot = self.normalize_topo(topo_rot, mean_height, std).astype(dtype=np.float32, copy=False)

        del std
        del mean_height

        # Reshape for tensorflow
        topo_rot = topo_rot.reshape((nb_time_step * nb_px_nwp_x * nb_px_nwp_y, self.n_rows, self.n_col, 1))

        # Load model
        self.load_model(dependencies=True)

        # Predictions
        prediction = self.model.predict(topo_rot)

        # Reshape predictions for analysis and broadcasting
        prediction = prediction.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, self.n_rows, self.n_col, 3)).astype(
            np.float32, copy=False)
        wind_speed_nwp = wind_speed_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1)).astype(np.float32,
                                                                                                          copy=False)
        wind_DIR_nwp = wind_DIR_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1)).astype(np.float32,
                                                                                                   copy=False)
        acceleration_cnn = prediction / 3
        print("____Acceleration maximum CNN", np.nanmax(acceleration_cnn))

        # Exposed wind speed
        if Z0:

            # Reshape for broadcasting
            Z0_nwp = Z0_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            Z0REL_nwp = Z0REL_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            ZS_nwp = ZS_nwp.reshape((1, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            peak_valley_height = peak_valley_height.reshape((1, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            ten_m_array = np.zeros_like(peak_valley_height) + 10
            three_m_array = np.zeros_like(peak_valley_height) + 10

            # Choose height in the formula
            height = self.select_height_for_exposed_wind_speed(height=peak_valley_height,
                                                               zs=ZS_nwp,
                                                               peak_valley=peak_valley)

            # Apply log profile: 10m => h/2 or 10m => Zs
            if log_profile_to_h_2:
                wind_speed_nwp = self.apply_log_profile(z_in=ten_m_array, z_out=peak_valley_height / 2,
                                                        wind_in=wind_speed_nwp, z0=Z0_nwp,
                                                        verbose=verbose, z_in_verbose="10m", z_out_verbose="height/2")
            # Unexpose wind speed
            exp_Wind, acceleration_factor = self.exposed_wind_speed(wind_speed=wind_speed_nwp,
                                                                    z_out=(height / 2),
                                                                    z0=Z0_nwp,
                                                                    z0_rel=Z0REL_nwp,
                                                                    verbose=True)

            del Z0REL_nwp

            if log_profile_from_h_2:
                # Apply log profile: h/2 => 10m or Zs => 10m
                exp_Wind = self.apply_log_profile(z_in=peak_valley_height / 2, z_out=ten_m_array,
                                                  wind_in=exp_Wind, z0=Z0_nwp,
                                                  verbose=verbose, z_in_verbose="height/2", z_out_verbose="10m")

        # Wind speed scaling
        scaling_wind = exp_Wind.view() if Z0 else wind_speed_nwp.view()
        prediction = self.wind_speed_scaling(scaling_wind, prediction, linear=True)

        if log_profile_10m_to_3m:
            # Apply log profile: 3m => 10m
            prediction = self.apply_log_profile(z_in=three_m_array, z_out=ten_m_array, wind_in=prediction, z0=Z0_nwp)

        # Wind computations
        U_old = prediction[:, :, :, :, :, 0].view(dtype=np.float32)  # Expressed in the rotated coord. system [m/s]
        V_old = prediction[:, :, :, :, :, 1].view(dtype=np.float32)  # Expressed in the rotated coord. system [m/s]
        # W_old = prediction[:, :, :, :, :, 2].view(dtype=np.float32)  # Good coord. but not on the right pixel [m/s]

        # Recalculate with respect to original coordinates
        UV = self.compute_wind_speed(U=U_old, V=V_old, W=None)  # Good coord. but not on the right pixel [m/s]
        alpha = self.angular_deviation(U_old, V_old)  # Expressed in the rotated coord. system [radian]
        UV_DIR = self.direction_from_alpha(wind_DIR_nwp, alpha)  # Good coord. but not on the right pixel [radian]

        del Z0_nwp
        del alpha

        # Reshape wind speed
        wind_speed_nwp = wind_speed_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x))

        # Calculate U and V along initial axis
        # Good coord. but not on the right pixel [m/s]
        prediction[:, :, :, :, :, 0], prediction[:, :, :, :, :, 1] = self.horizontal_wind_component(UV=UV,
                                                                                                    UV_DIR=UV_DIR)

        del UV_DIR
        del UV

        # Reduce size matrix of indexes
        wind = prediction.view(dtype=np.float32).reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79 * 69, 3))

        if type_rotation == 'indexes':
            y_center = 70
            x_center = 70
            wind_large = np.full((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 140, 140, 3), np.nan, dtype=np.float32)
            wind_large = self.select_rotation(all_mat=all_mat, wind_large=wind_large, wind=wind, angles=angle,
                                              type_rotation='wind_indexes', library='numba')
        if type_rotation == 'scipy':
            y_center = 39
            x_center = 34
            wind = np.moveaxis(wind, -1, 3)
            wind = wind.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 3, 79, 69))
            angle = angle.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1))
            wind_large = self.select_rotation(data=wind, wind_dir=angle, clockwise=True)
            wind_large = np.moveaxis(wind_large, 3, -1)

        wind_map, acceleration_all = self.replace_pixels_on_map(mnt_map=wind_map, wind=wind_large,
                                                                idx_x_mnt=idx_x_mnt, idx_y_mnt=idx_y_mnt,
                                                                x_center=x_center, y_center=y_center,
                                                                wind_speed_nwp=wind_speed_nwp, save_pixels=nb_pixels,
                                                                acceleration=True, library='numpy')

        del wind_large
        del angle
        del wind

        if interpolate_final_map:
            wind_map = self.interpolate_final_result(wind_map, library='numpy')

        if extract_stations_only:
            idx_x_stations, idx_y_stations, stations = self.extract_downscaled_wind_at_stations(mnt_data_x=mnt_data_x,
                                                                                                mnt_data_y=mnt_data_y,
                                                                                                xmin_mnt=xmin_mnt,
                                                                                                ymax_mnt=ymax_mnt,
                                                                                                resolution_x=resolution_x,
                                                                                                resolution_y=resolution_y)

            return wind_map[:, idx_y_stations, idx_x_stations, :], \
                   acceleration_all[:, idx_y_stations, idx_x_stations], \
                   nwp_data_initial.time.values, \
                   stations, [], []

        return wind_map, acceleration_all, coord, nwp_data_initial, nwp_data, mnt_data

    def predict_maps(self, line_profile=None, memory_profile=None, **kwargs):
        """
        This function is used to select map predictions, the line profiled version or the memory profiled version
        """

        if line_profile:
            from line_profiler import LineProfiler
            lp = LineProfiler()
            lp_wrapper = lp(self._predict_maps)
            lp_wrapper(**kwargs)
            lp.print_stats()
        elif memory_profile:
            from memory_profiler import profile
            mp = profile(self._predict_maps)
            mp(**kwargs)
        else:
            array_xr = self._predict_maps(**kwargs)
            return array_xr
