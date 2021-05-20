# Packages: numpy, pandas, xarray, scipy, tensorflow, matplotlib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import rotate
from scipy import interpolate
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random  # Warning
import datetime
import time
import os
import concurrent.futures

# Local imports
from Utils import print_current_line, environment_GPU, change_dtype_if_required, assert_equal_shapes

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

from MidpointNormalize import MidpointNormalize


# Custom Metrics : NRMSE
def nrmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred))) / (K.max(y_pred) - K.min(y_pred))


# Custom Metrics : RMSE
def root_mse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


class Processing:
    n_rows, n_col = 79, 69
    _numexpr = _numexpr
    _geopandas = _geopandas
    _numba = _numba
    _dask = _dask
    _shapely_geometry = _shapely_geometry

    def __init__(self, obs=None, mnt=None, nwp=None, model_path=None, GPU=False, data_path=None):
        self.observation = obs
        self.mnt = mnt
        self.nwp = nwp
        self.model_path = model_path
        self.data_path = data_path
        environment_GPU(GPU=GPU)

    @staticmethod
    def rotate_scipy(topography, wind_dir):
        return (rotate(topography, wind_dir, reshape=False, mode='constant', cval=np.nan))

    def rotate_vectorize(self, topography, wind_dir):
        return (np.vectorize(self.rotate_scipy, signature='(m,n),()->(m,n)')(topography, wind_dir))

    def rotate_topography(self, topography, wind_dir, clockwise=False, verbose=True):
        """Rotate a topography to a specified angle

        If wind_dir = 270° then angle = 270+90 % 360 = 360 % 360 = 0
        For wind coming from the West, there is no rotation
        """
        if verbose: print('__Begin rotate topographies')

        if not (clockwise):
            rotated_topography = self.rotate_vectorize(topography, 90 + wind_dir)
        if clockwise:
            rotated_topography = self.rotate_vectorize(topography, -90 - wind_dir)

        if verbose: print('__End rotate topographies')
        return (rotated_topography)

    def _rotate_topo_for_all_station(self):
        """Not used
        Rotate the topography at all stations for each 1 degree angle of wind direction"""

        def rotate_topo_for_all_degrees(self, station):
            dict_topo[station]["rotated_topo_HD"] = {}
            MNT_data, _, _ = observation.extract_MNT_around_station(self, station, mnt, 400, 400)
            for angle in range(360):
                tile = self.rotate_topography(MNT_data, angle)
                dict_topo[station]["rotated_topo_HD"][str(angle)] = []
                dict_topo[station]["rotated_topo_HD"][str(angle)].append(tile)
            return (dict_topo)

        dict_topo = {}
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(rotate_topo_for_all_degrees, self.observation.stations['name'].values)
        except:
            print(
                "Parallel computation using concurrent.futures didn't work, so rotate_topo_for_all_degrees will not be parallelized.")
            map(rotate_topo_for_all_degrees, self.observation.stations['name'].values)
        self.dict_rot_topo = dict_topo

    @staticmethod
    def apply_log_profile(z_in=None, z_out=None, wind_in=None, z0=None,
                          verbose=True, z_in_verbose=None, z_out_verbose=None):

        """
        Apply log profile to a wind time serie.

        Parameters
        ----------
        z_in : Input elevation
        z_out : Output elevation
        wind_in : Input wind
        z0 : Roughness length for momentum

        Returns
        -------
        wind_out: Output wind, after logarithmic profile
        """

        if verbose: print(f"Applied log profile: {z_in_verbose} => {z_out_verbose}")

        if _numexpr:
            return (ne.evaluate("(wind_in * log(z_out / z0)) / log(z_in / z0)"))
        else:
            return ((wind_in * np.log(z_out / z0)) / np.log(z_in / z0))

    def exposed_wind_speed(self, wind_speed=None, z_out=None, z0=None, z0_rel=None, apply_directly_to_nwp=False):
        """
        Expose wind speed (deoperate subgrid parameterization from NWP)

        Parameters
        ----------
        wind_speed : ndarray
            Wind speed from NWP
        z_out : ndarray
            Elevation at which desinfluencing is performed
        z0 : ndarray
            Roughness length for momentum
        z0_rel : ndarray
            Roughness length for momentum associated with mean topography
        apply_directly_to_nwp : boolean, optional
            If True, updates the nwp directly by adding a new variable (Default: False)

        Returns
        -------
        exp_Wind: Unexposed wind
        acceleration_factor: Acceleration related to unexposition (usually >= 1)
        """

        if apply_directly_to_nwp:
            self.nwp.data_xr = self.nwp.data_xr.assign(
                exp_Wind=lambda x: x.Wind * (np.log(10 / x.Z0) / np.log(10 / x.Z0REL)) * (x.Z0 / x.Z0REL) ** (0.0708))

        if not (apply_directly_to_nwp):

            if self._numexpr:
                acceleration_factor = ne.evaluate(
                    "log((z_out) / z0) * (z0 / (z0_rel+z0))**0.0706 / (log((z_out) / (z0_rel+z0)))")
                acceleration_factor = ne.evaluate("where(acceleration_factor > 0, acceleration_factor, 1)")
                exp_Wind = ne.evaluate("wind_speed * acceleration_factor")

            else:
                acceleration_factor = np.log(z_out / z0) * (z0 / (z0_rel + z0)) ** 0.0706 / (
                    np.log((z_out) / (z0_rel + z0)))
                acceleration_factor = np.where(acceleration_factor > 0, acceleration_factor, 1)
                exp_Wind = wind_speed * acceleration_factor

            return (exp_Wind, acceleration_factor)

    def normalize_topo(self, topo_HD, mean, std, dtype=np.float32, librairie='num', verbose=True):
        """
        Normalize a topography with mean and std.

        Parameters
        ----------
        topo_HD : array
        mean : array
        std : array
        dtype: numpy dtype, optional
        tensorflow: boolean, optional
            Use tensorflow array (Default: False)

        Returns
        -------
        Standardized topography : array
        """
        if verbose: print(f"__Normalize done with mean {len(mean)} means and std")

        if librairie == 'tensorflow':
            topo_HD = tf.constant(topo_HD, dtype=tf.float32)
            mean = tf.constant(mean, dtype=tf.float32)
            std = tf.constant(std, dtype=tf.float32)
            result = tf.subtract(topo_HD, mean)
            result = tf.divide(result, result)
            return (result)

        if librairie == 'num':
            topo_HD = np.array(topo_HD, dtype=dtype)
            if self._numexpr:
                return (ne.evaluate("(topo_HD - mean) / std"))
            else:
                return ((topo_HD - mean) / std)

    def _select_nwp_time_serie_at_pixel(self, station_name, Z0=False, verbose=False, time=True, all=False):
        """
        Extract a time serie from the nwp, for several variables, at a station location.

        It selects the index of the nearest neighbor to the station in the nwp grid.

        Outputs are arrays.

        Parameters
        ----------
        station_name : string
            The considered station
        Z0 : boolean, optional
            If True, extracts info related to Z0 (Default: False')

        Returns
        -------wind_dir, wind_speed, time, Z0, Z0REL, ZS
        wind_dir : 1-D ndarray
            Wind direction [°].
        wind_speed : 1-D ndarray
            Wind speed [m/s].
        time : 1-D ndarray
            Time index.
        Z0 : 1-D ndarray
            Z0 [m].
        Z0REL : 1-D ndarray
            Z0REL [m].
        ZS : 1-D ndarray
            Altitude [m].
        """

        # Select station
        nwp_name = self.nwp.name
        stations = self.observation.stations
        y_idx_nwp, x_idx_nwp = stations[f"index_{nwp_name}_NN_0"][stations["name"] == station_name].values[0]
        y_idx_nwp, x_idx_nwp = np.int16(y_idx_nwp), np.int16(x_idx_nwp)

        # Select NWP data
        nwp_instance = self.nwp.data_xr

        if time or all:
            time_index = nwp_instance.time.data
            if time:
                return(time_index)

        wind_speed = nwp_instance.Wind.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
        wind_dir = nwp_instance.Wind_DIR.isel(xx=x_idx_nwp, yy=y_idx_nwp).data

        # Select Z0 informaiton
        if Z0 == True:
            Z0 = nwp_instance.Z0.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
            Z0REL = nwp_instance.Z0REL.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
            ZS = nwp_instance.ZS.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
        else:
            Z0 = []
            Z0REL = []
            ZS = []

        if verbose: print(f"Selected time series for pixel at station: {station_name}")
        if all:
            return (wind_dir, wind_speed, time_index, Z0, Z0REL, ZS)
        else:
            return (wind_dir, wind_speed, Z0, Z0REL, ZS)

    def _select_time_serie_from_array_xr(self, array_xr, station_name='Col du Lac Blanc', variable='UV', center=True):
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
        return (prediction, time)

    def load_model(self, dependencies=False):
        """
        Load a CNN, ans optionnaly its dependencies.

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
        return (mean, std)

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

        if verbose: print("NWP: Working on specific time window")

    @staticmethod
    def _scale_wind_for_ideal_case(wind_speed, wind_dir, input_speed, input_dir, verbose=True):
        """
        Specify specific wind speed and direction for ideal cases.

        Parameters
        ----------
        wind_speed : array
        wind_dir : array
        input_speed : constant
        input_dir : constant

        Returns
        -------
        wind_speed : array
            Constant wind speed.
        wind_direction : array
            Constant wind direction.
        """
        wind_speed = wind_speed * 0 + input_speed
        wind_dir = wind_dir * 0 + input_dir
        if verbose: print("__Wind speed scaled for ideal cases")
        return (wind_speed, wind_dir)

    def wind_speed_ratio(self, num=None, den=None):
        if self._numexpr:
            a1 = ne.evaluate("where(den > 0, num / den, 1)")
        else:
            a1 = np.where(den > 0, num / den, 1)
        return (a1)

    def _3D_wind_speed(self, U=None, V=None, W=None, out=None, verbose=True):
        """
        UVW = np.sqrt(U**2 + V**2 + W**2)

        Parameters
        ----------
        U : dnarray
            Horizontal wind speed component U
        V : string, optional
            Horizontal wind speed component V
        W : string, optional
            Vertical wind speed component W
        out : ndarray, optional
            If specified, the output of the caluclation is directly written in out,
            which is best for memory consumption (Default: None)

        Returns
        -------
        UVW : ndarray
            Wind speed
        """
        if out is None:
            if self._numexpr:
                wind_speed = ne.evaluate("sqrt(U**2 + V**2 + W**2)")
            else:
                wind_speed = np.sqrt(U ** 2 + V ** 2 + W ** 2)
            if verbose: print("__UVW calculated")
            return (wind_speed)

        else:
            if self._numexpr:
                ne.evaluate("sqrt(U**2 + V**2 + W**2)", out=out)
            else:
                np.sqrt(U ** 2 + V ** 2 + W ** 2, out=out)
            if verbose: print("__UVW calculated")

    def _2D_wind_speed(self, U=None, V=None, out=None, verbose=True):
        """
        UV = np.sqrt(U**2 + V**2)

        Parameters
        ----------
        U : dnarray
            Horizontal wind speed component U
        V : string, optional
            Horizontal wind speed component V
        out : ndarray, optional
            If specified, the output of the caluclation is directly written in out,
            which is best for memory consumption (Default: None)

        Returns
        -------
        UV : ndarray
            Wind speed
        """
        if out is None:
            if self._numexpr:
                wind_speed = ne.evaluate("sqrt(U**2 + V**2)")
            else:
                wind_speed = np.sqrt(U ** 2 + V ** 2)
            if verbose: print("__UV calculated")
            return (wind_speed)

        else:
            if self._numexpr:
                ne.evaluate("sqrt(U**2 + V**2)", out=out)
            else:
                np.sqrt(U ** 2 + V ** 2, out=out)
            if verbose: print("__UV calculated")

    def compute_wind_speed(self, U=None, V=None, W=None,
                           computing='num', out=None,
                           xarray_data=None, u_name="U", v_name="V", verbose=True):
        """
        Calculates wind speed from wind sped components.

        First detects the number of wind component then calculates wind speed.
        The calculation can be performed on nmexpr, numpy or xarray dataset

        Parameters
        ----------
        U : dnarray
            Horizontal wind speed component U
        V : string, optional
            Horizontal wind speed component V
        W : string, optional
            Vertical wind speed component V
        out : ndarray, optional
            If specified, the output of the calculation is directly written in out,
            which is best for memory consumption (Default: None)
        computing : str, optional
            Select the librairie to use for calculation. If 'num' first test numexpr, if not available select numpy.
            If 'xarray', a xarray dataframe needs to be specified with the corresponding names for wind components.
            (Default: 'num')

        Returns
        -------
        UV : ndarray
            Wind speed
        """
        # Numexpr or numpy
        if computing == 'num':
            if out is None:
                if W is None:
                    wind_speed = self._2D_wind_speed(U=U, V=V, verbose=verbose)
                else:
                    wind_speed = self._3D_wind_speed(U=U, V=V, W=W, verbose=verbose)
                wind_speed = change_dtype_if_required(wind_speed, np.float32)
                return (wind_speed)
            else:
                if W is None:
                    self._2D_wind_speed(U=U, V=V, out=out, verbose=verbose)
                else:
                    self._3D_wind_speed(U=U, V=V, W=W, out=out, verbose=verbose)

        if computing == 'xarray':
            xarray_data = xarray_data.assign(Wind=lambda x: np.sqrt(x[u_name] ** 2 + x[v_name] ** 2))
            xarray_data = xarray_data.assign(
                Wind_DIR=lambda x: np.mod(180 + np.rad2deg(np.arctan2(x[u_name], x[v_name])), 360))
            if verbose: print("__Wind and Wind_DIR calculated on xarray")
            return (xarray_data)

    def wind_speed_scaling(self, scaling_wind, prediction, linear=True, verbose=True):
        """
        scaling_wind * prediction / 3

        Parameters
        ----------
        scaling_wind : dnarray
            Scaling wind (ex: NWP wind)
        prediction : ndarray
            CNN ouptut
        linear : boolean, optionnal
            Linear scaling (Default: True)

        Returns
        -------
        prediction : ndarray
            Scaled wind
        """
        if linear:
            if self._numexpr:
                prediction = ne.evaluate("scaling_wind * prediction / 3")
            else:
                prediction = scaling_wind * prediction / 3
        prediction = change_dtype_if_required(prediction, np.float32)
        if verbose: print('__Wind speed scaling done')
        return (prediction)

    def angular_deviation(self, U, V, verbose=True):
        """
        Angular deviation from incoming flow.

        THe incoming flow is supposed from the West so that V=0. If deviated, V != 0.
        The angular deviation is then np.arctan(V / U)

        Parameters
        ----------
        U : dnarray
            Horizontal wind speed component U
        V : string, optional
            Horizontal wind speed component V

        Returns
        -------
        alpha : ndarray
            Angular deviation [rad]
        """
        if self._numexpr:
            alpha = ne.evaluate("where(U == 0, where(V == 0, 0, V/abs(V) * 3.14159 / 2), arctan(V / U))")
        else:
            alpha = np.where(U == 0,
                             np.where(V == 0, 0, np.sign(V) * np.pi / 2),
                             np.arctan(V / U))
        alpha = change_dtype_if_required(alpha, np.float32)
        if verbose: print("__Angular deviation calculated")
        return (alpha)

    def direction_from_alpha(self, wind_dir, alpha, input_dir_in_degree=True, verbose=True):
        """
        wind_dir - alpha

        Wind direction modified by angular deviation due to wind/topography interaction.
        Warning: this function might return a new wind direction in a rotated coordinates.

        Parameters
        ----------
        wind_dir : dnarray
            Initial NWP wind direction
        alpha : dnarray
            Angular deviation
        input_dir_in_degree : boolean, optionnal
            If True, converts the input wind direction from radans to degrees (Default: True)

        Returns
        -------
        alpha : ndarray
            Modified wind direction
        """
        if input_dir_in_degree:
            if self._numexpr:
                UV_DIR = ne.evaluate("(3.14159/180) * wind_dir - alpha")
            else:
                UV_DIR = (np.pi / 180) * wind_dir - alpha
        UV_DIR = change_dtype_if_required(UV_DIR, np.float32)
        if verbose: print("__Wind_DIR calculated from alpha")
        return (UV_DIR)

    def horizontal_wind_component(self, UV=None, UV_DIR=None,
                                  working_with_xarray=False, xarray_data=None, wind_name="Wind",
                                  wind_dir_name="Wind_DIR", verbose=True):
        """
        U = -np.sin(UV_DIR) * UV
        V = -np.cos(UV_DIR) * UV

        Computes U and V component from wind direction and wind speed

        Parameters
        ----------
        UV : dnarray
            Wind speed
        UV_DIR : dnarray
            Wind_direction
        working_with_xarray : boolean, optionnal
            If True, computes U and V on an xarray dataframe (provided with names of variables)
            and returns the dataframe

        Returns
        -------
        U : dnarray
            Horizontal wind speed component U
        V : string, optional
            Horizontal wind speed component V
        """
        if not (working_with_xarray):

            if self._numexpr:
                U = ne.evaluate("-sin(UV_DIR) * UV")
                V = ne.evaluate("-cos(UV_DIR) * UV")

            else:
                U = -np.sin(UV_DIR) * UV
                V = -np.cos(UV_DIR) * UV

            if verbose: print('__Horizontal wind component calculated')
            U = change_dtype_if_required(U, np.float32)
            V = change_dtype_if_required(V, np.float32)

            return (U, V)

        if working_with_xarray:
            xarray_data = xarray_data.assign(theta=lambda x: (np.pi / 180) * (x[wind_dir_name] % 360))
            xarray_data = xarray_data.assign(U=lambda x: -x[wind_name] * np.sin(x["theta"]))
            xarray_data = xarray_data.assign(V=lambda x: -x[wind_name] * np.cos(x["theta"]))
            if verbose: print('__Horizontal wind component calculated on xarray')
            return (xarray_data)

    def direction_from_u_and_v(self, U, V, output_in_degree=True, verbose=True):
        """
        UV_DIR = 180 + (180/3.14159)*arctan2(U,V)) % 360

        Compute wind direction from U and V components

        Parameters
        ----------
        U : dnarray
            Horizontal wind speed component U
        V : string, optional
            Horizontal wind speed component V
        output_in_degree : boolean, optionnal
            If True, the wind direction is converted to degrees

        Returns
        -------
        UV_DIR : dnarray
            Wind direction
        """
        if output_in_degree:
            if self._numexpr:
                UV_DIR = ne.evaluate("(180 + (180/3.14159)*arctan2(U,V)) % 360")
            else:
                UV_DIR = np.mod(180 + np.rad2deg(np.arctan2(U, V)), 360)
            if verbose: print('__Wind_DIR calculated from U and V')
            UV_DIR = change_dtype_if_required(UV_DIR, np.float32)
            return (UV_DIR)

    @staticmethod
    def reshape_list_array(list_array=None, shape=None):
        """
        Utility function that takes as input a list of arrays to reshape to the same shape

        Parameters
        ----------
        list_array : list
            List of arrays
        shape : tuple
            typle of shape

        Returns
        -------
        result : list
            List of reshaped arrays
        """
        result = []
        for array in list_array:
            result.append(np.reshape(array, shape))
        return (result)

    @staticmethod
    def several_empty_like(array_like, nb_empty_arrays=None):
        result = []
        for array in range(nb_empty_arrays):
            result.append(np.empty_like(array_like))
        return (result)

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
            peak_valley_height = np.empty((nb_station), dtype=np.float32)
            mean_height = np.empty((nb_station), dtype=np.float32)
            all_topo_HD = np.empty((nb_station, self.n_rows, self.n_col), dtype=np.uint16)
            all_topo_x_small_l93 = np.empty((nb_station, self.n_col), dtype=np.float32)
            all_topo_y_small_l93 = np.empty((nb_station, self.n_rows), dtype=np.float32)
            ten_m_array = 10 * np.ones((nb_station, nb_sim), dtype=np.float32)
            three_m_array = 3 * np.ones((nb_station, nb_sim), dtype=np.float32)
            list_arrays_1 = [topo, wind_speed_all, wind_dir_all, Z0_all, Z0REL_all, ZS_all, peak_valley_height, mean_height]
            list_arrays_2 = [all_topo_HD, all_topo_x_small_l93, all_topo_y_small_l93, ten_m_array, three_m_array]
            list_return = list_arrays_1 + list_arrays_2
        return(list_return)

    def predict_at_stations(self, stations_name, fast=False, verbose=True, plot=False, Z0_cond=True, peak_valley=True,
                            log_profile_to_h_2=False, log_profile_from_h_2=False, log_profile_10m_to_3m=False,
                            ideal_case=False, input_speed=3, input_dir=270, line_profile=False):
        """
        This function is used to select predictions at stations or the lin profiled version
        """
        if line_profile:
            from line_profiler import LineProfiler
            lp = LineProfiler()
            lp_wrapper = lp(self._predict_at_stations)
            lp_wrapper(stations_name, fast=fast, verbose=verbose, plot=plot, Z0_cond=Z0_cond, peak_valley=peak_valley,
                       log_profile_to_h_2=log_profile_to_h_2, log_profile_from_h_2=log_profile_from_h_2,
                       log_profile_10m_to_3m=log_profile_10m_to_3m, ideal_case=ideal_case,
                       input_speed=input_speed, input_dir=input_dir)
            lp.print_stats()
        else:
            array_xr = self._predict_at_stations(stations_name, fast=fast, verbose=verbose, plot=plot, Z0_cond=Z0_cond,
                                      peak_valley=peak_valley,log_profile_to_h_2=log_profile_to_h_2,
                                      log_profile_from_h_2=log_profile_from_h_2,
                                      log_profile_10m_to_3m=log_profile_10m_to_3m, ideal_case=ideal_case,
                                      input_speed=input_speed, input_dir=input_dir)
            return(array_xr)

    def select_height_for_exposed_wind_speed(self, height=None, zs=None, peak_valley=None, nb_station=None):
        if peak_valley:
            return(height)
        else:
            return(zs)

    def get_closer_from_learning_conditions(self, topo, mean_height, std, P95=530, P05=-527, axis=(1,2)):
        max_alt_deviation = np.nanmax(topo.squeeze() - mean_height, axis=axis)
        min_alt_deviation = np.nanmin(topo.squeeze() - mean_height, axis=axis)

        alpha_max = np.abs(max_alt_deviation / P95)
        alpha_min = np.abs(min_alt_deviation / P05)
        alpha_i = np.nanmax((alpha_max, alpha_min))

        alpha = np.where(max_alt_deviation>P95,
                         np.where(min_alt_deviation<P05, alpha_i, alpha_max),
                         np.where(min_alt_deviation<P05, alpha_min, 1))

        return(alpha*std)


    def _predict_at_stations(self, stations_name, fast=False, verbose=True, plot=False, Z0_cond=True, peak_valley=True,
                            log_profile_to_h_2=False, log_profile_from_h_2=False, log_profile_10m_to_3m=False,
                            ideal_case=False, input_speed=3, input_dir=270):
        """
        Wind downscaling operated at observation stations sites only.

        17 min on CPU
        3.7 minutes on GPU

        Parameters
        ----------
        stations_name : list of strings
            List containing station names
        Z0_cond : boolean
            To expose wind speed
        peak_valley : boolean
            Use mean peak valley height to expose wind. An other option is to use mean height.
        log_profile_ : boolean(s)
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
                             Z0_cond=True,
                             peak_valley=True,
                             ideal_case=False)
        """

        # Select timeframe
        self._select_timeframe_nwp(ideal_case=ideal_case, verbose=True)

        # Simulation parameters
        time_xr = self._select_nwp_time_serie_at_pixel(random.choice(stations_name), time=True)
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
            Z0REL_all[idx_station, :], ZS_all[idx_station, :] = self._select_nwp_time_serie_at_pixel(single_station,
                                                                                                Z0=Z0_cond, time=False)
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
            topo[idx_station, :, :, :, 0] = self.rotate_topography(topo_HD, wind_dir_all[idx_station, :], clockwise=False)[:, y_offset_left:y_offset_right, x_offset_left:x_offset_right]

            # Store results
            all_topo_HD[idx_station, :, :] = topo_HD[y_offset_left:y_offset_right, x_offset_left:x_offset_right]
            all_topo_x_small_l93[idx_station, :] = topo_x_l93[x_offset_left:x_offset_right]
            all_topo_y_small_l93[idx_station, :] = topo_y_l93[y_offset_left:y_offset_right]
            peak_valley_height[idx_station] = np.int32(2 * np.nanstd(all_topo_HD[idx_station, :, :]))
            mean_height[idx_station] = np.int32(np.nanmean(all_topo_HD[idx_station, :, :]))

        # Exposed wind
        if Z0_cond:
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
        mean_height = mean_height.reshape((nb_station, 1, 1))
        std = self.get_closer_from_learning_conditions(topo, mean_height, std)
        mean_height = mean_height.reshape((nb_station, 1, 1, 1, 1))
        std = std.reshape((nb_station, nb_sim, 1, 1, 1))
        topo = self.normalize_topo(topo, mean_height, std)

        # Reshape for tensorflow
        topo = topo.reshape((nb_station * nb_sim, self.n_rows, self.n_col, 1))
        if verbose: print('__Reshaped tensorflow done')

        """
        Warning: change dependencies here
        """
        # Load model
        self.load_model(dependencies=True)

        # Predictions
        prediction = self.model.predict(topo)
        if verbose: print('__Prediction done')

        # Acceleration NWP to CNN
        UVW_int = self.compute_wind_speed(U=prediction[:, :, :, 0], V=prediction[:, :, :, 1], W=prediction[:, :, :, 2])
        acceleration_CNN = self.wind_speed_ratio(num=UVW_int, den=3 * np.ones(UVW_int.shape))

        # Reshape predictions for analysis
        prediction = prediction.reshape((nb_station, nb_sim, self.n_rows, self.n_col, 3))
        if verbose: print(f"__Prediction reshaped: {prediction.shape}")

        # Reshape for broadcasting
        wind_speed_all = wind_speed_all.reshape((nb_station, nb_sim, 1, 1, 1))
        if Z0_cond:
            exp_Wind, acceleration_factor, ten_m_array, three_m_array, Z0_all = self.reshape_list_array(
                list_array=[exp_Wind, acceleration_factor, ten_m_array, three_m_array, Z0_all],
                shape=(nb_station, nb_sim, 1, 1, 1))
            peak_valley_height = peak_valley_height.reshape((nb_station, 1, 1, 1, 1))
        wind_dir_all = wind_dir_all.reshape((nb_station, nb_sim, 1, 1))

        # Wind speed scaling
        scaling_wind = exp_Wind if Z0_cond else wind_speed_all
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
        if verbose: print('__Start rotating to initial position')
        prediction = np.moveaxis(prediction, -1, 2)
        wind_dir_all = wind_dir_all.reshape((nb_station, nb_sim, 1))
        prediction = self.rotate_topography(prediction[:, :, :, :, :], wind_dir_all[:, :, :], clockwise=True, verbose=False)
        prediction = np.moveaxis(prediction, 2, -1)

        U = prediction[:, :, :, :, 0].view()
        V = prediction[:, :, :, :, 1].view()
        W = prediction[:, :, :, :, 2].view()

        if verbose: print('__Wind prediction rotated for initial topography')

        # Compute wind direction
        UV_DIR = self.direction_from_u_and_v(U, V)  # Good axis and pixel [degree]

        # UVW
        UVW = self.compute_wind_speed(U=U, V=V, W=W)

        # Acceleration NWP to CNN
        acceleration_all = self.wind_speed_ratio(num=UVW, den=wind1.reshape(
            (nb_station, nb_sim, 1, 1))) if Z0_cond else UVW * np.nan

        # Reshape after broadcasting
        wind_speed_all, wind_dir_all, Z0_all = self.reshape_list_array(list_array=[wind_speed_all, wind_dir_all, Z0_all],
                                                               shape=(nb_station, nb_sim))
        if Z0_cond:
            exp_Wind, acceleration_factor, a1, a2, a3 = self.reshape_list_array(
                list_array=[exp_Wind, acceleration_factor, a1, a2, a3],
                shape=(nb_station, nb_sim))
            a4, acceleration_CNN = self.reshape_list_array(list_array=[np.max(a4, axis=4), acceleration_CNN],
                                                           shape=(nb_station, nb_sim, self.n_rows, self.n_col))
            peak_valley_height = peak_valley_height.reshape((nb_station))

        # Verification of shapes
        assert_equal_shapes([U,V,W,UV_DIR], (nb_station, nb_sim, self.n_rows, self.n_col))
        assert_equal_shapes([wind_speed_all,wind_dir_all], (nb_station, nb_sim))

        if verbose: print('__Reshape final predictions done')

        # Store results
        if verbose: print('__Start creating array')
        array_xr = xr.Dataset(data_vars={"U": (["station", "time", "y", "x"], U),
                                         "V": (["station", "time", "y", "x"], V),
                                         "W": (["station", "time", "y", "x"], W),
                                         "UV": (["station", "time", "y", "x"], np.sqrt(U ** 2 + V ** 2)),
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
                                             exp_Wind if Z0_cond else np.zeros((nb_station, nb_sim)),),
                                         "acceleration_factor": (
                                             ["station", "time"],
                                             acceleration_factor if Z0_cond else np.zeros((nb_station, nb_sim)),),
                                         "a1": (
                                             ["station", "time"],
                                             a1 if Z0_cond else np.zeros((nb_station, nb_sim)),),
                                         "a2": (
                                             ["station", "time"],
                                             a2 if Z0_cond else np.zeros((nb_station, nb_sim)),),
                                         "a3": (
                                             ["station", "time"],
                                             a3 if Z0_cond else np.zeros((nb_station, nb_sim)),),
                                         "a4": (
                                             ["station", "time", "y", "x"],
                                             a4 if Z0_cond else np.zeros(
                                                 (nb_station, nb_sim, self.n_rows, self.n_col)),),
                                         "acceleration_all": (
                                             ["station", "time", "y", "x"],
                                             acceleration_all if Z0_cond else np.zeros(
                                                 (nb_station, nb_sim, self.n_rows, self.n_col)),),
                                         "acceleration_CNN": (
                                             ["station", "time", "y", "x"],
                                             acceleration_CNN if Z0_cond else np.zeros(
                                                 (nb_station, nb_sim, self.n_rows, self.n_col)),),
                                         "Z0": (
                                             ["station", "time"],
                                             Z0_all if Z0_cond else np.zeros((nb_station, nb_sim)),),
                                         "Z0REL": (["station", "time"],
                                                   Z0REL_all if Z0_cond else np.zeros((nb_station, nb_sim)),),
                                         },

                              coords={"station": np.array(stations_name),
                                      "time": np.array(time_xr),
                                      "x": np.array(list(range(self.n_col))),
                                      "y": np.array(list(range(self.n_rows)))})
        if verbose: print('__Creating array done')

        return (array_xr)

    def _select_large_domain_around_station(self, station_name, dx, dy, type="NWP", additionnal_dx_mnt=None):
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
        type: str
            "NWP" or "MNT"
        additionnal_dx_mnt: float
            Additional length to the side of the domain (Default: True)

            Usually used when extracting MNT data. As CNN predictions are performed on maps but the information is only
            stored on specific pixels near the center, we require more MNT data than the original domain size.


        Returns
        -------
        result : list
            Xarray DataFrame on the specified domain
        """

        stations = self.observation.stations
        if type == "NWP":
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

        elif type == "MNT":
            # MNT domain must be larger than NWP domain as we extract MNT data around NWP data
            if additionnal_dx_mnt is not None:
                dx = dx + additionnal_dx_mnt
                dy = dy + additionnal_dx_mnt
            mnt_name = self.mnt.name
            mnt_x, mnt_y = stations[f"{mnt_name}_NN_0_cKDTree"][stations["name"] == station_name].values[0]

            x_min = mnt_x - dx
            x_max = mnt_x + dx
            y_min = mnt_y - dy
            y_max = mnt_y + dy

            data_xr = self.mnt.data_xr
            mask = (x_min <= data_xr.x) & (data_xr.x <= x_max) & (y_min <= data_xr.y) & (data_xr.y <= y_max)
            data_xr = data_xr.where(mask, drop=True)

        return (data_xr)

    def _get_mnt_data_and_shape(self, mnt_data):
        """
        This function takes as input a mnt and returns data, coordinates and shape
        """

        if self._dask:
            shape_x_mnt, shape_y_mnt = mnt_data.data.shape[1:]
            mnt_data_x = mnt_data.x.data
            mnt_data_y = mnt_data.y.data
            mnt_data = mnt_data.data
        else:
            mnt_data_x = mnt_data.x.data
            mnt_data_y = mnt_data.y.data
            mnt_data = mnt_data.__xarray_dataarray_variable__.data
            shape_x_mnt, shape_y_mnt = mnt_data[0, :, :].shape
        return (mnt_data, mnt_data_x, mnt_data_y, shape_x_mnt, shape_y_mnt)

    @staticmethod
    def interpolate_xarray_grid(xarray_data=None, interp=None, name_x='xx', name_y='yy', method='linear'):
        """
        Interpolate a regular grid on an xarray dataframe using multilinear interpolation.
        The interp parameters control the upsampling.


        Parameters
        ----------
        xarray_data : xarray dataframe
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
        return (xarray_data)

    def predict_map(self, station_name='Col du Lac Blanc', x_0=None, y_0=None, dx=10_000, dy=10_000, interp=3,
                    year_0=None, month_0=None, day_0=None, hour_0=None,
                    year_1=None, month_1=None, day_1=None, hour_1=None,
                    Z0_cond=False, verbose=True, peak_valley=True):

        # Select NWP data
        if verbose: print("Selecting NWP")
        nwp_data = self._select_large_domain_around_station(station_name, dx, dy, type="NWP", additionnal_dx_mnt=None)
        begin = datetime.datetime(year_0, month_0, day_0, hour_0)
        end = datetime.datetime(year_1, month_1, day_1, hour_1)
        nwp_data = nwp_data.sel(time=slice(begin, end))
        nwp_data_initial = nwp_data

        # Calculate U_nwp and V_nwp
        if verbose: print("U_nwp and V_nwp computation")
        nwp_data = nwp_data.assign(theta=lambda x: (np.pi / 180) * (x["Wind_DIR"] % 360))
        nwp_data = nwp_data.assign(U=lambda x: -x["Wind"] * np.sin(x["theta"]))
        nwp_data = nwp_data.assign(V=lambda x: -x["Wind"] * np.cos(x["theta"]))
        nwp_data = nwp_data.drop_vars(["Wind", "Wind_DIR"])

        # Interpolate AROME
        if verbose: print("AROME interpolation")
        new_x = np.linspace(nwp_data["xx"].min().data, nwp_data["xx"].max().data, nwp_data.dims["xx"] * interp)
        new_y = np.linspace(nwp_data["yy"].min().data, nwp_data["yy"].max().data, nwp_data.dims["yy"] * interp)
        nwp_data = nwp_data.interp(xx=new_x, yy=new_y, method='linear')
        nwp_data = nwp_data.assign(Wind=lambda x: np.sqrt(x["U"] ** 2 + x["V"] ** 2))
        nwp_data = nwp_data.assign(Wind_DIR=lambda x: np.mod(180 + np.rad2deg(np.arctan2(x["U"], x["V"])), 360))

        # Time scale and domain length
        times = nwp_data.time.data
        nwp_x_l93 = nwp_data.X_L93
        nwp_y_l93 = nwp_data.Y_L93
        nb_time_step = len(times)
        nb_px_nwp_y, nb_px_nwp_x = nwp_x_l93.shape

        # Select MNT data
        if verbose: print("Selecting NWP")
        mnt_data = self._select_large_domain_around_station(station_name, dx, dy, type="MNT", additionnal_dx_mnt=2_000)
        xmin_mnt = np.nanmin(mnt_data.x.data)
        ymax_mnt = np.nanmax(mnt_data.y.data)

        # NWP forcing data
        if verbose: print("Selecting forcing data")
        wind_speed_nwp = nwp_data["Wind"].data
        wind_DIR_nwp = nwp_data["Wind_DIR"].data
        if Z0_cond:
            Z0_nwp = nwp_data["Z0"].data
            Z0REL_nwp = nwp_data["Z0REL"].data
            ZS_nwp = nwp_data["ZS"].data

        # Weight
        x, y = np.meshgrid(np.linspace(-1, 1, self.n_col), np.linspace(-1, 1, self.n_rows))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 0.025, 0
        gaussian_weight = 0.5 + 50 * np.array(np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2))))

        # Initialize wind map
        if _dask:
            shape_x_mnt, shape_y_mnt = mnt_data.data.shape[1:]
            mnt_data_x = mnt_data.x.data
            mnt_data_y = mnt_data.y.data
            mnt_data = mnt_data.data
        else:
            mnt_data_x = mnt_data.x.data
            mnt_data_y = mnt_data.y.data
            mnt_data = mnt_data.__xarray_dataarray_variable__.data
            shape_x_mnt, shape_y_mnt = mnt_data[0, :, :].shape

        wind_map = np.zeros((nb_time_step, shape_x_mnt, shape_y_mnt, 3), dtype=np.float32)
        weights = np.zeros((nb_time_step, shape_x_mnt, shape_y_mnt), dtype=np.float32)
        peak_valley_height = np.empty((nb_px_nwp_y, nb_px_nwp_x), dtype=np.float32)
        topo_concat = np.empty((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79, 69), dtype=np.float32)

        # Concatenate topographies along single axis
        if verbose: print("Concatenate topographies along single axis")
        nb_pixel = 70
        for time_step, time in enumerate(times):
            for idx_y_nwp in range(nb_px_nwp_y):
                for idx_x_nwp in range(nb_px_nwp_x):
                    # Select index NWP
                    x_nwp_L93 = nwp_x_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data
                    y_nwp_L93 = nwp_y_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data

                    # Select indexes MNT
                    idx_x_mnt = int((x_nwp_L93 - xmin_mnt) // self.mnt.resolution_x)
                    idx_y_mnt = int((ymax_mnt - y_nwp_L93) // self.mnt.resolution_y)

                    # Large topo
                    topo_i = mnt_data[0, idx_y_mnt - nb_pixel:idx_y_mnt + nb_pixel,
                             idx_x_mnt - nb_pixel:idx_x_mnt + nb_pixel]
                    # mnt_x_i = mnt_data.x.data[idx_x_mnt - nb_pixel:idx_x_mnt + nb_pixel]
                    # mnt_y_i = mnt_data.y.data[idx_y_mnt - nb_pixel:idx_y_mnt + nb_pixel]

                    # Mean peak_valley altitude
                    if time_step == 0:
                        peak_valley_height[idx_y_nwp, idx_x_nwp] = np.int32(2 * np.nanstd(topo_i))

                    # Wind direction
                    wind_DIR = wind_DIR_nwp[time_step, idx_y_nwp, idx_x_nwp]

                    # Rotate topography
                    topo_i = self.rotate_topography(topo_i, wind_DIR)
                    topo_i = topo_i[nb_pixel - 39:nb_pixel + 40, nb_pixel - 34:nb_pixel + 35]

                    # Store result
                    topo_concat[time_step, idx_y_nwp, idx_x_nwp, :, :] = topo_i

        # Reshape for tensorflow
        topo_concat = topo_concat.reshape((nb_time_step * nb_px_nwp_x * nb_px_nwp_y, self.n_rows, self.n_col, 1))

        # Normalize
        if verbose: print("Topographies normalization")
        mean, std = self._load_norm_prm()
        topo_concat = self.normalize_topo(topo_concat, mean, std).astype(dtype=np.float32, copy=False)

        # Load model
        self.load_model(dependencies=True)

        # Predictions
        if verbose: print("Predictions")
        prediction = self.model.predict(topo_concat)

        # Reshape predictions for analysis
        prediction = prediction.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, self.n_rows, self.n_col, 3)).astype(
            np.float32, copy=False)

        # Wind speed scaling for broadcasting
        wind_speed_nwp = wind_speed_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1)).astype(np.float32,
                                                                                                          copy=False)
        wind_DIR_nwp = wind_DIR_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1)).astype(np.float32,
                                                                                                   copy=False)

        # Exposed wind speed
        if verbose: print("Exposed wind speed")
        if Z0_cond:

            Z0_nwp = Z0_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            Z0REL_nwp = Z0REL_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            ZS_nwp = ZS_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            peak_valley_height = peak_valley_height.reshape((1, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))

            # Choose height in the formula
            if peak_valley:
                height = peak_valley_height
            else:
                height = ZS_nwp

            if self._numexpr:
                acceleration_factor = ne.evaluate(
                    "log((height/2) / Z0_nwp) * (Z0_nwp / (Z0REL_nwp+Z0_nwp))**0.0706 / (log((height/2) / (Z0REL_nwp+Z0_nwp)))")
                acceleration_factor = ne.evaluate("where(acceleration_factor > 0, acceleration_factor, 1)")
                exp_Wind = ne.evaluate("wind_speed_all * acceleration_factor")
            else:
                acceleration_factor = np.log((height / 2) / Z0_nwp) * (Z0_nwp / (Z0REL_nwp + Z0_nwp)) ** 0.0706 / (
                    np.log((height / 2) / (Z0REL_nwp + Z0_nwp)))
                acceleration_factor = np.where(acceleration_factor > 0, acceleration_factor, 1)
                exp_Wind = wind_speed_all * acceleration_factor

        # Wind speed scaling
        if verbose: print("Wind speed scaling")
        scaling_wind = exp_Wind if Z0_cond else wind_speed_nwp
        if self._numexpr:
            prediction = ne.evaluate("scaling_wind * prediction / 3")
        else:
            prediction = scaling_wind * prediction / 3

        # Wind computations
        if verbose: print("Wind computations")
        U_old = prediction[:, :, :, :, :, 0]  # Expressed in the rotated coord. system [m/s]
        V_old = prediction[:, :, :, :, :, 1]  # Expressed in the rotated coord. system [m/s]
        W_old = prediction[:, :, :, :, :, 2]  # Good coord. but not on the right pixel [m/s]

        if self._numexpr:
            UV = ne.evaluate("sqrt(U_old**2 + V_old**2)")  # Good coord. but not on the right pixel [m/s]
            alpha = ne.evaluate(
                "where(U_old == 0, where(V_old == 0, 0, V_old/abs(V_old) * 3.14159 / 2), arctan(V_old / U_old))")
            UV_DIR = ne.evaluate(
                "(3.14159/180) * wind_DIR_nwp - alpha")  # Good coord. but not on the right pixel [radian]
        else:
            UV = np.sqrt(U_old ** 2 + V_old ** 2)  # Good coord. but not on the right pixel [m/s]
            alpha = np.where(U_old == 0,
                             np.where(V_old == 0, 0, np.sign(V_old) * np.pi / 2),
                             np.arctan(V_old / U_old))  # Expressed in the rotated coord. system [radian]
            UV_DIR = (np.pi / 180) * wind_DIR_nwp - alpha  # Good coord. but not on the right pixel [radian]

        # float64 to float32
        UV_DIR = UV_DIR.astype(dtype=np.float32, copy=False)

        # Reshape wind speed and wind direction
        wind_speed_nwp = wind_speed_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x))
        wind_DIR_nwp = wind_DIR_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x))

        # Calculate U and V along initial axis
        if self._numexpr:
            U_old = ne.evaluate("-sin(UV_DIR) * UV")
            V_old = ne.evaluate("-cos(UV_DIR) * UV")
        else:
            U_old = -np.sin(UV_DIR) * UV  # Good coord. but not on the right pixel [m/s]
            V_old = -np.cos(UV_DIR) * UV  # Good coord. but not on the right pixel [m/s]

        # Final results
        print("Creating wind map")

        # Good axis and pixel location [m/s]
        for time_step, time in enumerate(times):
            for idx_y_nwp in range(nb_px_nwp_y):
                for idx_x_nwp in range(nb_px_nwp_x):
                    wind_DIR = wind_DIR_nwp[time_step, idx_y_nwp, idx_x_nwp]
                    U = self.rotate_topography(
                        U_old[time_step, idx_y_nwp, idx_x_nwp, :, :],
                        wind_DIR,
                        clockwise=True)
                    V = self.rotate_topography(
                        V_old[time_step, idx_y_nwp, idx_x_nwp, :, :],
                        wind_DIR,
                        clockwise=True)
                    W = self.rotate_topography(
                        W_old[time_step, idx_y_nwp, idx_x_nwp, :, :],
                        wind_DIR,
                        clockwise=True)

                    # Select index NWP
                    x_nwp_L93 = nwp_x_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data
                    y_nwp_L93 = nwp_y_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data

                    # Select indexes MNT
                    idx_x_mnt = int((x_nwp_L93 - xmin_mnt) // self.mnt.resolution_x)
                    idx_y_mnt = int((ymax_mnt - y_nwp_L93) // self.mnt.resolution_y)

                    # Select center of the predictions
                    wind_map[time_step, idx_y_mnt - 8:idx_y_mnt + 9, idx_x_mnt - 8:idx_x_mnt + 9, 0] = U[39 - 8:40 + 8,
                                                                                                       34 - 8:35 + 8]

                    wind_map[time_step, idx_y_mnt - 8:idx_y_mnt + 9, idx_x_mnt - 8:idx_x_mnt + 9, 1] = V[39 - 8:40 + 8,
                                                                                                       34 - 8:35 + 8]

                    wind_map[time_step, idx_y_mnt - 8:idx_y_mnt + 9, idx_x_mnt - 8:idx_x_mnt + 9, 2] = W[39 - 8:40 + 8,
                                                                                                       34 - 8:35 + 8]

        # wind_map = wind_map / weights.reshape((nb_time_step, shape_x_mnt, shape_y_mnt, 1))
        return (wind_map, weights, nwp_data_initial, nwp_data, mnt_data)

    def predict_map_tensorflow(self, station_name='Col du Lac Blanc', x_0=None, y_0=None, dx=10_000, dy=10_000,
                               interp=3,
                               year_0=None, month_0=None, day_0=None, hour_0=None,
                               year_1=None, month_1=None, day_1=None, hour_1=None,
                               Z0_cond=False, verbose=True, peak_valley=True):

        # Select NWP data
        if verbose: print("Selecting NWP")
        nwp_data = self._select_large_domain_around_station(station_name, dx, dy, type="NWP", additionnal_dx_mnt=None)
        begin = datetime.datetime(year_0, month_0, day_0, hour_0)  # datetime
        end = datetime.datetime(year_1, month_1, day_1, hour_1)  # datetime
        nwp_data = nwp_data.sel(time=slice(begin, end))
        nwp_data_initial = nwp_data

        # Calculate U_nwp and V_nwp
        if verbose: print("U_nwp and V_nwp computation")
        nwp_data = nwp_data.assign(theta=lambda x: (np.pi / 180) * (x["Wind_DIR"] % 360))
        nwp_data = nwp_data.assign(U=lambda x: -x["Wind"] * np.sin(x["theta"]))
        nwp_data = nwp_data.assign(V=lambda x: -x["Wind"] * np.cos(x["theta"]))
        nwp_data = nwp_data.drop_vars(["Wind", "Wind_DIR"])

        # Interpolate AROME
        if verbose: print("AROME interpolation")
        new_x = np.linspace(nwp_data["xx"].min().data, nwp_data["xx"].max().data, nwp_data.dims["xx"] * interp)
        new_y = np.linspace(nwp_data["yy"].min().data, nwp_data["yy"].max().data, nwp_data.dims["yy"] * interp)
        nwp_data = nwp_data.interp(xx=new_x, yy=new_y, method='linear')
        nwp_data = nwp_data.assign(Wind=lambda x: np.sqrt(x["U"] ** 2 + x["V"] ** 2))
        nwp_data = nwp_data.assign(Wind_DIR=lambda x: np.mod(180 + np.rad2deg(np.arctan2(x["U"], x["V"])), 360))

        # Time scale and domain length
        times = nwp_data.time.data
        nwp_x_l93 = nwp_data.X_L93
        nwp_y_l93 = nwp_data.Y_L93
        nb_time_step = len(times)
        nb_px_nwp_y, nb_px_nwp_x = nwp_x_l93.shape

        # Select MNT data
        if verbose: print("Selecting NWP")
        mnt_data = self._select_large_domain_around_station(station_name, dx, dy, type="MNT", additionnal_dx_mnt=2_000)
        xmin_mnt = np.nanmin(mnt_data.x.data)
        ymax_mnt = np.nanmax(mnt_data.y.data)
        if _dask:
            shape_x_mnt, shape_y_mnt = mnt_data.data.shape[1:]
            mnt_data_x = mnt_data.x.data
            mnt_data_y = mnt_data.y.data
            mnt_data = tf.constant(mnt_data.data, dtype=tf.float32)
        else:
            mnt_data_x = mnt_data.x.data
            mnt_data_y = mnt_data.y.data
            mnt_data = tf.constant(mnt_data.__xarray_dataarray_variable__.data, dtype=tf.float32)
            shape_x_mnt, shape_y_mnt = mnt_data[0, :, :].shape

        # NWP forcing data
        if verbose: print("Selecting forcing data")
        wind_speed_nwp = tf.constant(nwp_data["Wind"].data, dtype=tf.float32)
        wind_DIR_nwp = tf.constant(nwp_data["Wind_DIR"].data, dtype=tf.float32)
        if Z0_cond:
            Z0_nwp = tf.constant(nwp_data["Z0"].data, dtype=tf.float32)
            Z0REL_nwp = tf.constant(nwp_data["Z0REL"].data, dtype=tf.float32)
            ZS_nwp = tf.constant(nwp_data["ZS"].data, dtype=tf.float32)

        # Initialize wind map
        wind_map = np.empty((nb_time_step, shape_x_mnt, shape_y_mnt, 3))

        # Concatenate topographies along single axis
        if verbose: print("Concatenate topographies along single axis")
        topo_concat = np.empty((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79, 69), dtype=np.float32)
        peak_valley_height = np.empty((nb_px_nwp_y, nb_px_nwp_x), dtype=np.float32)

        nb_pixel = 70
        for time_step, time in enumerate(times):
            for idx_y_nwp in range(nb_px_nwp_y):
                for idx_x_nwp in range(nb_px_nwp_x):
                    # Select index NWP
                    x_nwp_L93 = nwp_x_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data
                    y_nwp_L93 = nwp_y_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data

                    # Select indexes MNT
                    idx_x_mnt = int((x_nwp_L93 - xmin_mnt) // self.mnt.resolution_x)
                    idx_y_mnt = int((ymax_mnt - y_nwp_L93) // self.mnt.resolution_y)

                    # Large topo
                    topo_i = mnt_data[0, idx_y_mnt - nb_pixel:idx_y_mnt + nb_pixel,
                             idx_x_mnt - nb_pixel:idx_x_mnt + nb_pixel]
                    # mnt_x_i = mnt_data.x.data[idx_x_mnt - nb_pixel:idx_x_mnt + nb_pixel]
                    # mnt_y_i = mnt_data.y.data[idx_y_mnt - nb_pixel:idx_y_mnt + nb_pixel]

                    # Mean peak_valley altitude
                    if time_step == 0:
                        peak_valley_height[idx_y_nwp, idx_x_nwp] = np.int32(2 * np.nanstd(topo_i))

                    # Wind direction
                    wind_DIR = wind_DIR_nwp[time_step, idx_y_nwp, idx_x_nwp]

                    # Rotate topography
                    topo_i = self.rotate_topography(topo_i.numpy(), wind_DIR.numpy())
                    topo_i = topo_i[nb_pixel - 39:nb_pixel + 40, nb_pixel - 34:nb_pixel + 35]

                    # Store result
                    topo_concat[time_step, idx_y_nwp, idx_x_nwp, :, :] = topo_i

        with tf.device('/GPU:0'):
            # Reshape for tensorflow
            topo_concat = tf.constant(topo_concat, dtype=tf.float32)
            topo_concat = tf.reshape(topo_concat,
                                     [nb_time_step * nb_px_nwp_x * nb_px_nwp_y, self.n_rows, self.n_col, 1])

            # Normalize
            if verbose: print("Topographies normalization")
            mean, std = self._load_norm_prm()
            topo_concat = self.normalize_topo(topo_concat, mean, std, librairie='tensorflow')

            # Load model
            self.load_model(dependencies=True)

            # Predictions
            if verbose: print("Predictions")
            prediction = self.model.predict(topo_concat)
            del topo_concat
            # Reshape predictions for analysis
            prediction = tf.reshape(prediction, [nb_time_step, nb_px_nwp_y, nb_px_nwp_x, self.n_rows, self.n_col, 3])

            # Wind speed scaling for broadcasting
            wind_speed_nwp = tf.reshape(wind_speed_nwp, [nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1])
            wind_DIR_nwp = tf.reshape(wind_DIR_nwp, [nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1])

            # Exposed wind speed
            if verbose: print("Exposed wind speed")
            if Z0_cond:
                Z0_nwp = tf.reshape(Z0_nwp, [nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1])
                Z0REL_nwp = tf.reshape(Z0REL_nwp, [nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1])
                peak_valley_height = tf.reshape(peak_valley_height, [1, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1])

                # Choose height in the formula
                if peak_valley:
                    height = peak_valley_height
                else:
                    height = ZS_nwp

                acceleration_factor = tf.math.log((height / 2) / Z0_nwp) * (Z0_nwp / (Z0REL_nwp + Z0_nwp)) ** 0.0706 / (
                    np.log((height / 2) / (Z0REL_nwp + Z0_all)))
                acceleration_factor = np.where(acceleration_factor > 0, acceleration_factor, 1)
                exp_Wind = wind_speed_all * acceleration_factor

            # Wind speed scaling
            if verbose: print("Wind speed scaling")
            scaling_wind = exp_Wind if Z0_cond else wind_speed_nwp
            prediction = scaling_wind * prediction / 3

            # Wind computations
            if verbose: print("Wind computations")
            U_old = prediction[:, :, :, :, :, 0]  # Expressed in the rotated coord. system [m/s]
            V_old = prediction[:, :, :, :, :, 1]  # Expressed in the rotated coord. system [m/s]
            W_old = prediction[:, :, :, :, :, 2]  # Good coord. but not on the right pixel [m/s]
            del prediction
            UV = tf.math.sqrt(tf.square(U_old) + tf.square(V_old))  # Good coord. but not on the right pixel [m/s]
            alpha = tf.where(U_old == 0,
                             tf.where(V_old == 0, 0, tf.math.sign(V_old) * 3.14159 / 2),
                             tf.math.atan(V_old / U_old))  # Expressed in the rotated coord. system [radian]
            UV_DIR = (3.14159 / 180) * wind_DIR_nwp - alpha  # Good coord. but not on the right pixel [radian]

            # Reshape wind speed and wind direction
            wind_speed_nwp = tf.reshape(wind_speed_nwp, [nb_time_step, nb_px_nwp_y, nb_px_nwp_x])
            wind_DIR_nwp = tf.reshape(wind_DIR_nwp, [nb_time_step, nb_px_nwp_y, nb_px_nwp_x])

            # Calculate U and V along initial axis
            U_old = -tf.math.sin(UV_DIR) * UV  # Good coord. but not on the right pixel [m/s]
            V_old = -tf.math.sin(UV_DIR) * UV  # Good coord. but not on the right pixel [m/s]
            del UV
            del UV_DIR
            del alpha
            del wind_speed_nwp
            # Final results
            print("Creating wind map")

        # Good axis and pixel location [m/s]
        for time_step, time in enumerate(times):
            for idx_y_nwp in range(nb_px_nwp_y):
                for idx_x_nwp in range(nb_px_nwp_x):
                    wind_DIR = wind_DIR_nwp[time_step, idx_y_nwp, idx_x_nwp].numpy()
                    U = self.rotate_topography(
                        U_old[time_step, idx_y_nwp, idx_x_nwp, :, :].numpy(),
                        wind_DIR,
                        clockwise=True)
                    V = self.rotate_topography(
                        V_old[time_step, idx_y_nwp, idx_x_nwp, :, :].numpy(),
                        wind_DIR,
                        clockwise=True)
                    W = self.rotate_topography(
                        W_old[time_step, idx_y_nwp, idx_x_nwp, :, :].numpy(),
                        wind_DIR,
                        clockwise=True)

                    # Select index NWP
                    x_nwp_L93 = nwp_x_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data
                    y_nwp_L93 = nwp_y_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data

                    # Select indexes MNT
                    idx_x_mnt = int((x_nwp_L93 - xmin_mnt) // self.mnt.resolution_x)
                    idx_y_mnt = int((ymax_mnt - y_nwp_L93) // self.mnt.resolution_y)

                    # Select center of the predictions
                    wind_map[time_step, idx_y_mnt - 8:idx_y_mnt + 9, idx_x_mnt - 8:idx_x_mnt + 9, 0] = U[39 - 8:40 + 8,
                                                                                                       34 - 8:35 + 8]

                    wind_map[time_step, idx_y_mnt - 8:idx_y_mnt + 9, idx_x_mnt - 8:idx_x_mnt + 9, 1] = V[39 - 8:40 + 8,
                                                                                                       34 - 8:35 + 8]

                    wind_map[time_step, idx_y_mnt - 8:idx_y_mnt + 9, idx_x_mnt - 8:idx_x_mnt + 9, 2] = W[39 - 8:40 + 8,
                                                                                                       34 - 8:35 + 8]

        return (wind_map, [], nwp_data_initial, nwp_data, mnt_data)

    def prepare_time_and_domain_nwp(self, year_0, month_0, day_0, hour_0,year_1, month_1, day_1, hour_1,
                                    station_name=None, dx=None, dy=None):

        begin = datetime.datetime(year_0, month_0, day_0, hour_0)
        end = datetime.datetime(year_1, month_1, day_1, hour_1)
        self._select_timeframe_nwp(begin=begin,end=end)

        nwp_data = self._select_large_domain_around_station(station_name, dx, dy, type="NWP", additionnal_dx_mnt=None)
        return(nwp_data)

    def interpolate_wind_grid_xarray(self, nwp_data, interp=3, method='linear', verbose=True):

        # Calculate U_nwp and V_nwp
        nwp_data = self.horizontal_wind_component(working_with_xarray=True, xarray_data=nwp_data)

        # Drop variables
        nwp_data = nwp_data.drop_vars(["Wind", "Wind_DIR"])

        # Interpolate AROME
        nwp_data = self.interpolate_xarray_grid(xarray_data=nwp_data, interp=interp, method=method)
        nwp_data = self.compute_wind_speed(computing='xarray', xarray_data=nwp_data)

        return(nwp_data)

    def get_caracteristics_nwp(self, nwp_data):
        times = nwp_data.time.data
        nb_time_step = len(times)
        nwp_x_l93 = nwp_data.X_L93
        nwp_y_l93 = nwp_data.Y_L93
        nb_px_nwp_y, nb_px_nwp_x = nwp_x_l93.shape
        return(times, nb_time_step, nwp_x_l93, nwp_y_l93, nb_px_nwp_y, nb_px_nwp_x)

    def get_caracteristics_mnt(self, mnt_data, verbose=True):

        xmin_mnt = np.nanmin(mnt_data.x.data)
        ymax_mnt = np.nanmax(mnt_data.y.data)
        resolution_x = self.mnt.resolution_x
        resolution_y = self.mnt.resolution_y

        if verbose: print("__Selected NWP caracteristics")

        return(xmin_mnt, ymax_mnt, resolution_x, resolution_y)

    # todo save indexes second rotation
    def predict_map_indexes(self, station_name='Col du Lac Blanc', x_0=None, y_0=None, dx=10_000, dy=10_000, interp=3,
                            year_0=None, month_0=None, day_0=None, hour_0=None,
                            year_1=None, month_1=None, day_1=None, hour_1=None,
                            Z0_cond=False, verbose=True, peak_valley=True, method='linear',
                            log_profile_to_h_2=False, log_profile_from_h_2=False, log_profile_10m_to_3m=False):

        if not (_numba):
            raise ModuleNotFoundError('predict_map_indexes needs numba to operate')

        # Select NWP data
        if verbose: print("Selecting NWP")
        nwp_data = self.prepare_time_and_domain_nwp(year_0, month_0, day_0, hour_0, year_1, month_1, day_1, hour_1,
                                                    station_name=station_name, dx=dx, dy=dy)

        # Copy data
        nwp_data_initial = nwp_data.copy(deep=False)
        nwp_data = self.interpolate_wind_grid_xarray(nwp_data, interp=interp, method=method, verbose=verbose)

        # Time scale and domain length
        times, nb_time_step, nwp_x_l93, nwp_y_l93, nb_px_nwp_y, nb_px_nwp_x = self.get_caracteristics_nwp(nwp_data)

        # Select MNT data
        mnt_data = self._select_large_domain_around_station(station_name, dx, dy, type="MNT", additionnal_dx_mnt=2_000)
        xmin_mnt, ymax_mnt, resolution_x, resolution_y = self.get_caracteristics_mnt(mnt_data)

        # NWP forcing data
        if verbose: print("Selecting forcing data")
        wind_speed_nwp = nwp_data["Wind"].data
        wind_DIR_nwp = nwp_data["Wind_DIR"].data
        if Z0_cond:
            Z0_nwp = nwp_data["Z0"].data
            Z0REL_nwp = nwp_data["Z0REL"].data
            ZS_nwp = nwp_data["ZS"].data

        # Initialize wind map
        mnt_data, mnt_data_x, mnt_data_y, shape_x_mnt, shape_y_mnt = self._get_mnt_data_and_shape(mnt_data)
        coord = [mnt_data_x, mnt_data_y]
        wind_map = np.zeros((nb_time_step, shape_x_mnt, shape_y_mnt, 3), dtype=np.float32)
        topo_concat = np.empty((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79, 69), dtype=np.float32)
        peak_valley_height = np.empty((nb_px_nwp_y, nb_px_nwp_x), dtype=np.float32)
        mean_height = np.empty((nb_px_nwp_y, nb_px_nwp_x), dtype=np.float32)

        # Constants
        nb_pixel = 70

        # Load pre_rotated indexes
        all_mat = np.load(self.data_path + "MNT/indexes_rot.npy")
        all_mat = np.int32(all_mat.reshape(360, 79 * 69, 2))

        # Rotation function
        # (360, 5451, 2) (1, 21, 21, 5451) (21, 21, 140, 140) (1, 21, 21)
        #@jit([float32[:, :, :, :](int32[:, :, :], float32[:, :, :, :], float32[:, :, :, :], int32[:, :, :])], nopython=True)
        def rotate_topo(all_mat, topo_rot, topo_i, angles):
            for time in range(topo_rot.shape[0]):
                for y in range(topo_i.shape[0]):
                    for x in range(topo_i.shape[1]):
                        angle = angles[time, y, x]
                        for number in range(79 * 69):
                            topo_rot[time, y, x, number] = topo_i[y, x, all_mat[angle, number, 0], all_mat[angle, number, 1]]
            return (topo_rot)

        x_nwp_L93 = np.empty((nb_px_nwp_y, nb_px_nwp_x))
        y_nwp_L93 = np.empty((nb_px_nwp_y, nb_px_nwp_x))
        wind_DIR = np.empty((nb_time_step, nb_px_nwp_y, nb_px_nwp_x))
        i = 0
        for idx_y_nwp in range(nb_px_nwp_y):
            for idx_x_nwp in range(nb_px_nwp_x):
                # Select index NWP
                # todo change here
                x_nwp_L93[idx_y_nwp, idx_x_nwp] = nwp_x_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data
                y_nwp_L93[idx_y_nwp, idx_x_nwp] = nwp_y_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data

                for time in range(nb_time_step):
                    wind_DIR[time, idx_y_nwp, idx_x_nwp] = wind_DIR_nwp[time, idx_y_nwp, idx_x_nwp]

        # Select indexes MNT
        # todo change here
        idx_x_mnt = np.intp((x_nwp_L93[:,:] - xmin_mnt) // resolution_x)
        idx_y_mnt = np.intp((ymax_mnt - y_nwp_L93[:,:]) // resolution_y)

        # Large topo
        topo_i = np.empty((nb_px_nwp_y, nb_px_nwp_x, 140, 140)).astype(np.float32)
        for j in range(nb_px_nwp_y):
            for i in range(nb_px_nwp_x):
                y = idx_y_mnt[j,i]
                x = idx_x_mnt[j,i]
                topo_i[j, i,:,:] = mnt_data[0, y-nb_pixel:y + nb_pixel, x - nb_pixel:x + nb_pixel]

        # Mean peak_valley altitude
        peak_valley_height[:, :] = np.int32(2 * np.nanstd(topo_i))
        mean_height[:, :] = np.int32(np.mean(topo_i))

        # Wind direction
        angle = np.where(wind_DIR > 0, np.int32(wind_DIR - 1), np.int32(359))

        # Rotate topography
        topo_rot = np.empty((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79 * 69)).astype(np.float32)
        topo_rot = rotate_topo(all_mat, topo_rot, topo_i, angle)
        topo_rot = topo_rot.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79, 69))

        # Store result
        topo_concat[:, :, :, :, :] = topo_rot

        # Normalize
        _, std = self._load_norm_prm()
        mean_height = mean_height.reshape((1, nb_px_nwp_y, nb_px_nwp_x, 1, 1))
        std = self.get_closer_from_learning_conditions(topo_concat, mean_height, std, axis=(3,4))
        std = std.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1))
        topo_concat = self.normalize_topo(topo_concat, mean_height, std).astype(dtype=np.float32, copy=False)

        # Reshape for tensorflow
        topo_concat = topo_concat.reshape((nb_time_step * nb_px_nwp_x * nb_px_nwp_y, self.n_rows, self.n_col, 1))

        # Load model
        self.load_model(dependencies=True)

        # Predictions
        prediction = self.model.predict(topo_concat)
        print("Acceleration maximum CNN", np.nanmax(prediction/3))

        # Reshape predictions for analysis and broadcasting
        prediction = prediction.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, self.n_rows, self.n_col, 3)).astype(
            np.float32, copy=False)
        wind_speed_nwp = wind_speed_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1)).astype(np.float32,
                                                                                                          copy=False)
        wind_DIR_nwp = wind_DIR_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1)).astype(np.float32,
                                                                                                   copy=False)

        # Exposed wind speed
        if Z0_cond:

            # Reshape for broadcasting
            Z0_nwp = Z0_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            Z0REL_nwp = Z0REL_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            ZS_nwp = ZS_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            peak_valley_height = peak_valley_height.reshape((1, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            ten_m_array = peak_valley_height * 0 + 10
            three_m_array = peak_valley_height * 0 + 10

            # Choose height in the formula
            if peak_valley:
                height = peak_valley_height
            else:
                height = ZS_nwp

            # Apply log profile: 10m => h/2 or 10m => Zs
            if log_profile_to_h_2:
                wind_speed_nwp = self.apply_log_profile(z_in=ten_m_array, z_out=peak_valley_height / 2,
                                                        wind_in=wind_speed_nwp, z0=Z0_nwp,
                                                        verbose=verbose, z_in_verbose="10m", z_out_verbose="height/2")
            # Unexpose wind speed
            exp_Wind, acceleration_factor = self.exposed_wind_speed(wind_speed=wind_speed_nwp,
                                                                    z_out=(height / 2),
                                                                    z0=Z0_nwp,
                                                                    z0_rel=Z0REL_nwp)
            print("Acceleration maximum expose", np.nanmax(acceleration_factor))

            if log_profile_from_h_2:
                # Apply log profile: h/2 => 10m or Zs => 10m
                exp_Wind = self.apply_log_profile(z_in=peak_valley_height / 2, z_out=ten_m_array,
                                                  wind_in=exp_Wind, z0=Z0_nwp,
                                                  verbose=verbose, z_in_verbose="height/2", z_out_verbose="10m")

        # Wind speed scaling
        scaling_wind = exp_Wind if Z0_cond else wind_speed_nwp
        prediction = self.wind_speed_scaling(scaling_wind, prediction, linear=True)

        if log_profile_10m_to_3m:
            # Apply log profile: 3m => 10m
            prediction = self.apply_log_profile(z_in=three_m_array, z_out=ten_m_array, wind_in=prediction, z0=Z0_nwp)

        # Wind computations
        U_old = prediction[:, :, :, :, :, 0]  # Expressed in the rotated coord. system [m/s]
        V_old = prediction[:, :, :, :, :, 1]  # Expressed in the rotated coord. system [m/s]
        W_old = prediction[:, :, :, :, :, 2]  # Good coord. but not on the right pixel [m/s]

        # Recalculate with respect to original coordinates
        UV = self.compute_wind_speed(U=U_old, V=V_old, W=None)  # Good coord. but not on the right pixel [m/s]
        alpha = self.angular_deviation(U_old, V_old)  # Expressed in the rotated coord. system [radian]
        UV_DIR = self.direction_from_alpha(wind_DIR_nwp, alpha)  # Good coord. but not on the right pixel [radian]

        # float64 to float32
        UV_DIR = UV_DIR.astype(dtype=np.float32, copy=False)
        all_mat = all_mat.astype(np.int32, copy=False)

        # Reshape wind speed and wind direction
        wind_speed_nwp = wind_speed_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x))
        wind_DIR_nwp = wind_DIR_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x))

        # Calculate U and V along initial axis
        U_old, V_old = self.horizontal_wind_component(UV=UV,
                                                      UV_DIR=UV_DIR)  # Good coord. but not on the right pixel [m/s]

        # Reduce size matrix of indexes
        wind = np.empty((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79, 69, 3))
        wind[:, :, :, :, :, 0] = U_old
        wind[:, :, :, :, :, 1] = V_old
        wind[:, :, :, :, :, 2] = W_old
        wind = wind.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79 * 69, 3))

        # Good axis and pixel location [m/s]
        i = 0
        save_pixels = 10
        save_pixel_offset_left = 69 - save_pixels
        save_pixel_offset_right = 69 + save_pixels + 1

        @jit([float64[:,:,:,:,:,:](int32[:,:,:], float64[:,:,:,:,:,:], float64[:,:,:,:,:], int32[:,:,:])],
             nopython=True)
        def rotate_wind(all_mat, wind_large, wind, angle):
            for time in range(wind_large.shape[0]):
                for y in range(wind_large.shape[1]):
                    for x in range(wind_large.shape[2]):
                        angle_i = angle[time, y, x]
                        for number, (index_y, index_x) in enumerate(zip(all_mat[angle_i, :, 0], all_mat[angle_i, :, 1])):
                            wind_large[time, y, x, index_y, index_x, :] = wind[time, y, x, number, :]
            return (wind_large)

        print("__Wind rotations")
        wind_large = np.empty((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 140, 140, 3)) * np.nan
        wind_large = rotate_wind(all_mat, wind_large, wind, angle)


        for j in range(nb_px_nwp_y):
            for i in range(nb_px_nwp_x):
                mnt_y_left = idx_y_mnt[j,i] - save_pixels
                mnt_y_right = idx_y_mnt[j,i] + save_pixels + 1
                mnt_x_left = idx_x_mnt[j,i] - save_pixels
                mnt_x_right = idx_x_mnt[j,i] + save_pixels + 1

                wind_map[:, mnt_y_left:mnt_y_right,mnt_x_left:mnt_x_right, :] = wind_large[:,j,i,
                                                                                      save_pixel_offset_left:save_pixel_offset_right,
                                                                                      save_pixel_offset_left:save_pixel_offset_right]

        @jit([float32[:, :, :, :](int64, float32[:, :, :, :])], nopython=True)
        def interpolate_final_result(nb_time_step, wind_map):
            for time_step in range(nb_time_step):
                for component in range(3):
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
            return (wind_map)

        wind_map = interpolate_final_result(nb_time_step, wind_map)
        """
        # Interpolate final map
        print("Final interpolation")
        for time_step, time in enumerate(times):
            for component in range(3):

                # Select component to interpolate
                wind_component = wind_map[time_step, :, :, component]

                # Create x and y axis
                x = np.arange(0, wind_component.shape[1])
                y = np.arange(0, wind_component.shape[0])

                # Mask invalid values
                wind_component = np.ma.masked_invalid(wind_component)
                xx, yy = np.meshgrid(x, y)

                # Get only the valid values
                x1 = xx[~wind_component.mask]
                y1 = yy[~wind_component.mask]
                newarr = wind_component[~wind_component.mask]

                # Interpolate
                wind_map[time_step, :, :, component] = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear')
        """
        return (wind_map, coord, nwp_data_initial, nwp_data, mnt_data)