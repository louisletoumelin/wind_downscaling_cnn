import numpy as np
import pandas as pd
import xarray as xr
import datetime
from scipy.ndimage import rotate
from scipy import interpolate
import random  # Warning
import time
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import concurrent.futures
from numba import jit, prange, float64, float32, int32, int64

# Profiling

try:
    import numexpr as ne

    _numexpr = True
except:
    _numexpr = False

try:
    from shapely.geometry import Point

    _shapely_geometry = True
except:
    _shapely_geometry = False

try:
    import geopandas as gpd

    _geopandas = True
except:
    _geopandas = False

try:
    import dask

    _dask = True
except:
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

    def __init__(self, obs, mnt, nwp, model_path, GPU=False, data_path=None):
        self.observation = obs
        self.mnt = mnt
        self.nwp = nwp
        self.model_path = model_path
        self.data_path = data_path
        if GPU:
            # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            tf.debugging.set_log_device_placement(True)

    def exposed_wind_speed(self):
        self.nwp.data_xr = self.nwp.data_xr.assign(
            exp_Wind=lambda x: x.Wind * (np.log(10 / x.Z0) / np.log(10 / x.Z0REL)) * (x.Z0 / x.Z0REL) ** (0.0708))

    @staticmethod
    def rotate_topography(topography, wind_dir, clockwise=False):
        """Rotate a topography to a specified angle

        If wind_dir = 270° then angle = 270+90 % 360 = 360 % 360 = 0
        For wind coming from the West, there is no rotation
        """
        if not (clockwise):
            rotated_topography = rotate(topography, 90 + wind_dir, reshape=False, mode='constant', cval=np.nan)
        if clockwise:
            rotated_topography = rotate(topography, -90 - wind_dir, reshape=False, mode='constant', cval=np.nan)
        return (rotated_topography)

    def _rotate_topo_for_all_station(self):
        """Rotate the topography at all stations for each 1 degree angle of wind direction"""

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
    def apply_log_profile(z_in=None, z_out=None, wind_in=None, z0=None):
        if _numexpr:
            return(ne.evaluate("(wind_in * log(z_out / z0)) / log(z_in / z0)"))
        else:
            return((wind_in * np.log(z_out / z0)) / np.log(z_in / z0))

    @staticmethod
    def load_rotated_topo(wind_dir, station_name):
        angle_int = np.int64(np.round(wind_dir) % 360)
        angle_str = [str(angle) for angle in angle_int]
        topo_HD = [self.dict_rot_topo[station_name]["rotated_topo_HD"][angle][0][200 - 39:200 + 40, 200 - 34:200 + 35]
                   for angle in angle_str]
        return (topo_HD)

    def normalize_topo(self, topo_HD, mean, std, dtype=np.float32, tensorflow=False):
        if tensorflow:
            topo_HD = tf.constant(topo_HD, dtype=tf.float32)
            mean = tf.constant(mean, dtype=tf.float32)
            std = tf.constant(std, dtype=tf.float32)
            result = tf.subtract(topo_HD, mean)
            result = tf.divide(result, result)
            return (result)
        else:
            topo_HD = np.array(topo_HD, dtype=dtype)
        if self._numexpr:
            return (ne.evaluate("(topo_HD - mean) / std"))
        else:
            return ((topo_HD - mean) / std)

    def _select_nwp_time_serie_at_pixel(self, station_name, Z0=False):
        nwp_name = self.nwp.name
        stations = self.observation.stations
        y_idx_nwp, x_idx_nwp = stations[f"index_{nwp_name}_NN_0"][stations["name"] == station_name].values[0]
        y_idx_nwp, x_idx_nwp = np.int16(y_idx_nwp), np.int16(x_idx_nwp)
        nwp_instance = self.nwp.data_xr
        wind_dir = nwp_instance.Wind_DIR.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
        wind_speed = nwp_instance.Wind.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
        time = nwp_instance.time.data
        if Z0 == True:
            print("Indexes", x_idx_nwp, y_idx_nwp)
            Z0 = nwp_instance.Z0.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
            Z0REL = nwp_instance.Z0REL.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
            ZS = nwp_instance.ZS.isel(xx=x_idx_nwp, yy=y_idx_nwp).data
            return (wind_dir, wind_speed, time, Z0, Z0REL, ZS)
        return (wind_dir, wind_speed, time)

    # todo add xarray
    def _select_CNN_time_serie_at_pixel(self, station_name, variable):
        prediction = array_xr[variable].sel(station=station_name).isel(x=34, y=39).data
        time = array_xr.sel(station='Col du Lac Blanc').isel(x=34, y=39).time.data
        return (prediction, time)

    def load_model(self, dependencies=False):
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
        # todo load dependencies
        dict_norm = pd.read_csv(self.model_path + "dict_norm.csv")
        mean = dict_norm["0"].iloc[0]
        std = dict_norm["0"].iloc[1]
        return (mean, std)

    def predict_UV_with_CNN(self, stations_name, fast=False, verbose=True, plot=False, Z0_cond=True, peak_valley = True,
                            log_profile_to_h_2 = False, log_profile_from_h_2 = False, log_profile_10m_to_3m = False):
        """
        Predicts wind fields using CNN.

        This function takes advantage of broadcasting. It doesn't use np.concatenate to store data as it is slow on large
        arrays. This function should work on a GPU for distributed predictions (not tried yet).

        Processing time: 15 minutes for 1 month for 52 stations

        Input:

            station_name
            ex = ['Col du Lac Blanc']


        Options:

            fast: use pre-rotated topographies (not working)
            verbose: print detailed execution
            plot: plot intermediate results (to debug)


        Output:

            array_xr (xarray data storing wind fiels, forcing and topographic informations)

        """

        # Select timeframe
        self.nwp.select_timeframe()
        if verbose: print("Working on specific time window")

        # Simulation parameters
        nb_station = len(stations_name)
        _, wind_speed, _ = self._select_nwp_time_serie_at_pixel(random.choice(stations_name))
        nb_sim = len(wind_speed)

        # Load and rotate all topo
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
        ten_m_array = 10*np.ones((nb_station, nb_sim), dtype=np.float32)
        three_m_array = 3*np.ones((nb_station, nb_sim), dtype=np.float32)

        for idx_station, single_station in enumerate(stations_name):

            # Select nwp pixel
            if Z0_cond:
                wind_dir, wind_speed, time_xr, Z0, Z0REL, ZS = self._select_nwp_time_serie_at_pixel(single_station,
                                                                                                    Z0=Z0_cond)
            else:
                wind_dir, wind_speed, time_xr = self._select_nwp_time_serie_at_pixel(single_station, Z0=Z0_cond)

            if verbose: print(f"Selected time series for pixel at station: {single_station}")
            if verbose: print("Model selected: not Fast. Topographies are NOT already rotated")

            # Extract topography
            nb_pixel = 70  # min = 116/2
            topo_HD, topo_x_l93, topo_y_l93 = self.observation.extract_MNT_around_station(single_station,
                                                                                          self.mnt,
                                                                                          nb_pixel,
                                                                                          nb_pixel)

            # Rotate topographies
            if verbose: print('Begin rotate topographies')
            for time_step, angle in enumerate(wind_dir):

                # Rotate topography
                rotated_topo_large = self.rotate_topography(topo_HD, angle)
                rotated_topo = rotated_topo_large[nb_pixel - 39:nb_pixel + 40, nb_pixel - 34:nb_pixel + 35]
                topo[idx_station, time_step, :, :, 0] = rotated_topo

                if verbose:
                    if time_step == nb_sim // 5: print(" 1/5")
                    if time_step == 2 * (nb_sim // 5): print(" 2/5")
                    if time_step == 3 * (nb_sim // 5): print(" 3/5")
                    if time_step == 4 * (nb_sim // 5): print(" 4/5")
                    if time_step == 5 * (nb_sim // 5): print(" 5/5")

            # Store results
            wind_speed_all[idx_station, :] = wind_speed
            wind_dir_all[idx_station, :] = wind_dir
            if Z0_cond:
                Z0_all[idx_station, :] = Z0
                Z0REL_all[idx_station, :] = Z0REL
                ZS_all[idx_station, :] = ZS
            all_topo_HD[idx_station, :, :] = topo_HD[nb_pixel - 39:nb_pixel + 40, nb_pixel - 34:nb_pixel + 35]
            all_topo_x_small_l93[idx_station, :] = topo_x_l93[nb_pixel - 34:nb_pixel + 35]
            all_topo_y_small_l93[idx_station, :] = topo_y_l93[nb_pixel - 39:nb_pixel + 40]
            peak_valley_height[idx_station] = np.int32(2 * np.nanstd(topo_HD[nb_pixel - 39:nb_pixel + 40, nb_pixel - 34:nb_pixel + 35]))
            mean_height[idx_station] = np.int32(np.nanmean(topo_HD[nb_pixel - 39:nb_pixel + 40, nb_pixel - 34:nb_pixel + 35]))


        # Exposed wind
        if Z0_cond:
            Z0_all = np.where(Z0_all == 0, 1 * 10 ^ (-8), Z0_all)
            peak_valley_height = peak_valley_height.reshape((nb_station, 1))

            # Choose height in the formula
            height = peak_valley_height if peak_valley else ZS_all

            # Apply log profile: 10m => height/2
            wind1 = np.copy(wind_speed_all)
            if log_profile_to_h_2: wind_speed_all = self.apply_log_profile(z_in=ten_m_array, z_out=height/2, wind_in=wind_speed_all, z0=Z0_all)
            if verbose: print("Applied log profile: 10m => height/2")

            # Acceleration a1
            if self._numexpr:
                a1 = ne.evaluate("where(wind1 > 0, wind_speed_all / wind1, 1)")
            else:
                a1 = np.where(wind1 > 0, wind_speed_all / wind1, 1)


            wind2 = np.copy(wind_speed_all)
            if self._numexpr:
                acceleration_factor = ne.evaluate(
                    "log((height/2) / Z0_all) * (Z0_all / (Z0REL_all+Z0_all))**0.0706 / (log((height/2) / (Z0REL_all+Z0_all)))")
                exp_Wind = ne.evaluate("wind_speed_all * acceleration_factor")
            else:
                acceleration_factor = np.log((height/2) / Z0_all) * (Z0_all / (Z0REL_all + Z0_all)) ** 0.0706 / (np.log((height/2) / (Z0REL_all + Z0_all)))
                acceleration_factor = np.where(acceleration_factor > 0, acceleration_factor, 1)
                exp_Wind = wind_speed_all * acceleration_factor

            # Acceleration a2
            if self._numexpr:
                a2 = ne.evaluate("where(wind2 > 0, exp_Wind / wind2, 1)")
            else:
                a2 = np.where(wind2 > 0, exp_Wind / wind2, 1)


            # Apply log profile: height/2 => 3m
            wind3 = np.copy(exp_Wind)
            if log_profile_from_h_2: exp_Wind = self.apply_log_profile(z_in=height/2, z_out=three_m_array, wind_in=exp_Wind, z0=Z0_all)
            if verbose: print("Applied log profile: height => 3m")

            # Acceleration a3 and acceleration factor
            if self._numexpr:
                a3 = ne.evaluate("where(wind3 > 0, exp_Wind / wind3, 1)")
                acceleration_NWP = ne.evaluate("where(wind1 > 0, exp_Wind/wind1, 1)")
            else:
                a3 = np.where(wind3 > 0, exp_Wind / wind3, 1)
                acceleration_NWP = np.where(wind1 > 0, exp_Wind/wind1, 1)


        # Normalize
        mean, std = self._load_norm_prm()
        topo = self.normalize_topo(topo, mean_height.reshape((nb_station, 1, 1, 1, 1)), std)
        if verbose:
            print('Normalize done')
            print('Mean: ' + str(np.round(std)))

        # Reshape for tensorflow
        topo = topo.reshape((nb_station * nb_sim, self.n_rows, self.n_col, 1))
        if verbose: print('Reshaped tensorflow done')
        print(topo.shape)

        # Prediction
        """
        Warning: change dependencies here
        """
        self.load_model(dependencies=True)
        prediction = self.model.predict(topo)
        if verbose: print('Prediction done')

        # Acceleration NWP to CNN
        UVW_int = np.sqrt(prediction[:, :, :, 0] ** 2 + prediction[:, :, :, 1] ** 2 + prediction[:, :, :, 2] ** 2)
        three_m_s_constant = 3*np.ones(UVW_int.shape)
        if self._numexpr:
            acceleration_CNN = ne.evaluate("UVW_int/three_m_s_constant")
        else:
            acceleration_CNN = UVW_int/three_m_s_constant

        # Reshape predictions for analysis
        prediction = prediction.reshape((nb_station, nb_sim, self.n_rows, self.n_col, 3))
        if verbose:
            print('Prediction reshaped')
            print(prediction.shape)

        # Reshape for broadcasting
        wind_speed = wind_speed_all.reshape((nb_station, nb_sim, 1, 1, 1))
        if Z0_cond:
            exp_Wind = exp_Wind.reshape((nb_station, nb_sim, 1, 1, 1))
            acceleration_factor = acceleration_factor.reshape((nb_station, nb_sim, 1, 1, 1))
            peak_valley_height = peak_valley_height.reshape((nb_station, 1, 1, 1, 1))
            ten_m_array = ten_m_array.reshape((nb_station, nb_sim, 1, 1, 1))
            three_m_array = three_m_array.reshape((nb_station, nb_sim, 1, 1, 1))
            Z0_all = Z0_all.reshape((nb_station, nb_sim, 1, 1, 1))
        wind_dir = wind_dir_all.reshape((nb_station, nb_sim, 1, 1))


        # Wind speed scaling
        scaling_wind = exp_Wind if Z0_cond else wind_speed
        if self._numexpr:
            prediction = ne.evaluate("scaling_wind * prediction / 3")
        else:
            prediction = scaling_wind * prediction / 3

        if verbose: print('Wind speed scaling done')

        # Apply log profile: 3m => 10m
        wind4 = np.copy(prediction)
        if log_profile_10m_to_3m: prediction = self.apply_log_profile(z_in=three_m_array, z_out=ten_m_array, wind_in=prediction, z0=Z0_all)
        if verbose: print("Applied log profile: 3m => 10m")

        # Acceleration a4
        if self._numexpr:
            a4 = ne.evaluate("where(wind4 > 0, prediction/wind4, 1)")
        else:
            a4 = np.where(wind4 > 0, prediction / wind4, 1)

        # Wind computations
        U_old = prediction[:, :, :, :, 0]  # Expressed in the rotated coord. system [m/s]
        V_old = prediction[:, :, :, :, 1]  # Expressed in the rotated coord. system [m/s]
        W_old = prediction[:, :, :, :, 2]  # Good coord. but not on the right pixel [m/s]

        if self._numexpr:
            UV = ne.evaluate("sqrt(U_old**2 + V_old**2)")  # Good coord. but not on the right pixel [m/s]
            alpha = ne.evaluate(
                "where(U_old == 0, where(V_old == 0, 0, V_old/abs(V_old) * 3.14159 / 2), arctan(V_old / U_old))")
            UV_DIR = ne.evaluate("(3.14159/180) * wind_dir - alpha")  # Good coord. but not on the right pixel [radian]
        else:
            UV = np.sqrt(U_old ** 2 + V_old ** 2)  # Good coord. but not on the right pixel [m/s]
            alpha = np.where(U_old == 0,
                             np.where(V_old == 0, 0, np.sign(V_old) * np.pi / 2),
                             np.arctan(V_old / U_old))  # Expressed in the rotated coord. system [radian]
            UV_DIR = (np.pi / 180) * wind_dir - alpha  # Good coord. but not on the right pixel [radian]

        # Convert float64 to float32
        alpha = alpha.astype(np.float32, copy=False)
        UV_DIR = UV_DIR.astype(np.float32, copy=False)

        # Verification of shapes
        assert U_old.shape == (nb_station, nb_sim, self.n_rows, self.n_col)
        assert V_old.shape == (nb_station, nb_sim, self.n_rows, self.n_col)
        assert W_old.shape == (nb_station, nb_sim, self.n_rows, self.n_col)
        assert UV.shape == (nb_station, nb_sim, self.n_rows, self.n_col)
        assert alpha.shape == (nb_station, nb_sim, self.n_rows, self.n_col)
        assert UV.shape == UV_DIR.shape

        # Calculate U and V along initial axis
        if self._numexpr:
            U_old = ne.evaluate("-sin(UV_DIR) * UV")
            V_old = ne.evaluate("-cos(UV_DIR) * UV")
        else:
            U_old = -np.sin(UV_DIR) * UV  # Good coord. but not on the right pixel [m/s]
            V_old = -np.cos(UV_DIR) * UV  # Good coord. but not on the right pixel [m/s]

        if verbose: print('Wind computation done')

        # Rotate clockwise to put the wind value on the right topography pixel
        if verbose: print('Start rotating to initial position')

        U = np.empty((nb_station, nb_sim, self.n_rows, self.n_col), dtype=np.float32)
        V = np.empty((nb_station, nb_sim, self.n_rows, self.n_col), dtype=np.float32)
        W = np.empty((nb_station, nb_sim, self.n_rows, self.n_col), dtype=np.float32)

        for idx_station in range(nb_station):
            if verbose: print(f"Station {idx_station}/{nb_station}")
            for time_step in range(nb_sim):
                # Good axis and pixel location [m/s]
                U[idx_station, time_step, :, :] = self.rotate_topography(U_old[idx_station, time_step, :, :],
                                                                         wind_dir_all[idx_station, time_step],
                                                                         clockwise=True)
                V[idx_station, time_step, :, :] = self.rotate_topography(V_old[idx_station, time_step, :, :],
                                                                         wind_dir_all[idx_station, time_step],
                                                                         clockwise=True)
                W[idx_station, time_step, :, :] = self.rotate_topography(W_old[idx_station, time_step, :, :],
                                                                         wind_dir_all[idx_station, time_step],
                                                                         clockwise=True)

                if verbose:
                    if time_step == nb_sim // 10: print(" 1/10")
                    if time_step == 2 * (nb_sim // 10): print(" 2/10")
                    if time_step == 3 * (nb_sim // 10): print(" 3/10")
                    if time_step == 4 * (nb_sim // 10): print(" 4/10")
                    if time_step == 5 * (nb_sim // 10): print(" 5/10")
                    if time_step == 6 * (nb_sim // 10): print(" 6/10")
                    if time_step == 7 * (nb_sim // 10): print(" 7/10")
                    if time_step == 8 * (nb_sim // 10): print(" 8/10")
                    if time_step == 9 * (nb_sim // 10): print(" 9/10")
                    if time_step == 10 * (nb_sim // 10): print(" 10/10")

        if verbose: print('Wind prediction rotated for initial topography')

        # Compute wind direction
        if self._numexpr:
            UV_DIR = ne.evaluate("(180 + (180/3.14159)*arctan2(U,V)) % 360")
        else:
            UV_DIR = np.mod(180 + np.rad2deg(np.arctan2(U, V)), 360)  # Good axis and pixel [degree]
        if verbose: print('Final calculation: UV, UVW and UV_DIR_rad Done')

        # UVW
        UVW = np.sqrt(U ** 2 + V ** 2 + W ** 2)

        # Acceleration NWP to CNN
        wind1 = wind1.reshape((nb_station, nb_sim, 1, 1))
        if self._numexpr:
            acceleration_all = ne.evaluate("where(wind1 > 0, UVW/wind1, 1)")
        else:
            acceleration_all = np.where(wind1 > 0, UVW/wind1, 1)

        # Convert float64 to float32
        UV_DIR = UV_DIR.astype(np.float32, copy=False)

        # Reshape after broadcasting
        wind_speed = wind_speed.reshape((nb_station, nb_sim))
        wind_dir = wind_dir.reshape((nb_station, nb_sim))
        Z0_all = Z0_all.reshape((nb_station, nb_sim))
        if Z0_cond:
            exp_Wind = exp_Wind.reshape((nb_station, nb_sim))
            acceleration_factor = acceleration_factor.reshape((nb_station, nb_sim))
            a1 = a1.reshape((nb_station, nb_sim))
            a2 = a2.reshape((nb_station, nb_sim))
            a3 = a3.reshape((nb_station, nb_sim))
            a4 = np.max(a4, axis=4).reshape((nb_station, nb_sim, self.n_rows, self.n_col))
            acceleration_CNN = acceleration_CNN.reshape((nb_station, nb_sim, self.n_rows, self.n_col))
            peak_valley_height = peak_valley_height.reshape((nb_station))

        # Verification of shapes
        assert U.shape == (nb_station, nb_sim, self.n_rows, self.n_col)
        assert V.shape == (nb_station, nb_sim, self.n_rows, self.n_col)
        assert W.shape == (nb_station, nb_sim, self.n_rows, self.n_col)
        assert wind_speed.shape == (nb_station, nb_sim)
        assert wind_dir.shape == (nb_station, nb_sim)
        assert UV_DIR.shape == (nb_station, nb_sim, self.n_rows, self.n_col)
        if verbose: print('Reshape final predictions done')

        # Store results
        if verbose: 'Start creating array'
        array_xr = xr.Dataset(data_vars={"U": (["station", "time", "y", "x"], U),
                                         "V": (["station", "time", "y", "x"], V),
                                         "W": (["station", "time", "y", "x"], W),
                                         "UV": (["station", "time", "y", "x"], np.sqrt(U ** 2 + V ** 2)),
                                         "UVW": (["station", "time", "y", "x"], UVW),
                                         "UV_DIR_deg": (["station", "time", "y", "x"], UV_DIR),
                                         "alpha_deg": (["station", "time", "y", "x"],
                                                       wind_dir.reshape((nb_station, nb_sim, 1, 1)) - UV_DIR),
                                         "NWP_wind_speed": (["station", "time"], wind_speed),
                                         "NWP_wind_DIR": (["station", "time"], wind_dir),
                                         "ZS_mnt": (["station", "y", "x"], all_topo_HD,),
                                         "peak_valley_height" : (["station"],  peak_valley_height),
                                         "XX": (["station", "x"], all_topo_x_small_l93,),
                                         "YY": (["station", "y"], all_topo_y_small_l93,),
                                         "exp_Wind": (
                                         ["station", "time"], exp_Wind if Z0_cond else np.zeros((nb_station, nb_sim)),),
                                         "acceleration_factor": (
                                         ["station", "time"], acceleration_factor if Z0_cond else np.zeros((nb_station, nb_sim)),),
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
                                             a4 if Z0_cond else np.zeros((nb_station, nb_sim, self.n_rows, self.n_col)),),
                                         "acceleration_all": (
                                             ["station", "time", "y", "x"],
                                             acceleration_all if Z0_cond else np.zeros((nb_station, nb_sim, self.n_rows, self.n_col)),),
                                         "acceleration_CNN": (
                                             ["station", "time", "y", "x"],
                                             acceleration_CNN if Z0_cond else np.zeros((nb_station, nb_sim, self.n_rows, self.n_col)),),
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
        if verbose: 'Creating array done'

        return (array_xr)

    def predict_ideal_conditions(self, stations_name, fast=False, verbose=True, plot=False, Z0_cond=True, peak_valley=True, input_speed = 3, input_dir=270):
        """
        Ideal conditions

        """

        # Select timeframe
        self.nwp.select_timeframe()
        if verbose: print("Working on specific time window")

        # Simulation parameters
        nb_station = len(stations_name)
        _, wind_speed, _ = self._select_nwp_time_serie_at_pixel(random.choice(stations_name))
        nb_sim = len(wind_speed)

        # Load and rotate all topo
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

        for idx_station, single_station in enumerate(stations_name):

            # Select nwp pixel
            if Z0_cond:
                wind_dir, wind_speed, time_xr, Z0, Z0REL, ZS = self._select_nwp_time_serie_at_pixel(single_station,
                                                                                                    Z0=Z0_cond)
            else:
                wind_dir, wind_speed, time_xr = self._select_nwp_time_serie_at_pixel(single_station, Z0=Z0_cond)

            if verbose: print(f"Ideal: selected time series for pixel at station: {single_station}")

            wind_speed = wind_speed*0 + input_speed
            wind_dir = wind_dir*0 + input_dir

            # Extract topography
            nb_pixel = 70  # min = 116/2
            topo_HD, topo_x_l93, topo_y_l93 = self.observation.extract_MNT_around_station(single_station,
                                                                                          self.mnt,
                                                                                          nb_pixel,
                                                                                          nb_pixel)

            # Rotate topographies
            if verbose: print('Begin rotate topographies')
            for time_step, angle in enumerate(wind_dir):

                # Rotate topography
                rotated_topo_large = self.rotate_topography(topo_HD, angle)
                rotated_topo = rotated_topo_large[nb_pixel - 39:nb_pixel + 40, nb_pixel - 34:nb_pixel + 35]
                topo[idx_station, time_step, :, :, 0] = rotated_topo

            # Store results
            wind_speed_all[idx_station, :] = wind_speed
            wind_dir_all[idx_station, :] = wind_dir
            if Z0_cond:
                Z0_all[idx_station, :] = Z0
                Z0REL_all[idx_station, :] = Z0REL
                ZS_all[idx_station, :] = ZS
            all_topo_HD[idx_station, :, :] = topo_HD[nb_pixel - 39:nb_pixel + 40, nb_pixel - 34:nb_pixel + 35]
            all_topo_x_small_l93[idx_station, :] = topo_x_l93[nb_pixel - 34:nb_pixel + 35]
            all_topo_y_small_l93[idx_station, :] = topo_y_l93[nb_pixel - 39:nb_pixel + 40]
            peak_valley_height[idx_station] = np.int32(
                2 * np.nanstd(topo_HD[nb_pixel - 39:nb_pixel + 40, nb_pixel - 34:nb_pixel + 35]))
            mean_height[idx_station] = np.int32(
                np.nanmean(topo_HD[nb_pixel - 39:nb_pixel + 40, nb_pixel - 34:nb_pixel + 35]))

        # Exposed wind
        if Z0_cond:
            Z0_all = np.where(Z0_all == 0, 1 * 10 ^ (-8), Z0_all)
            peak_valley_height = peak_valley_height.reshape((nb_station, 1))

            # Choose height in the formula
            height = peak_valley_height if peak_valley else ZS_all

            # Apply log profile: 10m => height/2
            wind1 = np.copy(wind_speed_all)

            # Acceleration a1
            if self._numexpr:
                a1 = ne.evaluate("where(wind1 > 0, wind_speed_all / wind1, 1)")
            else:
                a1 = np.where(wind1 > 0, wind_speed_all / wind1, 1)

            wind2 = np.copy(wind_speed_all)
            if self._numexpr:
                acceleration_factor = ne.evaluate(
                    "log((height/2) / Z0_all) * (Z0_all / (Z0REL_all+Z0_all))**0.0706 / (log((height/2) / (Z0REL_all+Z0_all)))")
                exp_Wind = ne.evaluate("wind_speed_all * acceleration_factor")
            else:
                acceleration_factor = np.log((height / 2) / Z0_all) * (Z0_all / (Z0REL_all + Z0_all)) ** 0.0706 / (
                    np.log((height / 2) / (Z0REL_all + Z0_all)))
                acceleration_factor = np.where(acceleration_factor > 0, acceleration_factor, 1)
                exp_Wind = wind_speed_all * acceleration_factor

            # Acceleration a2
            if self._numexpr:
                a2 = ne.evaluate("where(wind2 > 0, exp_Wind / wind2, 1)")
            else:
                a2 = np.where(wind2 > 0, exp_Wind / wind2, 1)

            wind3 = np.copy(exp_Wind)

            # Acceleration a3 and acceleration factor
            if self._numexpr:
                a3 = ne.evaluate("where(wind3 > 0, exp_Wind / wind3, 1)")
                acceleration_NWP = ne.evaluate("where(wind1 > 0, exp_Wind/wind1, 1)")
            else:
                a3 = np.where(wind3 > 0, exp_Wind / wind3, 1)
                acceleration_NWP = np.where(wind1 > 0, exp_Wind / wind1, 1)

        # Normalize
        mean, std = self._load_norm_prm()
        topo = self.normalize_topo(topo, mean_height.reshape((nb_station, 1, 1, 1, 1)), std)
        if verbose:
            print('Normalize done')
            print('Mean: ' + str(np.round(std)))

        # Reshape for tensorflow
        topo = topo.reshape((nb_station * nb_sim, self.n_rows, self.n_col, 1))
        if verbose: print('Reshaped tensorflow done')
        print(topo.shape)

        # Prediction
        """
        Warning: change dependencies here
        """
        self.load_model(dependencies=True)
        prediction = self.model.predict(topo)
        if verbose: print('Prediction done')

        # Acceleration NWP to CNN
        UVW_int = np.sqrt(prediction[:, :, :, 0] ** 2 + prediction[:, :, :, 1] ** 2 + prediction[:, :, :, 2] ** 2)
        three_m_s_constant = 3 * np.ones(UVW_int.shape)
        if self._numexpr:
            acceleration_CNN = ne.evaluate("UVW_int/three_m_s_constant")
        else:
            acceleration_CNN = UVW_int / three_m_s_constant

        # Reshape predictions for analysis
        prediction = prediction.reshape((nb_station, nb_sim, self.n_rows, self.n_col, 3))
        if verbose:
            print('Prediction reshaped')
            print(prediction.shape)

        # Reshape for broadcasting
        wind_speed = wind_speed_all.reshape((nb_station, nb_sim, 1, 1, 1))
        if Z0_cond:
            exp_Wind = exp_Wind.reshape((nb_station, nb_sim, 1, 1, 1))
            acceleration_factor = acceleration_factor.reshape((nb_station, nb_sim, 1, 1, 1))
            peak_valley_height = peak_valley_height.reshape((nb_station, 1, 1, 1, 1))
            ten_m_array = ten_m_array.reshape((nb_station, nb_sim, 1, 1, 1))
            three_m_array = three_m_array.reshape((nb_station, nb_sim, 1, 1, 1))
            Z0_all = Z0_all.reshape((nb_station, nb_sim, 1, 1, 1))
        wind_dir = wind_dir_all.reshape((nb_station, nb_sim, 1, 1))

        # Wind speed scaling
        scaling_wind = exp_Wind if Z0_cond else wind_speed
        if self._numexpr:
            prediction = ne.evaluate("scaling_wind * prediction / 3")
        else:
            prediction = scaling_wind * prediction / 3

        if verbose: print('Wind speed scaling done')

        wind4 = np.copy(prediction)

        # Acceleration a4
        if self._numexpr:
            a4 = ne.evaluate("where(wind4 > 0, prediction/wind4, 1)")
        else:
            a4 = np.where(wind4 > 0, prediction / wind4, 1)

        # Wind computations
        U_old = prediction[:, :, :, :, 0]  # Expressed in the rotated coord. system [m/s]
        V_old = prediction[:, :, :, :, 1]  # Expressed in the rotated coord. system [m/s]
        W_old = prediction[:, :, :, :, 2]  # Good coord. but not on the right pixel [m/s]

        if self._numexpr:
            UV = ne.evaluate("sqrt(U_old**2 + V_old**2)")  # Good coord. but not on the right pixel [m/s]
            alpha = ne.evaluate(
                "where(U_old == 0, where(V_old == 0, 0, V_old/abs(V_old) * 3.14159 / 2), arctan(V_old / U_old))")
            UV_DIR = ne.evaluate("(3.14159/180) * wind_dir - alpha")  # Good coord. but not on the right pixel [radian]
        else:
            UV = np.sqrt(U_old ** 2 + V_old ** 2)  # Good coord. but not on the right pixel [m/s]
            alpha = np.where(U_old == 0,
                             np.where(V_old == 0, 0, np.sign(V_old) * np.pi / 2),
                             np.arctan(V_old / U_old))  # Expressed in the rotated coord. system [radian]
            UV_DIR = (np.pi / 180) * wind_dir - alpha  # Good coord. but not on the right pixel [radian]

        # Convert float64 to float32
        alpha = alpha.astype(np.float32, copy=False)
        UV_DIR = UV_DIR.astype(np.float32, copy=False)

        # Calculate U and V along initial axis
        if self._numexpr:
            U_old = ne.evaluate("-sin(UV_DIR) * UV")
            V_old = ne.evaluate("-cos(UV_DIR) * UV")
        else:
            U_old = -np.sin(UV_DIR) * UV  # Good coord. but not on the right pixel [m/s]
            V_old = -np.cos(UV_DIR) * UV  # Good coord. but not on the right pixel [m/s]

        if verbose: print('Wind computation done')

        # Rotate clockwise to put the wind value on the right topography pixel
        if verbose: print('Start rotating to initial position')

        U = np.empty((nb_station, nb_sim, self.n_rows, self.n_col), dtype=np.float32)
        V = np.empty((nb_station, nb_sim, self.n_rows, self.n_col), dtype=np.float32)
        W = np.empty((nb_station, nb_sim, self.n_rows, self.n_col), dtype=np.float32)

        for idx_station in range(nb_station):
            if verbose: print(f"Station {idx_station}/{nb_station}")
            for time_step in range(nb_sim):
                # Good axis and pixel location [m/s]
                U[idx_station, time_step, :, :] = self.rotate_topography(U_old[idx_station, time_step, :, :],
                                                                         wind_dir_all[idx_station, time_step],
                                                                         clockwise=True)
                V[idx_station, time_step, :, :] = self.rotate_topography(V_old[idx_station, time_step, :, :],
                                                                         wind_dir_all[idx_station, time_step],
                                                                         clockwise=True)
                W[idx_station, time_step, :, :] = self.rotate_topography(W_old[idx_station, time_step, :, :],
                                                                         wind_dir_all[idx_station, time_step],
                                                                         clockwise=True)

        if verbose: print('Wind prediction rotated for initial topography')

        # Compute wind direction
        if self._numexpr:
            UV_DIR = ne.evaluate("(180 + (180/3.14159)*arctan2(U,V)) % 360")
        else:
            UV_DIR = np.mod(180 + np.rad2deg(np.arctan2(U, V)), 360)  # Good axis and pixel [degree]
        if verbose: print('Final calculation: UV, UVW and UV_DIR_rad Done')

        # UVW
        UVW = np.sqrt(U ** 2 + V ** 2 + W ** 2)

        # Acceleration NWP to CNN
        wind1 = wind1.reshape((nb_station, nb_sim, 1, 1))
        if self._numexpr:
            acceleration_all = ne.evaluate("where(wind1 > 0, UVW/wind1, 1)")
        else:
            acceleration_all = np.where(wind1 > 0, UVW / wind1, 1)

        # Convert float64 to float32
        UV_DIR = UV_DIR.astype(np.float32, copy=False)

        # Reshape after broadcasting
        wind_speed = wind_speed.reshape((nb_station, nb_sim))
        wind_dir = wind_dir.reshape((nb_station, nb_sim))
        Z0_all = Z0_all.reshape((nb_station, nb_sim))
        if Z0_cond:
            exp_Wind = exp_Wind.reshape((nb_station, nb_sim))
            acceleration_factor = acceleration_factor.reshape((nb_station, nb_sim))
            a1 = a1.reshape((nb_station, nb_sim))
            a2 = a2.reshape((nb_station, nb_sim))
            a3 = a3.reshape((nb_station, nb_sim))
            a4 = np.max(a4, axis=4).reshape((nb_station, nb_sim, self.n_rows, self.n_col))
            acceleration_CNN = acceleration_CNN.reshape((nb_station, nb_sim, self.n_rows, self.n_col))
            peak_valley_height = peak_valley_height.reshape((nb_station))

        # Verification of shapes
        if verbose: print('Reshape final predictions done')

        # Store results
        if verbose: 'Start creating array'
        array_xr = xr.Dataset(data_vars={"U": (["station", "time", "y", "x"], U),
                                         "V": (["station", "time", "y", "x"], V),
                                         "W": (["station", "time", "y", "x"], W),
                                         "UV": (["station", "time", "y", "x"], np.sqrt(U ** 2 + V ** 2)),
                                         "UVW": (["station", "time", "y", "x"], UVW),
                                         "UV_DIR_deg": (["station", "time", "y", "x"], UV_DIR),
                                         "alpha_deg": (["station", "time", "y", "x"],
                                                       wind_dir.reshape((nb_station, nb_sim, 1, 1)) - UV_DIR),
                                         "NWP_wind_speed": (["station", "time"], wind_speed),
                                         "NWP_wind_DIR": (["station", "time"], wind_dir),
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
        if verbose: 'Creating array done'

        return (array_xr)

    def _select_large_domain_around_station(self, station_name, dx, dy, type="NWP", additionnal_dx_mnt=None):
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

    @staticmethod
    def nansum_matrix(A, B):
        return (np.where(np.isnan(A), B, A + np.nan_to_num(B)))

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
                        peak_valley_height[idx_y_nwp, idx_x_nwp] = np.int32(2*np.nanstd(topo_i))

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
                acceleration_factor = ne.evaluate("log((height/2) / Z0_nwp) * (Z0_nwp / (Z0REL_nwp+Z0_nwp))**0.0706 / (log((height/2) / (Z0REL_nwp+Z0_nwp)))")
                acceleration_factor = ne.evaluate("where(acceleration_factor > 0, acceleration_factor, 1)")
                exp_Wind = ne.evaluate("wind_speed_all * acceleration_factor")
            else:
                acceleration_factor = np.log((height/2) / Z0_nwp) * (Z0_nwp / (Z0REL_nwp + Z0_nwp)) ** 0.0706 / (np.log((height/2) / (Z0REL_nwp + Z0_nwp)))
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

                    """
                    wind_map[time_step, idx_x_mnt - 39:idx_x_mnt + 40, idx_y_mnt - 34:idx_y_mnt + 35,
                    0] = self.nansum_matrix(
                        wind_map[time_step, idx_x_mnt - 39:idx_x_mnt + 40, idx_y_mnt - 34:idx_y_mnt + 35, 0],
                        gaussian_weight * U[time_step, idx_x_nwp, idx_y_nwp, :, :])

                    wind_map[time_step, idx_x_mnt - 39:idx_x_mnt + 40, idx_y_mnt - 34:idx_y_mnt + 35,
                    1] = self.nansum_matrix(
                        wind_map[time_step, idx_x_mnt - 39:idx_x_mnt + 40, idx_y_mnt - 34:idx_y_mnt + 35, 1],
                        gaussian_weight * V[time_step, idx_x_nwp, idx_y_nwp, :, :])

                    wind_map[time_step, idx_x_mnt - 39:idx_x_mnt + 40, idx_y_mnt - 34:idx_y_mnt + 35, 2] = self.nansum_matrix(
                        wind_map[time_step, idx_x_mnt - 39:idx_x_mnt + 40, idx_y_mnt - 34:idx_y_mnt + 35, 2],
                        gaussian_weight * W[time_step, idx_x_nwp, idx_y_nwp, :, :])

                    nan_location = np.sign(U[time_step, idx_x_nwp, idx_y_nwp, :, :] + 1000)
                    weights[time_step, idx_x_mnt - 39:idx_x_mnt + 40, idx_y_mnt - 34:idx_y_mnt + 35] = self.nansum_matrix(weights[time_step, idx_x_mnt - 39:idx_x_mnt + 40, idx_y_mnt - 34:idx_y_mnt + 35], gaussian_weight * nan_location)
                    """
        # wind_map = wind_map / weights.reshape((nb_time_step, shape_x_mnt, shape_y_mnt, 1))
        print(wind_map.shape)
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
            topo_concat = self.normalize_topo(topo_concat, mean, std, tensorflow=True)

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

                acceleration_factor = tf.math.log((height/2) / Z0_nwp) * (Z0_nwp / (Z0REL_nwp + Z0_nwp)) ** 0.0706 / (np.log((height/2) / (Z0REL_nwp + Z0_all)))
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

        print(wind_map.shape)
        return (wind_map, [], nwp_data_initial, nwp_data, mnt_data)

    # todo save indexes second rotation
    def predict_map_indexes(self, station_name='Col du Lac Blanc', x_0=None, y_0=None, dx=10_000, dy=10_000, interp=3,
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

        coord = [mnt_data_x, mnt_data_y]

        # Concatenate topographies along single axis
        if verbose: print("Concatenate topographies along single axis")
        wind_map = np.zeros((nb_time_step, shape_x_mnt, shape_y_mnt, 3), dtype=np.float32)
        topo_concat = np.empty((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79, 69), dtype=np.float32)
        peak_valley_height = np.empty((nb_px_nwp_y, nb_px_nwp_x), dtype=np.float32)
        nb_pixel = 70

        # Load pre_rotated indexes
        all_mat = np.load(self.data_path +"MNT/indexes_rot.npy")
        all_mat = np.int32(all_mat.reshape(360, 79 * 69, 2))

        @jit([float32[:](int32[:,:,:], float32[:], float32[:,:], int32)], nopython=True)
        def rotate_topo(all_mat, topo_rot, topo_i, angle):
            for number in range(79 * 69):
                topo_rot[number] = topo_i[all_mat[angle, number, 0], all_mat[angle, number, 1]]
            return(topo_rot)

        i=0
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

                    # Mean peak_valley altitude
                    if time_step == 0:
                        peak_valley_height[idx_y_nwp, idx_x_nwp] = np.int32(2*np.nanstd(topo_i))

                    if i == 100:
                        plt.figure()
                        plt.imshow(topo_i)
                        plt.title("Initial topo")

                    # Wind direction
                    wind_DIR = wind_DIR_nwp[time_step, idx_y_nwp, idx_x_nwp]
                    angle = np.int32(wind_DIR - 1) if wind_DIR > 0 else np.int32(359)

                    # Rotate topography
                    topo_rot = np.empty((79*69)).astype(np.float32)
                    topo_rot = rotate_topo(all_mat, topo_rot, topo_i, angle)
                    topo_rot = topo_rot.reshape((79, 69))

                    # Plot
                    if i == 100:
                        plt.figure()
                        plt.imshow(topo_rot)
                        plt.title("Rotated for wind dir: " + str(np.round(wind_DIR)))
                    i = i +1

                    # Store result
                    topo_concat[time_step, idx_y_nwp, idx_x_nwp, :, :] = topo_rot

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

            # Reshape for broadcasting
            Z0_nwp = Z0_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            Z0REL_nwp = Z0REL_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            ZS_nwp = ZS_nwp.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            peak_valley_height = peak_valley_height.reshape((1, nb_px_nwp_y, nb_px_nwp_x, 1, 1, 1))
            ten_m_array = peak_valley_height*0 + 10
            three_m_array = peak_valley_height * 0 + 10

            # Choose height in the formula
            if peak_valley:
                height = peak_valley_height
            else:
                height = ZS_nwp

            # Apply log profile: 10m => h/2 or 10m => Zs
            wind_speed_nwp = self.apply_log_profile(z_in=ten_m_array, z_out=peak_valley_height/2, wind_in=wind_speed_nwp, z0=Z0_nwp)

            if self._numexpr:
                acceleration_factor = ne.evaluate(
                    "log((height/2) / Z0_nwp) * (Z0_nwp / (Z0REL_nwp+Z0_nwp))**0.0706 / (log((height/2) / (Z0REL_nwp+Z0_nwp)))")
                acceleration_factor = ne.evaluate("where(acceleration_factor > 0, acceleration_factor, 1)")
                exp_Wind = ne.evaluate("wind_speed_nwp * acceleration_factor")
            else:
                acceleration_factor = np.log((height / 2) / Z0_nwp) * (Z0_nwp / (Z0REL_nwp + Z0_nwp)) ** 0.0706 / (
                    np.log((height / 2) / (Z0REL_nwp + Z0_nwp)))
                acceleration_factor = np.where(acceleration_factor > 0, acceleration_factor, 1)
                exp_Wind = wind_speed_nwp * acceleration_factor

            # Apply log profile: h/2 => 10m or Zs => 10m
            exp_Wind = self.apply_log_profile(z_in=peak_valley_height/2, z_out=ten_m_array, wind_in=exp_Wind, z0=Z0_nwp)

        # Wind speed scaling
        if verbose: print("Wind speed scaling")
        scaling_wind = exp_Wind if Z0_cond else wind_speed_nwp
        if self._numexpr:
            prediction = ne.evaluate("scaling_wind * prediction / 3")
        else:
            prediction = scaling_wind * prediction / 3

        # Apply log profile: 3m => 10m
        prediction = self.apply_log_profile(z_in=three_m_array, z_out=ten_m_array, wind_in=prediction, z0=Z0_nwp)

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

        # Reduce size matrix of indexes
        wind = np.empty((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79, 69, 3))
        wind[:,:,:,:,:,0] = U_old
        wind[:,:,:,:,:,1] = V_old
        wind[:,:,:,:,:,2] = W_old
        wind = wind.reshape((nb_time_step, nb_px_nwp_y, nb_px_nwp_x, 79*69, 3))

        @jit([float64[:,:,:](int32[:,:,:], float64[:,:,:], float64[:,:,:,:,:], int32, int64, int64)], nopython=True)
        def rotate_wind(all_mat, wind_large, wind, angle, idx_y_nwp, idx_x_nwp):
            for number, (index_y, index_x) in enumerate(zip(all_mat[angle, :, 0], all_mat[angle, :, 1])):
                wind_large[index_y, index_x, :] = wind[time_step, idx_y_nwp, idx_x_nwp, number, :]
            return(wind_large)

        # Good axis and pixel location [m/s]
        i = 0
        save_pixels = 10
        all_mat = all_mat.astype(np.int32, copy=False)
        print("Wind rotations")
        for time_step, time in enumerate(times):
            for idx_y_nwp in range(nb_px_nwp_y):
                for idx_x_nwp in range(nb_px_nwp_x):

                    # Wind direction
                    wind_DIR = wind_DIR_nwp[time_step, idx_y_nwp, idx_x_nwp]
                    angle = np.int32(wind_DIR - 1) if wind_DIR > 0 else np.int32(359)
                    wind_large = np.empty((140, 140, 3))*np.nan
                    wind_large = rotate_wind(all_mat, wind_large, wind, angle, idx_y_nwp, idx_x_nwp)

                    # Select index NWP
                    x_nwp_L93 = nwp_x_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data
                    y_nwp_L93 = nwp_y_l93.isel(xx=idx_x_nwp, yy=idx_y_nwp).data

                    # Select indexes MNT
                    idx_x_mnt = int((x_nwp_L93 - xmin_mnt) // self.mnt.resolution_x)
                    idx_y_mnt = int((ymax_mnt - y_nwp_L93) // self.mnt.resolution_y)

                    # Select center of the predictions
                    if i == 100:
                        plt.figure()
                        plt.imshow(np.sqrt((wind_large[:, :, 0]**2+wind_large[:, :, 1]**2)))
                        plt.title("Wind large")
                    i = i + 1
                    wind_map[time_step, idx_y_mnt - save_pixels:idx_y_mnt + save_pixels + 1, idx_x_mnt - save_pixels:idx_x_mnt + save_pixels + 1, :] = wind_large[69 - save_pixels:69 + save_pixels+1, 69 - save_pixels:69 + save_pixels + 1]

        @jit([float32[:,:,:,:](int64, float32[:,:,:,:])], nopython=True)
        def interpolate_final_result(nb_time_step, wind_map):
            for time_step in range(nb_time_step):
                for component in range(3):
                    # Select component to interpolate
                    wind_component = wind_map[time_step, :, :, component]
                    nan_indexes = np.argwhere(np.isnan(wind_component))
                    for y, x in nan_indexes:
                        right = (y, x+1)
                        left = (y, x-1)
                        up = (y+1, x)
                        down = (y-1, x)
                        neighbors_indexes = [right, left, up, down]
                        neighbors = np.array([wind_component[index] for index in neighbors_indexes])
                        if np.isnan(neighbors).sum() <= 3:
                            wind_map[time_step, y, x, component] = np.mean(neighbors[~np.isnan(neighbors)])
            return(wind_map)

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
