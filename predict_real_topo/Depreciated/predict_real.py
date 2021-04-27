import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from scipy.ndimage import rotate
import random # Warning

import concurrent.futures # Warning
from tensorflow.keras.models import load_model

import pyproj # Warning
from shapely.geometry import Point # Warning
from shapely.geometry import Polygon # Warning
import geopandas as gpd # Warning
import cartopy.crs as ccrs # Warning
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d # Warning
import cartopy.feature as cfeature # Warning

from tensorflow.keras import backend as K


# Custom Metrics : NRMSE
def nrmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred))) / (K.max(y_pred) - K.min(y_pred))


# Custom Metrics : RMSE
def root_mse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


class Data_2D:

    def __init__(self, path_to_file, name):
        self.path_to_file = path_to_file
        self.name = name

    @staticmethod
    def x_y_to_stacked_xy(x_array, y_array):
        stacked_xy = np.dstack((x_array, y_array))
        return (stacked_xy)

    @staticmethod
    def grid_to_flat(stacked_xy):
        x_y_flat = [tuple(i) for line in stacked_xy for i in line]
        return (x_y_flat)

    @property
    def length(self):
        return (self.shape[1])

    @property
    def height(self):
        return (self.shape[0])


class MNT(Data_2D):

    def __init__(self, path_to_file, name):
        super().__init__(path_to_file, name)
        self.data_xr = xr.open_rasterio(path_to_file)
        self.data = self.data_xr.values[0, :, :]

    def find_nearest_MNT_index(self, x, y):
        xmin_MNT = np.min(self.data_xr.x.data)
        ymax_MNT = np.max(self.data_xr.y.data)

        index_x_MNT = (x - xmin_MNT) // self.resolution_x
        index_y_MNT = (ymax_MNT - y) // self.resolution_y
        return (index_x_MNT, index_y_MNT)

    @property
    def shape(self):
        return (self.data.shape)

    @property
    def resolution_x(self):
        return (self.data_xr.res[0])

    @property
    def resolution_y(self):
        return (self.data_xr.res[1])


class NWP(Data_2D):

    def __init__(self, path_to_file, name, begin, end):
        super().__init__(path_to_file, name)
        self.data_xr = xr.open_dataset(path_to_file)
        self.begin = begin
        self.end = end

    @property
    def shape(self):
        return (self.data_xr['LON'].shape)

    def gps_to_l93(self):
        """Converts LAT/LON information from a NWP to L93"""

        # Initialization
        X_L93 = np.zeros(self.shape)
        Y_L93 = np.zeros(self.shape)

        # Load transformer
        gps_to_l93_func = pyproj.Transformer.from_crs(4326, 2154, always_xy=True)

        # Transform coordinates of each points
        for i in range(self.height):
            for j in range(self.length):
                projected_points = [point for point in
                                    gps_to_l93_func.itransform(
                                        [(self.data_xr['LON'][i, j], self.data_xr['LAT'][i, j])])]
                X_L93[i, j], Y_L93[i, j] = projected_points[0]

        # Create a new variable with new coordinates
        self.data_xr["X_L93"] = (("yy", "xx"), X_L93)
        self.data_xr["Y_L93"] = (("yy", "xx"), Y_L93)

    def select_timeframe(self):
        self.data_xr = self.data_xr.sel(time=slice(self.begin, self.end))


class Observation:

    def __init__(self, path_to_list_stations, path_to_time_series):
        self.path_to_list_stations = path_to_list_stations
        self.path_to_time_series = path_to_time_series
        self.stations = pd.read_csv(path_to_list_stations)
        self.time_series = pd.read_csv(path_to_time_series)
        self.time_series.index = self.time_series['date'].apply(lambda x: np.datetime64(x))

    def stations_to_gdf(self, from_epsg="epsg:4326", x="LON", y="LAT"):
        """
        Input: Dataframe 1D
        Output: GeoDataFrame 1D
        """

        if from_epsg == "epsg:4326":
            crs = {"init": from_epsg}
        else:
            crs = from_epsg

        self.stations = gpd.GeoDataFrame(self.stations,
                                         geometry=gpd.points_from_xy(self.stations[x], self.stations[y]),
                                         crs=crs)

    def update_stations_with_KNN_from_NWP(self, number_neighbor, nwp):
        """Update a Observations.station (DataFrame) with index of NN in nwp

        ex: BDclim.update_stations_with_KNN_from_NWP(4, AROME) gives information about the 4 KNN at the each BDclim
        station from AROME
        """

        def K_N_N_point(point):
            distance, idx = tree.query(point, k=number_neighbor)
            return (distance, idx)

        # Reference stations
        list_coord_station = zip(self.stations['X'].values, self.stations['Y'].values)

        # Coordinates where to find neighbors
        stacked_xy = Data_2D.x_y_to_stacked_xy(nwp.data_xr["X_L93"], nwp.data_xr["Y_L93"])
        grid_flat = Data_2D.grid_to_flat(stacked_xy)
        tree = cKDTree(grid_flat)

        # Parallel computation of nearest neighbors
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                list_nearest = executor.map(K_N_N_point, list_coord_station)
        except:
            print("Parallel computation using concurrent.futures didn't work, so update_stations_with_KNN_from_NWP will not be parallelized.")
            list_nearest = map(K_N_N_point, list_coord_station)

        # Store results as array
        list_nearest = np.array([np.array(station) for station in list_nearest])
        list_index = [(x, y) for x in range(nwp.height) for y in range(nwp.length)]

        # Update DataFrame
        for neighbor in range(number_neighbor):
            self.stations[f'delta_x_{nwp.name}_NN_{neighbor}'] = list_nearest[:, 0, neighbor]
            self.stations[f'{nwp.name}_NN_{neighbor}'] = [Point(grid_flat[int(index)]) for index in
                                                          list_nearest[:, 1, neighbor]]
            self.stations[f'index_{nwp.name}_NN_{neighbor}'] = [list_index[int(index)] for index in
                                                                list_nearest[:, 1, neighbor]]

    def update_stations_with_KNN_from_MNT(self, mnt):
        index_x_MNT, index_y_MNT = mnt.find_nearest_MNT_index(self.stations["X"], self.stations["Y"])
        self.stations[f"index_X_NN_{mnt.name}"] = index_x_MNT
        self.stations[f"index_Y_NN_{mnt.name}"] = index_y_MNT

    def update_stations_with_KNN_from_MNT_using_cKDTree(self, mnt, number_of_neighbors=4):

        all_MNT_index_x, all_MNT_index_y = mnt.find_nearest_MNT_index(self.stations["X"], self.stations["Y"])
        nb_station = len(all_MNT_index_x)

        arrays_nearest_neighbors_l93 = np.zeros((number_of_neighbors, nb_station, 2))
        arrays_nearest_neighbors_index = np.zeros((number_of_neighbors, nb_station, 2))
        arrays_nearest_neighbors_delta_x = np.zeros((number_of_neighbors, nb_station))
        for idx_station in range(nb_station):
            l93_station_x, l93_station_y = self.stations["X"].values[idx_station], self.stations["Y"].values[
                idx_station]
            index_MNT_x = int(all_MNT_index_x.values[idx_station])
            index_MNT_y = int(all_MNT_index_y.values[idx_station])

            list_nearest_neighbors = []
            list_index_neighbors = []
            for i in [-2, -1, 0, 1, 2]:
                for j in [-2, -1, 0, 1, 2]:
                    l93_neighbor_x = mnt.data_xr.x.data[index_MNT_x + i]
                    l93_neighbor_y = mnt.data_xr.y.data[index_MNT_y + j]
                    list_nearest_neighbors.append((l93_neighbor_x, l93_neighbor_y))
                    list_index_neighbors.append((index_MNT_x + i, index_MNT_y + j))

            tree = cKDTree(list_nearest_neighbors)
            distance, all_idx = tree.query((l93_station_x, l93_station_y), k=number_of_neighbors)
            for index, idx_neighbor in enumerate(all_idx):
                l93_nearest_neighbor = list_nearest_neighbors[idx_neighbor]
                index_MNT_nearest_neighbor = list_index_neighbors[idx_neighbor]
                arrays_nearest_neighbors_l93[index, idx_station, :] = list(l93_nearest_neighbor)
                arrays_nearest_neighbors_index[index, idx_station, :] = list(index_MNT_nearest_neighbor)
                arrays_nearest_neighbors_delta_x[index, idx_station] = distance[index]

        mnt_name = mnt.name
        for neighbor in range(number_of_neighbors):
            self.stations[f"index_{mnt_name}_NN_{neighbor}_cKDTree"] = [tuple(index) for index in
                                                                        arrays_nearest_neighbors_index[neighbor, :]]
            self.stations[f"{mnt_name}_NN_{neighbor}_cKDTree"] = [tuple(coord) for coord in
                                                                  arrays_nearest_neighbors_l93[neighbor, :]]
            self.stations[f"delta_x_{mnt_name}_NN_{neighbor}_cKDTree"] = arrays_nearest_neighbors_delta_x[neighbor, :]

    def extract_MNT_around_station(self, station, mnt, nb_pixel_x, nb_pixel_y):
        condition = self.stations["name"] == station
        (index_x, index_y) = self.stations[[f"index_{mnt.name}_NN_0_cKDTree"]][condition].values[0][0]
        index_x, index_y = int(index_x), int(index_y)
        MNT_data = mnt.data[index_y - nb_pixel_y:index_y + nb_pixel_y, index_x - nb_pixel_x:index_x + nb_pixel_x]
        MNT_x = mnt.data_xr.x.data[index_x - nb_pixel_x:index_x + nb_pixel_x]
        MNT_y = mnt.data_xr.y.data[index_y - nb_pixel_y:index_y + nb_pixel_y]
        return (MNT_data, MNT_x, MNT_y)


class Processing:
    n_rows, n_col = 79, 69

    def __init__(self, obs, mnt, nwp, model_path):
        self.observation = obs
        self.mnt = mnt
        self.nwp = nwp
        self.model_path = model_path

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

    def rotate_topo_for_all_station(self):
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
            print("Parallel computation using concurrent.futures didn't work, so rotate_topo_for_all_degrees will not be parallelized.")
            map(rotate_topo_for_all_degrees, self.observation.stations['name'].values)
        self.dict_rot_topo = dict_topo

    @staticmethod
    def load_rotated_topo(wind_dir, station_name):
        angle_int = np.int64(np.round(wind_dir) % 360)
        angle_str = [str(angle) for angle in angle_int]
        topo_HD = [self.dict_rot_topo[station_name]["rotated_topo_HD"][angle][0][200 - 39:200 + 40, 200 - 34:200 + 35]
                   for angle in angle_str]
        return (topo_HD)

    @staticmethod
    def normalize_topo(topo_HD, mean, std):
        return ((np.array(topo_HD) - mean) / std)

    def select_nwp_time_serie_at_pixel(self, station_name):
        nwp_name = self.nwp.name
        stations = self.observation.stations
        (x_idx_nwp, y_idx_nwp) = stations[f"index_{nwp_name}_NN_0"][stations["name"] == station_name].values[0]
        nwp_instance = self.nwp.data_xr
        wind_dir = nwp_instance.Wind_DIR.data[:, x_idx_nwp, y_idx_nwp]
        wind_speed = nwp_instance.Wind.data[:, x_idx_nwp, y_idx_nwp]
        time = nwp_instance.time.data
        return (wind_dir, wind_speed, time)

    def load_model(self, dependencies=False):
        # todo load dependencies
        if dependencies:

            def root_mse(y_true, y_pred):
                return K.sqrt(K.mean(K.square(y_true - y_pred)))

            dependencies = {'root_mse': root_mse}
            model = load_model(self.model_path + "/fold_0.h5", custom_objects=dependencies)
        else:
            model = load_model(self.model_path + "/fold_0.h5")
        self.model = model

    def load_norm_prm(self):
        # todo load dependencies
        dict_norm = pd.read_csv(self.model_path + "/dict_norm.csv")
        mean = dict_norm["0"].iloc[0]
        std = dict_norm["0"].iloc[1]
        return (mean, std)

    def predict_UV_with_CNN(self, stations_name, fast=False, verbose=True, plot=False):

        # Select timeframe
        self.nwp.select_timeframe()
        if verbose: print("Working on specific time window")

        # Station names
        if stations_name == 'all':
            stations_name = self.observation.stations["name"].values

        # If fast: pre-rotate all topo
        if fast:
            self.rotate_topo_for_all_station()

        if plot:
            random_station_idx = random.choice(list(range(len(stations_name))))

        # Load and rotate all topo
        nb_station = len(stations_name)
        for idx_station, single_station in enumerate(stations_name):

            # Select nwp pixel
            wind_dir, wind_speed, time_xr = self.select_nwp_time_serie_at_pixel(single_station)
            if verbose: print(f"Selected time series for pixel at station: {single_station}")
            if fast:
                if verbose: print("Model selected: Fast. Topographies are already rotated")
                rotated_topo = load_rotated_topo(wind_dir, single_station)
            else:
                if verbose: print("Model selected: not Fast. Topographies are NOT already rotated")

                nb_sim = len(wind_dir)
                if plot:
                    random_index = random.choice(list(range(nb_sim)))

                # Extract topography
                topo_HD, topo_x_l93, topo_y_l93 = self.observation.extract_MNT_around_station(single_station,
                                                                                              self.mnt,
                                                                                              400,
                                                                                              400)
                topo_x_l93_small_domain = topo_x_l93[400 - 34:400 + 35]
                topo_y_l93_small_domain = topo_y_l93[400 - 39:400 + 40]

                # Rotate topographies
                if verbose: print('Begin rotate topographies')
                for time_step, angle in enumerate(wind_dir):

                    # Rotate topography
                    rotated_topo_large = self.rotate_topography(topo_HD, angle)
                    rotated_topo = rotated_topo_large[400 - 39:400 + 40, 400 - 34:400 + 35]

                    if plot and (time_step == random_index) and (idx_station == random_station_idx):
                        plt.figure()
                        plt.contourf(topo_x_l93, topo_y_l93, rotated_topo_large)
                        plt.colorbar()
                        plt.title('Rotated topo large')
                        plt.figure()
                        plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain, rotated_topo)
                        plt.colorbar()
                        plt.title('Rotated topo')

                    # Reshape
                    rotated_topo = rotated_topo.reshape((1, self.n_rows, self.n_col, 1))

                    # Concatenate
                    if time_step == 0:
                        all_time_steps = rotated_topo
                    else:
                        all_time_steps = np.concatenate((all_time_steps, rotated_topo), axis=0)

                # Reshape
                rotated_topo = all_time_steps
                rotated_topo = rotated_topo.reshape((1, nb_sim, self.n_rows, self.n_col, 1))
                wind_speed = wind_speed.reshape((1, nb_sim))
                wind_dir = wind_dir.reshape((1, nb_sim))
                topo_HD_i = topo_HD[400 - 39:400 + 40, 400 - 34:400 + 35].reshape((1, self.n_rows, self.n_col))
                topo_x_l93_small_domain_reshaped = topo_x_l93_small_domain.reshape((1, self.n_col))
                topo_y_l93_small_domain_reshaped = topo_y_l93_small_domain.reshape((1, self.n_rows))

                # Concatenate
                if idx_station == 0:
                    rotated_topo_all_stations = rotated_topo
                    wind_speed_all = wind_speed
                    wind_dir_all = wind_dir
                    all_topo_HD = topo_HD_i
                    all_topo_x_small_l93 = topo_x_l93_small_domain_reshaped
                    all_topo_y_small_l93 = topo_y_l93_small_domain_reshaped
                else:
                    rotated_topo_all_stations = np.concatenate((rotated_topo_all_stations, rotated_topo), axis=0)
                    wind_speed_all = np.concatenate((wind_speed_all, wind_speed), axis=0)
                    wind_dir_all = np.concatenate((wind_dir_all, wind_dir), axis=0)
                    all_topo_HD = np.concatenate((all_topo_HD, topo_HD_i), axis=0)
                    all_topo_x_small_l93 = np.concatenate((all_topo_x_small_l93, topo_x_l93_small_domain_reshaped),
                                                          axis=0)
                    all_topo_y_small_l93 = np.concatenate((all_topo_y_small_l93, topo_y_l93_small_domain_reshaped),
                                                          axis=0)

        # Normalize
        mean, std = self.load_norm_prm()
        topo_norm = self.normalize_topo(rotated_topo_all_stations, mean, std)
        if verbose:
            print('Normalize done')
            print('Mean: ' + str(np.round(mean)))
            print('Mean: ' + str(np.round(std)))

        if plot:
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         topo_norm[random_station_idx, random_index, :, :, 0])
            plt.colorbar()
            plt.title('Normalized topo (rotated)')

        # Reshape for tensorflow
        topo_all = topo_norm.reshape((nb_station * nb_sim, self.n_rows, self.n_col, 1))
        if verbose: print('Reshaped tensorflow done')
        print(topo_all.shape)

        # Prediction
        """
        Warning: change dependencies here
        """
        self.load_model(dependencies=True)
        prediction = self.model.predict(topo_all)
        if verbose: print('Prediction done')

        # Reshape predictions for analysis
        prediction = prediction.reshape((nb_station, nb_sim, self.n_rows, self.n_col, 3))
        if verbose:
            print('Prediction reshaped')
            print(prediction.shape)

        if plot:
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         prediction[random_station_idx, random_index, :, :, 0])
            plt.colorbar()
            plt.title('U not scaled (rotated)')
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         prediction[random_station_idx, random_index, :, :, 1])
            plt.colorbar()
            plt.title('V not scaled (rotated)')
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         prediction[random_station_idx, random_index, :, :, 2])
            plt.colorbar()
            plt.title('W not scaled (rotated)')

        # Wind speed scaling by broadcasting
        wind_speed = wind_speed_all.reshape((nb_station, nb_sim, 1, 1, 1))
        wind_dir = wind_dir_all.reshape((nb_station, nb_sim, 1, 1, 1))
        prediction_scaled = wind_speed * prediction / 3
        if verbose: print('Wind speed scaling done')
        if plot:
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         prediction_scaled[random_station_idx, random_index, :, :, 0])
            plt.colorbar()
            plt.title('U scaled (rotated)')
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         prediction_scaled[random_station_idx, random_index, :, :, 1])
            plt.colorbar()
            plt.title('V scaled (rotated)')
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         prediction_scaled[random_station_idx, random_index, :, :, 2])
            plt.colorbar()
            plt.title('W scaled (rotated)')

        # Wind computations
        U_rot = prediction_scaled[:, :, :, :, 0]  # Expressed in the rotated coord. system
        V_rot = prediction_scaled[:, :, :, :, 1]  # Expressed in the rotated coord. system
        W_rot = prediction_scaled[:, :, :, :, 2]  # Good coord. but not on the right pixel
        UV_old = np.sqrt(U_rot ** 2 + V_rot ** 2)  # Good coord. but not on the right pixel
        UVW_old = np.sqrt(U_rot ** 2 + V_rot ** 2 + W_rot ** 2)  # Good coord. but not on the right pixel
        alpha = np.where(U_rot == 0,
                         np.where(V_rot == 0, 0, np.sign(V_rot) * np.pi / 2),
                         np.arctan(V_rot / U_rot))  # Expressed in the rotated coord. system

        if plot:
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         UV_old[random_station_idx, random_index, :, :])
            plt.colorbar()
            plt.title('UV_old (rotated)')
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         UVW_old[random_station_idx, random_index, :, :])
            plt.colorbar()
            plt.title('UVW_old (rotated)')
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         alpha[random_station_idx, random_index, :, :])
            plt.colorbar()
            plt.title('alpha (rotated)')

        # Reshape for broadcasting
        wind_speed = wind_speed_all.reshape((nb_station, nb_sim, 1, 1))
        wind_dir = wind_dir_all.reshape((nb_station, nb_sim, 1, 1))
        UV_DIR_rad_old = (np.pi / 180) * wind_dir - alpha  # Good coord. but not on the right pixel

        assert UV_old.shape == UV_DIR_rad_old.shape

        U_old = -np.sin(UV_DIR_rad_old) * UV_old  # Good coord. but not on the right pixel
        V_old = -np.cos(UV_DIR_rad_old) * UV_old  # Good coord. but not on the right pixel
        W_old = W_rot  # Good coord. but not on the right pixel
        alpha_old = alpha  # Good coord. but not on the right pixel
        if verbose: print('Wind computation done')
        if plot:
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         U_old[random_station_idx, random_index, :, :])
            plt.colorbar()
            plt.title('U_old (rotated)')
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         UVW_old[random_station_idx, random_index, :, :])
            plt.colorbar()
            plt.title('V_old (rotated)')
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         alpha[random_station_idx, random_index, :, :])
            plt.colorbar()
            plt.title('W_old (rotated)')
        # Rotate clockwise to put the wind value on the right topography pixel
        if verbose: print('Start rotating to initial position')
        for idx_station in range(nb_station):
            for time_step in range(nb_sim):
                U_i = self.rotate_topography(U_old[idx_station, time_step, :, :],
                                             wind_dir_all[idx_station, time_step],
                                             clockwise=True)
                V_i = self.rotate_topography(V_old[idx_station, time_step, :, :],
                                             wind_dir_all[idx_station, time_step],
                                             clockwise=True)
                W_i = self.rotate_topography(W_old[idx_station, time_step, :, :],
                                             wind_dir_all[idx_station, time_step],
                                             clockwise=True)
                UV_i = self.rotate_topography(UV_old[idx_station, time_step, :, :],
                                              wind_dir_all[idx_station, time_step],
                                              clockwise=True)
                UVW_i = self.rotate_topography(UVW_old[idx_station, time_step, :, :],
                                               wind_dir_all[idx_station, time_step],
                                               clockwise=True)

                UV_DIR_rad_i = self.rotate_topography(UV_DIR_rad_old[idx_station, time_step, :, :],
                                                      wind_dir_all[idx_station, time_step],
                                                      clockwise=True)

                alpha_i = self.rotate_topography(alpha_old[idx_station, time_step, :, :],
                                                 wind_dir_all[idx_station, time_step],
                                                 clockwise=True)
                # Concatenate
                if (time_step == 0) and (idx_station == 0):
                    U = U_i
                    V = V_i
                    W = W_i
                    UV = UV_i
                    UVW = UVW_i
                    UV_DIR_rad = UV_DIR_rad_i
                    alpha = alpha_i
                else:
                    U = np.concatenate((U, U_i), axis=0)
                    V = np.concatenate((V, V_i), axis=0)
                    W = np.concatenate((W, W_i), axis=0)
                    UV = np.concatenate((UV, UV_i), axis=0)
                    UVW = np.concatenate((UVW, UVW_i), axis=0)
                    UV_DIR_rad = np.concatenate((UV_DIR_rad, UV_DIR_rad_i), axis=0)
                    alpha = np.concatenate((alpha, alpha_i), axis=0)

        if verbose: print('Wind prediction rotated for initial topography')

        # Reshape
        U = U.reshape((nb_station, nb_sim, self.n_rows, self.n_col))
        V = V.reshape((nb_station, nb_sim, self.n_rows, self.n_col))
        W = W.reshape((nb_station, nb_sim, self.n_rows, self.n_col))
        UV = UV.reshape((nb_station, nb_sim, self.n_rows, self.n_col))
        UVW = UVW.reshape((nb_station, nb_sim, self.n_rows, self.n_col))
        UV_DIR_rad = UV_DIR_rad.reshape((nb_station, nb_sim, self.n_rows, self.n_col))
        alpha = alpha.reshape((nb_station, nb_sim, self.n_rows, self.n_col))
        wind_speed = wind_speed_all.reshape((nb_station, nb_sim))
        wind_dir = wind_dir_all.reshape((nb_station, nb_sim))
        if verbose: print('Reshape final predictions done')

        # Convert radians to degree
        UV_DIR_deg = (180 / np.pi) * UV_DIR_rad
        alpha_deg = (180 / np.pi) * alpha

        # Store results
        array_xr = xr.Dataset(data_vars={"U": (["station", "time", "y", "x"], U),
                                         "V": (["station", "time", "y", "x"], V),
                                         "W": (["station", "time", "y", "x"], W),
                                         "UV": (["station", "time", "y", "x"], UV),
                                         "UVW": (["station", "time", "y", "x"], UVW),
                                         "UV_DIR_rad": (["station", "time", "y", "x"], UV_DIR_rad),
                                         "UV_DIR_deg": (["station", "time", "y", "x"], UV_DIR_deg),
                                         "alpha": (["station", "time", "y", "x"], alpha),
                                         "alpha_deg": (["station", "time", "y", "x"], alpha_deg),
                                         "NWP_wind_speed": (["station", "time"], wind_speed),
                                         "NWP_wind_DIR": (["station", "time"], wind_dir),
                                         "ZS_mnt": (["station", "y", "x"], all_topo_HD,),
                                         "XX": (["station", "x"], all_topo_x_small_l93,),
                                         "YY": (["station", "y"], all_topo_y_small_l93,),

                                         },

                              coords={"station": np.array(stations_name),
                                      "time": np.array(time_xr),
                                      "x": np.array(list(range(self.n_col))),
                                      "y": np.array(list(range(self.n_rows)))})

        if plot:
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         U[random_station_idx, random_index, :, :])
            plt.colorbar()
            plt.title('U')
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         V[random_station_idx, random_index, :, :])
            plt.colorbar()
            plt.title('V')
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         W[random_station_idx, random_index, :, :])
            plt.colorbar()
            plt.title('W')
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         UVW[random_station_idx, random_index, :, :])
            plt.colorbar()
            plt.title('UVW')
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         UV_DIR_deg[random_station_idx, random_index, :, :])
            plt.colorbar()
            plt.title('UV_DIR_deg')
            plt.figure()
            plt.contourf(topo_x_l93_small_domain, topo_y_l93_small_domain,
                         alpha_deg[random_station_idx, random_index, :, :])
            plt.colorbar()
            plt.title('alpha_deg')
            print(f"NWP wind speed: {wind_speed[random_station_idx, random_index]}")
            print(f"NWP wind direction: {wind_dir[random_station_idx, random_index]}")

        return (array_xr)

    def predict_ideal_conditions(self, stations_name, wind_direction, wind_speed, visualization=True):

        MNT_data, MNT_x, MNT_y = self.observation.extract_MNT_around_station(stations_name,
                                                                             self.mnt,
                                                                             400,
                                                                             400)
        if visualization:
            plt.figure()
            plt.contourf(MNT_x, MNT_y, MNT_data)
            plt.title("Initial topo")

        # Rotate topography
        rotated_topo_large = self.rotate_topography(MNT_data, wind_direction)

        if visualization:
            plt.figure()
            plt.contourf(MNT_x, MNT_y, rotated_topo_large)
            plt.title("Large rotated topo")

        rotated_topo = rotated_topo_large[400 - 39:400 + 40, 400 - 34:400 + 35]

        if visualization:
            plt.figure()
            plt.contourf(MNT_x, MNT_y, rotated_topo_large)
            print(rotated_topo.shape)
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], rotated_topo, cmap='Reds')
            plt.title("Small rotated topo")

        # Reshape
        rotated_topo = rotated_topo.reshape((1, self.n_rows, self.n_col, 1))

        # Normalize
        mean, std = self.load_norm_prm()
        topo_norm = self.normalize_topo(rotated_topo, mean, std)
        print('Normalize done')
        print('Mean: ' + str(np.round(mean)))
        print('Mean: ' + str(np.round(std)))

        if visualization:
            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], topo_norm[0, :, :, 0], cmap='Reds')
            plt.colorbar()
            plt.title("Normalized topo")

        # Prediction
        """
        Warning: change dependencies here
        """
        self.load_model(dependencies=True)
        prediction = self.model.predict(topo_norm)

        if visualization:
            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40],
                         np.sqrt(prediction[0, :, :, 0] ** 2 + prediction[0, :, :, 1] ** 2),
                         cmap='bwr', norm=MidpointNormalize(midpoint=wind_speed))
            plt.colorbar()
            plt.title("Prediction not scaled")

        # Wind speed scaling
        prediction_scaled = wind_speed * prediction / 3

        if visualization:
            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40],
                         np.sqrt(prediction_scaled[0, :, :, 0] ** 2 + prediction_scaled[0, :, :, 1] ** 2),
                         cmap='bwr', norm=MidpointNormalize(midpoint=wind_speed))
            plt.colorbar()
            plt.title("Prediction scaled")

        # Wind computations
        U_rot = prediction_scaled[:, :, :, 0]
        V_rot = prediction_scaled[:, :, :, 1]
        W_rot = prediction_scaled[:, :, :, 2]
        UV_old = np.sqrt(U_rot ** 2 + V_rot ** 2)
        UVW_old = np.sqrt(U_rot ** 2 + V_rot ** 2 + W_rot ** 2)
        alpha = np.where(U_rot == 0,
                         np.where(V_rot == 0, 0, np.sign(V_rot) * np.pi / 2),
                         np.arctan(V_rot / U_rot))
        UV_DIR_rad = (np.pi / 180) * wind_direction - alpha[:, :, :]

        if visualization:
            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], alpha[0, :, :],
                         cmap='bwr', norm=MidpointNormalize(midpoint=0))
            plt.colorbar()
            plt.title("Alpha")

            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], (180 / np.pi) * alpha[0, :, :],
                         cmap='bwr', norm=MidpointNormalize(midpoint=0))
            plt.colorbar()
            plt.title("Alpha_DEG")

            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], UVW_old[0, :, :],
                         cmap='bwr', norm=MidpointNormalize(midpoint=wind_speed))
            plt.colorbar()
            plt.title("UVW_old")

            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], U_rot[0, :, :],
                         cmap='bwr', norm=MidpointNormalize(midpoint=wind_speed))
            plt.colorbar()
            plt.title("U_rot")

            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], V_rot[0, :, :],
                         cmap='bwr', norm=MidpointNormalize(midpoint=0))
            plt.colorbar()
            plt.title("V_rot")

            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], W_rot[0, :, :],
                         cmap='bwr', norm=MidpointNormalize(midpoint=0))
            plt.colorbar()
            plt.title("W_rot")

            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], UV_DIR_rad[0, :, :],
                         cmap='bwr', norm=MidpointNormalize(midpoint=(np.pi / 180) * wind_direction))
            plt.colorbar()
            plt.title("UV_DIR_rad")

            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], (180 / np.pi) * UV_DIR_rad[0, :, :] % 360,
                         cmap='bwr', norm=MidpointNormalize(midpoint=wind_direction))
            plt.colorbar()
            plt.title("UV_DIR_deg")

        assert UV_old.shape == UV_DIR_rad.shape

        U_old = -np.sin(UV_DIR_rad) * UV_old
        V_old = -np.cos(UV_DIR_rad) * UV_old
        W_old = W_rot

        # Rotate clockwise to put the wind value on the right topography pixel
        U = self.rotate_topography(U_old[0, :, :], wind_direction, clockwise=True)
        V = self.rotate_topography(V_old[0, :, :], wind_direction, clockwise=True)
        W = self.rotate_topography(W_old[0, :, :], wind_direction, clockwise=True)
        UVW = self.rotate_topography(UVW_old[0, :, :], wind_direction, clockwise=True)

        if visualization:
            plt.colorbar()
            plt.title("U")

            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], U_old[0, :, :],
                         cmap='bwr', norm=MidpointNormalize(midpoint=0))
            plt.colorbar()
            plt.title("U_old")

            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], V_old[0, :, :],
                         cmap='bwr', norm=MidpointNormalize(midpoint=0))
            plt.colorbar()
            plt.title("V_old")

            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], W_old[0, :, :],
                         cmap='bwr', norm=MidpointNormalize(midpoint=wind_speed))
            plt.colorbar()
            plt.title("W_old")

            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], U[:, :],
                         cmap='bwr', norm=MidpointNormalize(midpoint=wind_speed))
            plt.colorbar()
            plt.title("U")

            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], V[:, :],
                         cmap='bwr', norm=MidpointNormalize(midpoint=0))
            plt.colorbar()
            plt.title("V")

            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], W[:, :],
                         cmap='bwr', norm=MidpointNormalize(midpoint=0))
            plt.colorbar()
            plt.title("W")

            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], UVW[:, :],
                         cmap='bwr', norm=MidpointNormalize(midpoint=wind_speed))
            plt.colorbar()
            plt.title("UVW")

            plt.figure()
            plt.contourf(MNT_x[400 - 34:400 + 35], MNT_y[400 - 39:400 + 40], UVW[:, :] / wind_speed,
                         cmap='bwr', norm=MidpointNormalize(midpoint=1))
            plt.colorbar()
            plt.title("Acceleration ratio")

        self.U_ideal_test = U
        self.V_ideal_test = V
        self.W_ideal_test = W
        self.UVW_ideal_test = UVW
        print('Done')
        print('Not saved on disk')


class Visualization:

    def __init__(self, p):
        self.p = p
        self.l93 = ccrs.epsg(2154)

    def _plot_observation_station(self):
        ax = plt.gca()
        self.p.observation.stations_to_gdf(from_epsg=self.l93, x="X", y="Y")
        self.p.observation.stations.plot(ax=ax, markersize=1, color='C3', label='observation stations')

    def _plot_station_names(self):
        # Plot stations name
        ax = plt.gca()
        stations = self.p.observation.stations
        nb_stations = len(stations['X'].values)
        for idx_station in range(nb_stations):
            X = list(stations['X'].values)[idx_station]
            Y = list(stations['Y'].values)[idx_station]

            ax.text(X, Y, list(stations['name'].values)[idx_station])

    def plot_area(self):
        fig = plt.figure()

        ax = fig.add_subplot(111, projection=self.l93)

        ax.coastlines(resolution='10m', alpha=0.1)
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        ax.set_xlim(207_775, 1_207_775)
        ax.set_ylim(6_083_225.0, 6_783_225.0)

        MNT_polygon = self.polygon_from_grid(self.p.mnt.data_xr.x.data, self.p.mnt.data_xr.y.data)
        NWP_polygon = self.polygon_from_grid(self.p.nwp.data_xr["X_L93"], self.p.nwp.data_xr["Y_L93"])

        ax.plot(*MNT_polygon.exterior.xy, label='MNT')
        ax.plot(*NWP_polygon.exterior.xy, label='NWP')

        self._plot_observation_station()

        plt.legend()
        plt.show()

    def plot_nwp_grid(self):

        self.plot_area()

        # Plot AROME grid
        ax = plt.gca()
        x_l93, y_l93 = self.p.nwp.data_xr["X_L93"], self.p.nwp.data_xr["Y_L93"]
        stacked_xy = self.p.mnt.x_y_to_stacked_xy(x_l93, y_l93)
        x_y_flat = self.p.mnt.grid_to_flat(stacked_xy)
        NWP_flat_gpd = gpd.GeoDataFrame(geometry=[Point(cell) for cell in x_y_flat],
                                        crs=self.l93)
        NWP_flat_gpd.plot(ax=ax, markersize=5, label='NWP grid')

        plt.legend()
        plt.show()

    def plot_KNN_NWP(self, number_of_neighbors=4):

        # Plot NWP grid
        self.plot_nwp_grid()

        # Plot NWP nearest neighbors to stations
        ax = plt.gca()
        self.p.observation.update_stations_with_KNN_from_NWP(number_of_neighbors, self.p.nwp)
        nwp_neighbors = self.p.observation.stations.copy()
        nwp_name = self.p.nwp.name
        for neighbor in range(number_of_neighbors):
            geometry_knn_i = gpd.GeoDataFrame(geometry=nwp_neighbors[f'{nwp_name}_NN_{neighbor}'])
            geometry_knn_i.plot(ax=ax, label=f'{nwp_name}_NN_{neighbor}')

        # Plot station names
        self._plot_station_names()

        plt.legend()
        plt.show()

    def plot_KNN_MNT(self, number_of_neighbors=4):
        self.plot_KNN_NWP()

        # Plot MNT nearest neighbors to stations
        ax = plt.gca()
        self.p.observation.update_stations_with_KNN_from_MNT_using_cKDTree(self.p.mnt)
        mnt_neighbors = self.p.observation.stations.copy()
        mnt_name = self.p.mnt.name
        for neighbor in range(number_of_neighbors):
            geometry_knn_i = gpd.GeoDataFrame(geometry=mnt_neighbors[f'{mnt_name}_NN_{neighbor}_cKDTree'].apply(Point))
            geometry_knn_i.plot(ax=ax, label=f'{mnt_name}_NN_{neighbor}_cKDTree')
        plt.legend()

    def _plot_single_observation_station(self, station_name):
        ax = plt.gca()
        stations = self.p.observation.stations
        stations = stations[stations["name"] == station_name]
        ax.plot(stations['X'].values[0],
                stations['Y'].values[0],
                marker='x',
                label='observation station',
                color="black")

    def _plot_single_NWP_nearest_neighbor(self, station_name):
        ax = plt.gca()
        stations = self.p.observation.stations
        stations = stations[stations["name"] == station_name]
        nwp_name = self.p.nwp.name
        ax.plot(stations[f"{nwp_name}_NN_0"].values[0].x,
                stations[f"{nwp_name}_NN_0"].values[0].y,
                marker='x',
                label=f"{nwp_name}_NN_0",
                color="C0")

    def _plot_single_MNT_nearest_neighbor(self, station_name):
        ax = plt.gca()
        stations = self.p.observation.stations
        stations = stations[stations["name"] == station_name]
        mnt_name = self.p.mnt.name
        ax.plot(stations[f"{mnt_name}_NN_0_cKDTree"].values[0][0],
                stations[f"{mnt_name}_NN_0_cKDTree"].values[0][1],
                marker='x',
                label=f"{mnt_name}_NN_0_cKDTree",
                color="C1")

    def _plot_single_station_name(self, station_name):
        ax = plt.gca()
        stations = self.p.observation.stations
        stations = stations[stations["name"] == station_name]
        X = stations['X'].values[0]
        Y = stations['Y'].values[0]
        name = stations['name'].values[0]
        ax.text(X, Y, name)

    def plot_topography_around_station_2D(self, station_name, nb_pixel_x=100, nb_pixel_y=100, create_figure=True):
        MNT_data, MNT_x, MNT_y = self.p.observation.extract_MNT_around_station(station_name,
                                                                               self.p.mnt,
                                                                               nb_pixel_x,
                                                                               nb_pixel_y)
        if create_figure:
            plt.figure(figsize=(20, 20))
        ax = plt.gca()

        # Plot topography
        plt.contourf(MNT_x, MNT_y, MNT_data, cmap='gist_earth')
        plt.colorbar()

        # Plot observation station
        self._plot_single_observation_station(station_name)

        # Plot NWP neighbor
        self._plot_single_NWP_nearest_neighbor(station_name)

        # Plot MNT neighbor
        self._plot_single_MNT_nearest_neighbor(station_name)

        plt.legend()
        plt.title(station_name)

    def plot_multiple_topographies_2D(self, centered_station_name, nb_pixel_x=1000, nb_pixel_y=1000):

        MNT_data, MNT_x, MNT_y = self.p.observation.extract_MNT_around_station(centered_station_name,
                                                                               self.p.mnt,
                                                                               nb_pixel_x,
                                                                               nb_pixel_y)
        plt.contourf(MNT_x, MNT_y, MNT_data, cmap='gist_earth')
        plt.colorbar()

        min_x = np.min(MNT_x)
        max_x = np.max(MNT_x)
        min_y = np.min(MNT_y)
        max_y = np.max(MNT_y)
        stations = self.p.observation.stations
        for station_name in stations["name"].values:
            station = stations[stations["name"] == station_name]
            if (min_x < station["X"].values[0] < max_x) and (min_y < station["Y"].values[0] < max_y):
                # Plot observation station
                self._plot_single_observation_station(station_name)

                # Plot NWP neighbor
                self._plot_single_NWP_nearest_neighbor(station_name)

                # Plot MNT neighbor
                self._plot_single_MNT_nearest_neighbor(station_name)

                # Plot station name
                self._plot_single_station_name(station_name)

        plt.legend(('Observation station', 'NWP NN', 'MNT NN'))
        plt.title(centered_station_name)

    def plot_rotated_topo(self, station_name, wind_direction, type="large", nb_pixel_large_x=400, nb_pixel_large_y=400):
        MNT_data, MNT_x, MNT_y = self.p.observation.extract_MNT_around_station(station_name,
                                                                               self.p.mnt,
                                                                               nb_pixel_large_x,
                                                                               nb_pixel_large_y)
        # Initial topography
        plt.contourf(MNT_x, MNT_y, MNT_data, cmap='gist_earth')

        # Rotate topography
        rotated_topo_large = self.p.rotate_topography(MNT_data, wind_direction)
        rotated_topo = rotated_topo_large[nb_pixel_large_x // 2 - 39:nb_pixel_large_x // 2 + 40,
                       nb_pixel_large_y // 2 - 34:nb_pixel_large_y // 2 + 35]
        if type == "large":
            plt.contourf(MNT_x, MNT_y, rotated_topo_large)
        if type == "small":
            plt.contourf(MNT_x, MNT_y, rotated_topo)
        if type == "both":
            plt.contourf(MNT_x, MNT_y, rotated_topo, cmap='viridis')
            plt.contourf(MNT_x, MNT_y, rotated_topo_large, cmap='cividis')
        plt.colorbar()
        plt.title(f"Wind direction: {wind_direction}")

    def _plot_arrow_for_NWP(self, array_xr, time_index, station_name='Col du Lac Blanc'):
        # Arrow for NWP wind
        ax = plt.gca()
        stations = self.p.observation.stations
        nwp_name = self.p.nwp.name
        point_nwp = stations[f"{nwp_name}_NN_0"][stations["name"] == station_name].values[0]
        x_nwp = point_nwp.x
        y_nwp = point_nwp.y

        NWP_wind_speed = array_xr.NWP_wind_speed.sel(station=station_name).isel(time=time_index).data
        NWP_wind_DIR = array_xr.NWP_wind_DIR.sel(station=station_name).isel(time=time_index).data
        U_nwp = -np.sin((np.pi / 180) * NWP_wind_DIR) * NWP_wind_speed
        V_nwp = -np.cos((np.pi / 180) * NWP_wind_DIR) * NWP_wind_speed

        ax.quiver(x_nwp, y_nwp, U_nwp, V_nwp, color='red')
        ax.text(x_nwp - 100, y_nwp + 100, str(np.round(NWP_wind_speed, 1)) + " m/s", color='red')
        ax.text(x_nwp + 100, y_nwp - 100, str(np.round(NWP_wind_DIR)) + '°', color='red')
        ax.axis('equal')

    def _plot_arrow_for_observation_station(self, array_xr, time_index, station_name='Col du Lac Blanc'):
        # Arrow for station wind
        ax = plt.gca()
        stations = self.p.observation.stations
        time_series = self.p.observation.time_series
        x_observation = stations["X"][stations["name"] == station_name].values[0]
        y_observation = stations["Y"][stations["name"] == station_name].values[0]
        dates = pd.DatetimeIndex([array_xr.time.data[time_index]])
        time_condition = (time_series.index.year == dates.year.values[0]) \
                         & (time_series.index.month == dates.month.values[0]) \
                         & (time_series.index.day == dates.day.values[0]) \
                         & (time_series.index.hour == dates.hour.values[0])
        try:
            observed_UV = time_series['vw10m(m/s)'][(time_series['name'] == station_name) & time_condition].values[0]
            observed_UV_DIR = \
                time_series['winddir(deg)'][(time_series['name'] == station_name) & time_condition].values[0]
        except:
            return ()

        U_observed = -np.sin((np.pi / 180) * observed_UV_DIR) * observed_UV
        V_observed = -np.cos((np.pi / 180) * observed_UV_DIR) * observed_UV
        ax.quiver(x_observation, y_observation, U_observed, V_observed, color='black')

    def _plot_arrows_for_CNN(self, array_xr, time_index, station_name='Col du Lac Blanc'):
        ax = plt.gca()
        U = array_xr.U.sel(station=station_name).isel(time=time_index).data
        V = array_xr.V.sel(station=station_name).isel(time=time_index).data
        UV = array_xr.UV.sel(station=station_name).isel(time=time_index).data
        NWP_wind_speed = array_xr.NWP_wind_speed.sel(station=station_name).isel(time=time_index).data

        x = array_xr["XX"].sel(station=station_name).data
        y = array_xr["YY"].sel(station=station_name).data
        XX, YY = np.meshgrid(x, y)

        arrows = ax.quiver(XX, YY, U, V, UV,
                           scale=1 / 0.005,
                           cmap='coolwarm',
                           norm=MidpointNormalize(midpoint=NWP_wind_speed))
        plt.colorbar(arrows, orientation='horizontal')

    def plot_predictions_2D(self, stations_name=['Col du Lac Blanc'], array_xr=None):

        if array_xr is None:
            array_xr = self.p.predict_UV_with_CNN(self, stations_name)

        random_time_idx = random.choice(list(range(len(array_xr.time.data))))
        random_station_name = random.choice(stations_name)

        # Plot topography
        self.plot_topography_around_station_2D(random_station_name)

        # NWP wind speed and direction
        nwp_speed = array_xr.NWP_wind_speed.sel(station=random_station_name).isel(time=random_time_idx).data
        nwp_DIR = array_xr.NWP_wind_DIR.sel(station=random_station_name).isel(time=random_time_idx).data
        print(f"NWP wind speed: {nwp_speed}")
        print(f"NWP wind direction: {nwp_DIR}")

        # Experience time
        time = array_xr.time.data[random_time_idx]

        # l93 coordinates
        XX_station = array_xr["XX"].sel(station=random_station_name).data
        YY_station = array_xr["YY"].sel(station=random_station_name).data

        # Plot results
        for variable in ["U", "V", "W", "UV", "UVW", "UV_DIR_deg", "alpha_deg"]:

            # Define midpoint for color normalization
            if variable in ["U", "UV", "UVW"]:
                midpoint = nwp_speed
            elif variable in ["V", "W", "alpha_deg"]:
                midpoint = 0
            else:
                midpoint = nwp_DIR

            plt.figure()
            display = array_xr[variable].sel(station=random_station_name).isel(time=random_time_idx).data
            plt.contourf(XX_station, YY_station, display, cmap='bwr', norm=MidpointNormalize(midpoint=midpoint))
            plt.title(variable + '\n' + random_station_name + '\n' + str(time))
            plt.colorbar()
            ax = plt.gca()
            ax.axis('equal')

            if variable in ["UV", "UVW"]:
                self._plot_arrow_for_NWP(array_xr, random_time_idx, station_name=random_station_name)
                self._plot_arrow_for_observation_station(array_xr, random_time_idx, station_name=random_station_name)
                self._plot_arrows_for_CNN(array_xr, random_time_idx, station_name=random_station_name)

    def _plot_3D_topography(self, XX, YY, ZZ, station_name, alpha=1, figsize=(30, 30)):
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection='3d')
        im = ax.plot_surface(XX, YY, ZZ, cmap="gist_earth", lw=0.5, rstride=1, cstride=1, alpha=alpha)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(im)
        plt.title("Topography [m]" + '\n' + station_name)
        self._set_axes_equal(ax)
        plt.show()

    def plot_3D_topography_array_xr(self, array_xr, station_name='Col du Lac Blanc', alpha=1):
        """
        Display the topographic map of the area in 3D

        Input:
                array_xr : file with predictions
                station_name : ex: 'Col du Lac Blanc'
        """

        x = array_xr["XX"].sel(station=station_name).data
        y = array_xr["YY"].sel(station=station_name).data
        XX, YY = np.meshgrid(x, y)
        ZZ = array_xr["ZS_mnt"].sel(station=station_name).data

        self._plot_3D_topography(XX, YY, ZZ, station_name, alpha=alpha)

    def plot_3D_topography_NWP(self, station_name='Col du Lac Blanc', length_x_km=30, length_y_km=30, alpha=1):
        """
        Display the topographic map of the area in 3D

        Input:
                array_xr : file with predictions
                station_name : ex: 'Col du Lac Blanc'
        """
        stations = self.p.observation.stations
        nwp_name = self.p.nwp.name
        idx_nwp_x, idx_nwp_y = stations[f"index_{nwp_name}_NN_0"][stations["name"] == station_name].values[0]

        shift_x = length_x_km // 1.3
        shift_y = length_y_km // 1.3

        XX = self.p.nwp.data_xr.X_L93.data[int(idx_nwp_x - shift_x):int(idx_nwp_x + shift_x), int(idx_nwp_y - shift_y):int(idx_nwp_y + shift_y)]
        YY = self.p.nwp.data_xr.Y_L93.data[int(idx_nwp_x - shift_x):int(idx_nwp_x + shift_x), int(idx_nwp_y - shift_y):int(idx_nwp_y + shift_y)]
        ZZ = self.p.nwp.data_xr.ZS.data[int(idx_nwp_x - shift_x):int(idx_nwp_x + shift_x), int(idx_nwp_y - shift_y):int(idx_nwp_y + shift_y)]

        self._plot_3D_topography(XX, YY, ZZ, station_name, alpha=alpha)

    def plot_3D_topography_large_domain(self, station_name='Col du Lac Blanc', nb_pixel_x=400, nb_pixel_y=400):
        """
        Display the topographic map of a large area in 3D
        """

        MNT_data, MNT_x, MNT_y = self.p.observation.extract_MNT_around_station(station_name, self.p.mnt, nb_pixel_x,
                                                                               nb_pixel_y)
        XX, YY = np.meshgrid(MNT_x, MNT_y)
        self._plot_3D_topography(XX, YY, MNT_data, station_name)

    def plot_comparison_topography_MNT_NWP(self, station_name='Col du Lac Blanc', length_x_km=10, length_y_km=10, alpha=1):
        self.plot_3D_topography_NWP(station_name=station_name, length_x_km=length_x_km, length_y_km=length_y_km,
                                    alpha=alpha)

        # Converting km to number of pixel for mnt
        nb_pixel_x = int(length_x_km*1000 // self.p.mnt.resolution_x)
        nb_pixel_y = int(length_y_km*1000 // self.p.mnt.resolution_y)

        self.plot_3D_topography_large_domain(station_name=station_name, nb_pixel_x=nb_pixel_x, nb_pixel_y=nb_pixel_y)

    def _plot_3D_variable(self, XX, YY, ZZ, variable, display, time, midpoint, station_name, alpha=1,
                          rstride=1, cstride=1):

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        color_dimension = display
        minn, maxx = np.nanmin(color_dimension), np.nanmax(color_dimension)
        norm = colors.Normalize(minn, maxx)
        m = plt.cm.ScalarMappable(norm=MidpointNormalize(minn, maxx, midpoint), cmap="bwr")
        m.set_array([])
        fcolors = m.to_rgba(color_dimension)
        ax.plot_surface(XX, YY, ZZ,
                        cmap="bwr",
                        facecolors=fcolors,
                        linewidth=0,
                        norm=MidpointNormalize(midpoint=midpoint),
                        shade=False,
                        alpha=alpha,
                        rstride=rstride,
                        cstride=cstride)
        cb = plt.colorbar(m)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("Wind predictions with CNN" + '\n' + variable + '\n' + station_name + '\n' + str(time))
        self._set_axes_equal(ax)

    def plot_predictions_3D(self, stations_name=['Col du Lac Blanc'], array_xr=None):
        """
        3D maps of wind speed (color) on topography (3D data)
        Input:
        Output: 3D Map winds
        """

        # Select station and timestep
        random_time_idx = random.choice(list(range(len(array_xr.time.data))))
        random_station_name = random.choice(stations_name)

        # NWP wind speed and direction
        nwp_speed = array_xr.NWP_wind_speed.sel(station=random_station_name).isel(time=random_time_idx).data
        nwp_DIR = array_xr.NWP_wind_DIR.sel(station=random_station_name).isel(time=random_time_idx).data
        print(f"NWP wind speed: {nwp_speed}")
        print(f"NWP wind direction: {nwp_DIR}")

        # Experience time
        time = array_xr.time.data[random_time_idx]

        # Select topography
        x = array_xr["XX"].sel(station=random_station_name).data
        y = array_xr["YY"].sel(station=random_station_name).data
        XX, YY = np.meshgrid(x, y)
        ZZ = array_xr["ZS_mnt"].sel(station=random_station_name).data

        # Select variable
        for variable in ["U", "V", "W", "UV", "UVW", "UV_DIR_deg", "alpha_deg"]:

            # Define midpoint for color normalization
            if variable in ["U", "UV", "UVW"]:
                midpoint = nwp_speed
            elif variable in ["V", "W", "alpha_deg"]:
                midpoint = 0
            else:
                midpoint = nwp_DIR

            display = array_xr[variable].sel(station=random_station_name).isel(time=random_time_idx).data
            self._plot_3D_variable(XX, YY, ZZ, variable, display, time, midpoint, random_station_name)
            if variable in ["U", "UV", "UVW"]:
                self._plot_arrow_for_NWP_3D(array_xr, random_time_idx, station_name=random_station_name)

        U = array_xr["U"].sel(station=random_station_name).isel(time=random_time_idx).data
        V = array_xr["V"].sel(station=random_station_name).isel(time=random_time_idx).data
        W = array_xr["W"].sel(station=random_station_name).isel(time=random_time_idx).data
        UVW = array_xr["UVW"].sel(station=random_station_name).isel(time=random_time_idx).data
        self._plot_3D_variable(XX, YY, ZZ, variable, UVW, time, nwp_speed, random_station_name, alpha=0.5)
        self._plot_arrows_for_CNN_3D(XX, YY, ZZ, U, V, W, None, nwp_speed)

    def _plot_arrows_for_CNN_3D(self, XX, YY, ZZ, U, V, W, variable_color, midpoint, cmap="coolwarm",
                                figsize=(30, 30), colors='black', linewidth=1, length=10,
                                arrow_length_ratio=0.3):
        if variable_color is not None:
            c1 = variable_color
            minn, maxx = np.nanmin(c1), np.nanmax(c1)
            m = plt.cm.ScalarMappable(norm=MidpointNormalize(minn, maxx, midpoint), cmap=cmap)
            c2 = m.to_rgba(UVW.flatten())
            colors = c2

        fig = plt.gcf()
        ax = fig.gca(projection='3d')
        q = ax.quiver(XX, YY, ZZ,
                      U, V, W,
                      colors=colors,
                      linewidth=linewidth,
                      length=length,
                      arrow_length_ratio=arrow_length_ratio,
                      cmap='coolwarm')
        if variable_color is not None:
            plt.colorbar(m)
        self._set_axes_equal(ax)

    def _plot_arrow_for_NWP_3D(self, array_xr, time_index, station_name='Col du Lac Blanc', length=250):
        # Arrow for NWP wind
        fig = plt.gcf()
        ax = fig.gca(projection='3d')
        stations = self.p.observation.stations
        nwp_name = self.p.nwp.name
        point_nwp = stations[f"{nwp_name}_NN_0"][stations["name"] == station_name].values[0]
        x_nwp = point_nwp.x
        y_nwp = point_nwp.y
        index_x_MNT, index_y_MNT = self.p.mnt.find_nearest_MNT_index(x_nwp, y_nwp)
        z_nwp = self.p.mnt.data[int(index_x_MNT), int(index_y_MNT)]

        NWP_wind_speed = array_xr.NWP_wind_speed.sel(station=station_name).isel(time=time_index).data
        NWP_wind_DIR = array_xr.NWP_wind_DIR.sel(station=station_name).isel(time=time_index).data
        U_nwp = -np.sin((np.pi / 180) * NWP_wind_DIR) * NWP_wind_speed
        V_nwp = -np.cos((np.pi / 180) * NWP_wind_DIR) * NWP_wind_speed

        ax.quiver(x_nwp, y_nwp, z_nwp, U_nwp, V_nwp, 0, color='green', length=length)

    @staticmethod
    def polygon_from_grid(grid_x, grid_y):
        xmin, ymin, xmax, ymax = np.min(grid_x), np.min(grid_y), np.max(grid_x), np.max(grid_y)
        polygon = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
        return (polygon)

    @staticmethod
    def _set_axes_equal(ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


data_path = "C:/Users/louis/git/wind_downscaling_CNN/Data/1_Raw/"
topo_path = data_path + "Topo/IGN_25m/ign_L93_25m_alpesIG.tif"
AROME_path = data_path + "AROME/FORCING_alp_2019060107_2019070106.nc"
BDclim_stations_path = data_path + "BDclim/precise_localisation/liste_postes_alps_l93.csv"
BDclim_data_path = data_path + "BDclim/extract_BDClim_et_sta_alp_20171101_20190501.csv"
experience_path = "C:/Users/louis/git/wind_downscaling_CNN/Models/ARPS/"
model_experience = "date_16_02_name_simu_FINAL_1_0_model_UNet/"
model_path = experience_path + model_experience

day_1 = 1
day_2 = 30
month = 6
year = 2019
begin = str(year) + "-" + str(month) + "-" + str(day_1)
end = str(year) + "-" + str(month) + "-" + str(day_2)

# IGN
IGN = MNT(topo_path, "IGN")

# AROME
AROME = NWP(AROME_path, "AROME", begin, end)
AROME.gps_to_l93()

# BDclim
BDclim = Observation(BDclim_stations_path, BDclim_data_path)
# BDclim.stations_to_gdf(ccrs.epsg(2154), x="X", y="Y")
number_of_neighbors = 4
BDclim.update_stations_with_KNN_from_NWP(number_of_neighbors, AROME)
# BDclim.update_stations_with_KNN_from_MNT(IGN)
BDclim.update_stations_with_KNN_from_MNT_using_cKDTree(IGN)
# MNT_data, MNT_x, MNT_y = BDclim.extract_MNT_around_station('Col du Lac Blanc', IGN, 100, 100)

# Processing
p = Processing(BDclim, IGN, AROME, model_path)
array_xr = p.predict_UV_with_CNN(['Col du Lac Blanc'], fast=False, verbose=True, plot=False)

# Visualization
v = Visualization(p)
# v.plot_predictions_2D(['Col du Lac Blanc'], array_xr)
#v.plot_predictions_3D(['Col du Lac Blanc'], array_xr)
#v.plot_predictions_2D(['Col du Lac Blanc'], array_xr)
v.plot_comparison_topography_MNT_NWP(station_name='Col du Lac Blanc')