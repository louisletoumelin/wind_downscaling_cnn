import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from time import time as t
from datetime import datetime


try:
    from shapely.geometry import Point
    from shapely.geometry import Polygon
    _shapely_geometry = True
except:
    _shapely_geometry = False

try:
    import geopandas as gpd
    _geopandas = True
except:
    _geopandas = False
try:
    concurrent.futures
except:
    pass

from Data_2D import Data_2D


class Observation:

    _shapely_geometry = _shapely_geometry

    def __init__(self, path_to_list_stations, path_to_time_series, path_vallot, path_saint_sorlin, path_argentiere, begin=None, end=None, select_date_time_serie=True, vallot=True, saint_sorlin=True, argentiere=True):
        t0 = t()

        # Path and dates
        self.begin = begin
        self.end = end
        self.path_to_list_stations = path_to_list_stations
        self.path_to_time_series = path_to_time_series
        self.path_vallot = path_vallot
        self.path_saint_sorlin = path_saint_sorlin
        self.path_argentiere = path_argentiere

        # Stations
        if _shapely_geometry:
            self.stations = pd.read_csv(path_to_list_stations)
        else:
            self.stations = pd.read_pickle(path_to_list_stations)

        # Add additional stations
        if vallot: self._add_station(name='Vallot')
        if saint_sorlin: self._add_station(name='Saint-Sorlin')
        if argentiere: self._add_station(name='Argentiere')

        # Time series
        self.time_series = pd.read_csv(path_to_time_series)
        self.time_series.index = self.time_series['date'].apply(lambda x: np.datetime64(x))
        if select_date_time_serie: self._select_date_time_serie()

        # Add additional time series
        if vallot: self._add_time_serie_vallot(log_profile=True)
        if saint_sorlin: self._add_time_serie_glacier(name='Saint-Sorlin', log_profile=False)
        if argentiere: self._add_time_serie_glacier(name='Argentiere', log_profile=False)

        t1 = t()
        print(f"\nObservation created in {np.round(t1-t0, 2)} seconds\n")

    def _select_date_time_serie(self):
        mask = (self.time_series.index > self.begin) & (self.time_series.index < self.end)
        self.time_series = self.time_series[mask]

    def _add_station(self, name=None):

        if name == 'Vallot':
            X = 998884.573304192
            Y = 6533967.012767595
            numposte = np.nan
            alti = 4360
            lon = 45.83972222
            lat = 6.85222222
            pb_localisation = np.nan

        if name == 'Saint-Sorlin':
            X = 948949.3641216389
            Y = 6457790.489842982
            numposte = np.nan
            alti = 2720
            lon = 45.17444
            lat = 6.17
            pb_localisation = np.nan

        if name == 'Argentiere':
            X = 1007766.7474749532
            Y = 6548636.997793528
            numposte = np.nan
            alti = 2434
            lon = 45.967699
            lat = 6.976024
            pb_localisation = np.nan

        new_station = pd.DataFrame(self.stations.iloc[0]).transpose()
        new_station['X'] = X
        new_station['Y'] = Y
        new_station["numposte"] = numposte
        new_station["name"] = name
        new_station["alti"] = alti
        new_station["lon"] = lon
        new_station["lat"] = lat
        new_station["PB-localisation"] = pb_localisation
        new_station["Unnamed: 8"] = np.nan
        self.stations = pd.concat([self.stations, new_station], ignore_index=True)

    def _add_time_serie_vallot(self, log_profile=True):

        # Create a DataFrame with all yearly files
        vallot = []
        for year in range(2013, 2019):
            vallot_year = pd.read_csv(self.path_vallot + f"Vallot_corrected_Halfh_{year}.csv", sep=';')
            vallot.append(vallot_year)
        vallot = pd.concat(vallot)

        # Discard nan in dates
        vallot = vallot[vallot["date"].notna()]

        # Date
        vallot['date'] = vallot['date'].apply(lambda x: np.datetime64(datetime.strptime(x, "%d/%m/%Y %H:%M")))
        vallot.index = vallot["date"]

        # Columns to fit BDclim
        vallot["name"] = 'Vallot'
        vallot["numposte"] = np.nan
        vallot["vwmax_dir(deg)"] = np.nan
        vallot["P(mm)"] = np.nan
        vallot["HTN(cm)"] = np.nan

        # 45°50’22.93N / 6°51’7.60E, altitude 4360 m
        vallot["lon"] = 45.83972222
        vallot["lat"] = 6.85222222
        vallot["alti'"] = 4360

        # Discard duplicates
        vallot = vallot[~vallot.index.duplicated()]

        # Resample to hourly values: keep only top of hour values
        vallot = vallot.resample('1H').first()

        # The measurement height is 3m and we apply a log profile to 10m
        if log_profile:
            z0_vallot = 0.00549
            log_profile = np.log(10 / z0_vallot) / np.log(3 / z0_vallot)
            vallot['vw10m(m/s)'] = vallot['vw10m(m/s)'] * log_profile

        self.time_series = pd.concat([self.time_series, vallot])

    def _add_time_serie_glacier(self, log_profile=False, name=None, verbose=False):

        # Create a file containing all years
        glacier = []

        if name == 'Saint-Sorlin':
            for year in range(2006, 2020):
                glacier_year = pd.read_csv(self.path_saint_sorlin + f"SaintSorlin{year}-halfhourly.csv", sep=';', header=2)
                glacier.append(glacier_year)

        if name == 'Argentiere':
            for year in range(2007, 2020):
                # Corrected dates in 2018
                if year == 2018:
                    glacier_year = pd.read_csv(self.path_argentiere + f"Argentiere{year}-halfhourly_corrected.csv", sep=';', header=2)
                else:
                    glacier_year = pd.read_csv(self.path_argentiere + f"Argentiere{year}-halfhourly.csv", sep=';', header=2)
                glacier.append(glacier_year)

        glacier = pd.concat(glacier)

        # Rename columns
        glacier = glacier.rename(
            columns={"Unnamed: 0": "date", "T (°C)": "T2m(degC)", "Wind speed (m/s)": "vw10m(m/s)",
                     "Wind dir. (°/N)": "winddir(deg)"})

        # Select data
        if name == 'Saint-Sorlin':
            glacier = glacier[["date", "T2m(degC)", "vw10m(m/s)", "winddir(deg)"]]
        if name == 'Argentiere':
            glacier = glacier[["date", "T2m(degC)", "vw10m(m/s)", "winddir(deg)", "Unnamed: 7"]]

        # Print number of NaNs
        if verbose:
            nb_nan = len(glacier[glacier["date"].isna()])
            print("Found NaNs in dates: " + str(nb_nan))

        # Discard NaNs in dates
        glacier = glacier[glacier["date"].notna()]

        # Columns to fit BDclim
        glacier["name"] = name
        glacier["numposte"] = np.nan
        glacier["vwmax_dir(deg)"] = np.nan
        glacier["P(mm)"] = np.nan
        glacier["HTN(cm)"] = np.nan

        # 45°10’28.3’’N / 6°10’12.1’’E, altitude 2720 m
        glacier["lon"] = self.stations["lon"][self.stations["name"] == name].values[0]
        glacier["lat"] = self.stations["lat"][self.stations["name"] == name].values[0]
        glacier["alti'"] = self.stations["alti"][self.stations["name"] == name].values[0]

        # Dates are converted to np.datetime64
        glacier['date'] = glacier['date'].apply(
            lambda x: np.datetime64(datetime.strptime(x, "%d/%m/%Y %H:%M")))

        # Create the index
        glacier.index = glacier["date"]

        # Discard duplicates
        if verbose:
            nb_duplicate = len(glacier[glacier.index.duplicated()])
            print("Found date duplicate: " + str(nb_duplicate))
        glacier = glacier[~glacier.index.duplicated()]

        if name == 'Argentiere':
            if verbose:
                # Print number of annotated observations
                nb_annotated_observations = len(glacier[glacier["Unnamed: 7"].notna()])
                print("Annotated observations: " + str(nb_annotated_observations))

            # Discard annotated observations
            glacier = glacier[glacier["Unnamed: 7"].isna()]
            glacier = glacier.drop("Unnamed: 7", axis=1)

        # Resample to hourly values: keep only top of hour values
        glacier = glacier.resample('1H').first()

        if verbose:
            nb_missing_dates = len(glacier.asfreq('1H').index) - len(glacier.index)
            print("Number missing dates: " + str(nb_missing_dates))

        if log_profile:
            # Apply log profile
            if name == "Saint-Sorlin":
                z0_glacier = 0.0135
            if name == "Argentiere":
                z0_glacier = 1.015
            log_profile = np.log(10 / z0_glacier) / np.log(3 / z0_glacier)
            z0_glacier['Wind speed (m/s)'] = z0_glacier['Wind speed (m/s)'] * log_profile

        self.time_series = pd.concat([self.time_series, glacier])

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
            print("Parallel computation worked for update_stations_with_KNN_from_NWP")
        except:
            print("Parallel computation using concurrent.futures didn't work, so update_stations_with_KNN_from_NWP will not be parallelized.")
            list_nearest = map(K_N_N_point, list_coord_station)

        # Store results as array
        list_nearest = np.array([np.array(station) for station in list_nearest])
        list_index = [(x, y) for x in range(nwp.height) for y in range(nwp.length)]

        # Update DataFrame
        for neighbor in range(number_neighbor):
            self.stations[f'delta_x_{nwp.name}_NN_{neighbor}'] = list_nearest[:, 0, neighbor]
            self.stations[f'{nwp.name}_NN_{neighbor}'] = [grid_flat[int(index)] for index in
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

    @staticmethod
    def _degToCompass(num):
        if np.isnan(num):
            return(np.nan)
        else:
            val=int((num/22.5)+.5)
            arr=["N","NNE","NE","ENE","E","ESE", "SE", "SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
            return(arr[(val % 16)])

    def qc_initilization(self, wind_speed='vw10m(m/s)', wind_direction='winddir(deg)'):
        time_series = self.time_series
        list_stations = time_series["name"].unique()

        list_dataframe = []
        for station in list_stations:

            # Select station
            filter = time_series["name"] == station
            time_series_station = time_series[filter]

            # Create UV_DIR
            time_series_station["UV_DIR"] = time_series_station[wind_direction]

            # Create validity
            time_series_station["validity"] = 1

            # Create validity
            time_series_station["last_flagged"] = 0
            time_series_station["last_unflagged"] = 0

            # Create resolution
            time_series_station['resolution_speed'] = np.nan
            time_series_station['resolution_direction'] = np.nan

            # N, NNE, NE etc
            time_series_station['cardinal'] = [self._degToCompass(direction) for direction in time_series_station[wind_direction].values]

            # Add station to list of dataframe
            list_dataframe.append(time_series_station)

        self.time_series = pd.concat(list_dataframe)

    def qc_check_duplicates_in_index(self):
        """
        Quality control

        This function looks for duplicated dates in observations index
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []
        nb_problem = 0

        for station in list_stations:
            filter = time_series["name"] == station
            time_series_station = time_series[filter]

            if time_series_station.index.duplicated().sum() > 0:
                print("Found duplicated index")
                print(station)
                nb_problem += 1
            else:
                pass

        print(f"Found {nb_problem} duplicated dates")

    def qc_resample_index(self, frequency = '1H'):
        """
        Quality control

        This function fill NaN at missing dates in index
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []
        for station in list_stations:
            filter = time_series["name"] == station
            time_series_station = time_series[filter].asfreq(frequency)
            time_series_station["name"] = station
            list_dataframe.append(time_series_station.asfreq(frequency))

        self.time_series = pd.concat(list_dataframe)

    def qc_get_wind_speed_resolution(self, wind_speed = 'vw10m(m/s)'):
        """
        Quality control

        This function determines the resolution of wind speed observations
        Possible resolutions are 1m/s, 0.1m/s, 0.01m/s, 0.001m/s, 0.0001m/s
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []
        for station in list_stations:

            # Select station
            filter = time_series["name"] == station
            time_series_station = time_series[filter]
            for wind_per_day in time_series_station[wind_speed].groupby(pd.Grouper(freq='D')):

                # Check for resolution
                wind = wind_per_day[1]
                resolution_found = False
                decimal = 0
                while not(resolution_found):

                    wind_array = wind.values
                    wind_round_array = wind.round(decimal).values

                    if np.allclose(wind_array, wind_round_array, equal_nan=True):
                        time_series_station['resolution_speed'][time_series_station.index.isin(wind_per_day[1].index)] = decimal
                        resolution_found = True
                    else:
                        decimal += 1

                    if decimal >= 4:
                        time_series_station['resolution_speed'][time_series_station.index.isin(wind_per_day[1].index)] = decimal
                        resolution_found = True

            # Check that high resolution are not mistaken as low resolution
            resolution = 5
            nb_obs = time_series_station['resolution_speed'].count()
            while resolution >= 0:
                if time_series_station['resolution_speed'][time_series_station['resolution_speed'] == resolution].count() > 0.8 * nb_obs:
                    time_series_station['resolution_speed'] = resolution
                    resolution = -1
                else:
                    resolution = resolution - 1

            # Add station to list of dataframe
            list_dataframe.append(time_series_station)
        self.time_series = pd.concat(list_dataframe)

    def qc_get_wind_direction_resolution(self, wind_direction='winddir(deg)'):
        """
        Quality control

        This function determines the resolution of wind direction observations
        Possible resolutions are 10°, 5°, 1°, 0.1°
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []
        for station in list_stations:

            # Select station
            filter = time_series["name"] == station
            time_series_station = time_series[filter]
            for wind_per_day in time_series_station[wind_direction].groupby(pd.Grouper(freq='D')):

                # Check for resolution
                wind = wind_per_day[1]
                resolution_found = False
                resolutions = [10, 5, 1, 0.1]
                index = 0
                while not(resolution_found):
                    resolution = resolutions[index]
                    wind_array = wind.values
                    wind_round_array = np.around(wind_array/resolution, decimals=0)*resolution

                    if np.allclose(wind_array, wind_round_array, equal_nan=True):
                        time_series_station['resolution_direction'][time_series_station.index.isin(wind_per_day[1].index)] = resolution
                        resolution_found = True
                    else:
                        index += 1

                    if index >= 3:
                        resolution = resolutions[index]
                        time_series_station['resolution_direction'][time_series_station.index.isin(wind_per_day[1].index)] = resolution
                        resolution_found = True

            # Check that high resolution are not mistaken as low resolution
            resolutions = [10, 5, 1, 0.1]
            index = 0
            nb_obs = time_series_station['resolution_direction'].count()
            while index <= 3:
                resolution = resolutions[index]
                if time_series_station['resolution_direction'][time_series_station['resolution_direction'] == resolution].count() > 0.8 * nb_obs:
                    time_series_station['resolution_direction'] = resolution
                    index = 100
                else:
                    index = index + 1

            # Add station to list of dataframe
            list_dataframe.append(time_series_station)
        self.time_series = pd.concat(list_dataframe)

    def qc_calm_criteria(self, wind_speed='vw10m(m/s)', wind_direction='winddir(deg)'):
        """
        Quality control

        This function apply calm criteria
        UV = 0m/s => UV_DIR = 0°
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []
        for station in list_stations:

            # Select station
            filter = time_series["name"] == station
            time_series_station = time_series[filter]

            # Calm criteria: UV = 0m/s => UV_DIR = 0°
            time_series_station["UV_DIR"][time_series_station[wind_speed] == 0] = 0

            # Add station to list of dataframe
            list_dataframe.append(time_series_station)

        self.time_series = pd.concat(list_dataframe)

    def qc_true_north(self, wind_speed='vw10m(m/s)', wind_direction='winddir(deg)'):
        """
        Quality control

        This function apply true north criteria
        UV != 0m/s and (UV_DIR=0 or UV_DIR=360) => UV_DIR = 360
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []
        for station in list_stations:

            # Select station
            filter = time_series["name"] == station
            time_series_station = time_series[filter]

            # Calm criteria: UV = 0m/s => UV_DIR = 0°
            filter_1 = (time_series_station[wind_speed] != 0)
            filter_2 = (time_series_station[wind_direction] == 0)
            filter_3 = (time_series_station[wind_direction] == 360)
            time_series_station["UV_DIR"][filter_1 & (filter_2 | filter_3)] = 360

            # Add station to list of dataframe
            list_dataframe.append(time_series_station)

        self.time_series = pd.concat(list_dataframe)

    def qc_removal_unphysical_values(self, wind_speed='vw10m(m/s)', wind_direction='winddir(deg)'):
        """
        Quality control

        This function flags unphysical values
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []
        for station in list_stations:

            # Select station
            filter = time_series["name"] == station
            time_series_station = time_series[filter]

            # Specify result of the test
            time_series_station["qc_1"] = 1

            # Calm criteria: UV = 0m/s => UV_DIR = 0°
            filter_1 = (time_series_station[wind_speed] < 0)
            filter_2 = (time_series_station[wind_speed] > 100)
            filter_3 = (time_series_station[wind_direction] < 0)
            filter_4 = (time_series_station[wind_direction] > 360)

            time_series_station['validity'][(filter_1 | filter_2 | filter_3 | filter_4)] = 0
            time_series_station["qc_1"][(filter_1 | filter_2)] = "unphysical_wind_speed"
            time_series_station["last_flagged"][(filter_1 | filter_2)] = "unphysical_wind_speed"
            time_series_station["qc_1"][(filter_3 | filter_4)] = "unphysical_wind_direction"
            time_series_station["last_flagged"][(filter_3 | filter_4)] = "unphysical_wind_direction"
            time_series_station["qc_1"][(filter_1 | filter_2) & (filter_3 | filter_4)] = "unphysical_wind_speed_and_direction"

            # Add station to list of dataframe
            list_dataframe.append(time_series_station)

        self.time_series = pd.concat(list_dataframe)

    def qc_constant_sequences(self, wind_speed='vw10m(m/s)', wind_direction='winddir(deg)'):
        """
        Quality control

        This function detect constant sequences and their lengths
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []
        dict_constant_sequence = {}
        for station in list_stations:

            # Select station
            filter = time_series["name"] == station
            time_series_station = time_series[filter]

            # Initialize dictionary direction
            dict_constant_sequence[station] = {}
            dict_constant_sequence[station]["length_constant_direction"] = {}
            dict_constant_sequence[station]["date_constant_direction_begin"] = {}
            dict_constant_sequence[station]["date_constant_direction_end"] = {}
            for variable in ["length_constant_direction", "date_constant_direction_begin", "date_constant_direction_end"]:
                for resolution in [10, 5, 1, 0.1]:
                    dict_constant_sequence[station][variable][str(resolution)] = {}
                    dict_constant_sequence[station][variable][str(resolution)]["=0"] = []
                    dict_constant_sequence[station][variable][str(resolution)]["!=0"] = []

            # Initialize dictionary speed
            dict_constant_sequence[station]["length_constant_speed"] = {}
            dict_constant_sequence[station]["date_constant_speed_begin"] = {}
            dict_constant_sequence[station]["date_constant_speed_end"] = {}
            for variable in ["length_constant_speed", "date_constant_speed_begin", "date_constant_speed_end"]:
                for resolution in range(5):
                    dict_constant_sequence[station][variable][str(resolution)] = {}
                    dict_constant_sequence[station][variable][str(resolution)]["<1m/s"] = []
                    dict_constant_sequence[station][variable][str(resolution)][">=1m/s"] = []

            # Inputs
            speed_values = time_series_station[wind_speed].values
            direction_values = time_series_station[wind_direction].values
            dates = time_series_station.index.values
            resolution_speed = time_series_station["resolution_speed"].values
            resolution_direction = time_series_station["resolution_direction"].values
            nb_values = len(speed_values)

            # Outputs
            constant_speed = np.zeros(nb_values)
            constant_direction = np.zeros(nb_values)

            # Lengths
            lenght_speed = 0
            lenght_direction = 0

            # Begin with non constant sequence
            previous_step_speed = False
            previous_step_direction = False

            resolution_constant_sequence = []
            speed_sequence = []
            resolution_constant_sequence_direction = []
            direction_sequence = []
            for index in range(nb_values-1):

                # Constant speed sequences
                if np.isnan(speed_values[index]):
                    constant_speed[index] = np.nan
                    previous_step_speed = False
                    resolution_constant_sequence = []
                    speed_sequence = []
                elif speed_values[index] != speed_values[index+1]:
                    constant_speed[index+1] = 0
                    # If the previous sequence was constant and is finished
                    if previous_step_speed:
                        if np.mean(speed_sequence) >= 1:
                            dict_constant_sequence[station]["length_constant_speed"][str(np.bincount(resolution_constant_sequence).argmax())][">=1m/s"].append(lenght_speed)
                            dict_constant_sequence[station]["date_constant_speed_end"][str(np.bincount(resolution_constant_sequence).argmax())][">=1m/s"].append(dates[index])
                            dict_constant_sequence[station]["date_constant_speed_begin"][str(np.bincount(resolution_constant_sequence).argmax())][">=1m/s"].append(date_begin)
                        else:
                            dict_constant_sequence[station]["length_constant_speed"][str(np.bincount(resolution_constant_sequence).argmax())]["<1m/s"].append(lenght_speed)
                            dict_constant_sequence[station]["date_constant_speed_end"][str(np.bincount(resolution_constant_sequence).argmax())]["<1m/s"].append(dates[index])
                            dict_constant_sequence[station]["date_constant_speed_begin"][str(np.bincount(resolution_constant_sequence).argmax())]["<1m/s"].append(date_begin)
                    previous_step_speed = False
                    resolution_constant_sequence = []
                    speed_sequence = []
                else:
                    constant_speed[index] = 1
                    constant_speed[index+1] = 1
                    resolution_constant_sequence.append(resolution_speed[index])
                    speed_sequence.append(speed_values[index])
                    # If the previous sequence was constant and continues
                    if previous_step_speed:
                        lenght_speed += 1
                    else:
                        lenght_speed = 2
                        date_begin = dates[index]
                    # If the time_serie end with a constant sequence
                    if index == nb_values - 2:
                        if np.mean(speed_sequence) >= 1:
                            dict_constant_sequence[station]["length_constant_speed"][str(np.bincount(resolution_constant_sequence).argmax())][">=1m/s"].append(lenght_speed)
                            dict_constant_sequence[station]["date_constant_speed_end"][str(np.bincount(resolution_constant_sequence).argmax())][">=1m/s"].append(dates[index])
                            dict_constant_sequence[station]["date_constant_speed_begin"][str(np.bincount(resolution_constant_sequence).argmax())][">=1m/s"].append(date_begin)
                        else:
                            dict_constant_sequence[station]["length_constant_speed"][str(np.bincount(resolution_constant_sequence).argmax())]["<1m/s"].append(lenght_speed)
                            dict_constant_sequence[station]["date_constant_speed_end"][str(np.bincount(resolution_constant_sequence).argmax())]["<1m/s"].append(dates[index])
                            dict_constant_sequence[station]["date_constant_speed_begin"][str(np.bincount(resolution_constant_sequence).argmax())]["<1m/s"].append(date_begin)
                    previous_step_speed = True

                # Constant direction sequences
                if np.isnan(direction_values[index]):
                    constant_direction[index] = np.nan
                    previous_step_direction = False
                    resolution_constant_sequence_direction = []
                    direction_sequence = []
                if direction_values[index] != direction_values[index+1]:
                    constant_direction[index + 1] = 0
                    # If the previous sequence was constant and is finished
                    if previous_step_direction:
                        if np.mean(direction_sequence) == 0:
                            most_frequent_value = np.bincount([10 * value for value in resolution_constant_sequence_direction]).argmax() / 10
                            most_frequent_value = int(most_frequent_value) if most_frequent_value >= 1 else most_frequent_value
                            dict_constant_sequence[station]["length_constant_direction"][str(most_frequent_value)]["=0"].append(lenght_direction)
                            dict_constant_sequence[station]["date_constant_direction_end"][str(most_frequent_value)]["=0"].append(dates[index])
                            dict_constant_sequence[station]["date_constant_direction_begin"][str(most_frequent_value)]["=0"].append(date_begin)
                        else:
                            most_frequent_value = np.bincount([10 * value for value in resolution_constant_sequence_direction]).argmax() / 10
                            most_frequent_value = int(most_frequent_value) if most_frequent_value >= 1 else most_frequent_value
                            dict_constant_sequence[station]["length_constant_direction"][str(most_frequent_value)]["!=0"].append(lenght_direction)
                            dict_constant_sequence[station]["date_constant_direction_end"][str(most_frequent_value)]["!=0"].append(dates[index])
                            dict_constant_sequence[station]["date_constant_direction_begin"][str(most_frequent_value)]["!=0"].append(date_begin)
                    previous_step_direction = False
                    resolution_constant_sequence_direction = []
                    direction_sequence = []
                else:
                    constant_direction[index] = 1
                    constant_direction[index + 1] = 1
                    resolution_constant_sequence_direction.append(resolution_direction[index])
                    direction_sequence.append(direction_values[index])
                    # If the previous sequence was constant and continues
                    if previous_step_direction:
                        lenght_direction += 1
                    else:
                        lenght_direction = 2
                        date_begin = dates[index]
                    # If the time_serie end with a constant sequence
                    if index == nb_values - 2:
                        if np.mean(direction_sequence) == 0:
                            most_frequent_value = np.bincount([10 * value for value in resolution_constant_sequence_direction]).argmax() / 10
                            most_frequent_value = int(most_frequent_value) if most_frequent_value >= 1 else most_frequent_value
                            dict_constant_sequence[station]["length_constant_direction"][str(most_frequent_value)]["=0"].append(lenght_direction)
                            dict_constant_sequence[station]["date_constant_direction_end"][str(most_frequent_value)]["=0"].append(dates[index])
                            dict_constant_sequence[station]["date_constant_direction_begin"][str(most_frequent_value)]["=0"].append(date_begin)
                        else:
                            most_frequent_value = np.bincount([10 * value for value in resolution_constant_sequence_direction]).argmax() / 10
                            most_frequent_value = int(most_frequent_value) if most_frequent_value >= 1 else most_frequent_value
                            dict_constant_sequence[station]["length_constant_direction"][str(most_frequent_value)]["!=0"].append(lenght_direction)
                            dict_constant_sequence[station]["date_constant_direction_end"][str(most_frequent_value)]["!=0"].append(dates[index])
                            dict_constant_sequence[station]["date_constant_direction_begin"][str(most_frequent_value)]["!=0"].append(date_begin)
                    previous_step_direction = True

            # Specify result of the test
            time_series_station["constant_speed"] = constant_speed
            time_series_station["constant_direction"] = constant_direction

            # Add station to list of dataframe
            list_dataframe.append(time_series_station)

        self.time_series = pd.concat(list_dataframe)
        return(dict_constant_sequence)

    def qc_excessive_MISS(self, dict_constant_sequence, wind_speed='vw10m(m/s)', wind_direction='winddir(deg)'):
        """
        Quality control

        This function detect suspicious constant sequences based on the number of missing values
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []
        for station in list_stations:

            # Select station
            filter = time_series["name"] == station
            time_series_station = time_series[filter]

            # Initialize dictionary
            time_series_station['qc_2'] = 1

            # Wind speed
            rolling_speed = time_series_station[wind_speed].isna().rolling('1D').sum()
            for resolution in range(5):
                for speed in ['<1m/s', '>=1m/s']:

                    # Select constant sequences
                    begins = dict_constant_sequence[station]["date_constant_speed_begin"][str(resolution)][speed]
                    ends = dict_constant_sequence[station]["date_constant_speed_end"][str(resolution)][speed]


                    for begin, end in zip(begins, ends):

                        missing_period_speed = 0

                        # If current sequence has many nans
                        if time_series_station[wind_speed][begin:end].isna().sum() > 0.9*time_series_station[wind_speed][begin:end].count():
                            missing_period_speed += 1

                        # If previous sequence has many nans
                        try:
                            if rolling_speed[begin-np.timedelta64(1, 'D')] > 0.9 * 24:
                                missing_period_speed += 1
                        except KeyError:
                            pass

                        # If next sequence has many nans
                        if rolling_speed[end] > 0.9 * 24:
                            missing_period_speed += 1

                        # If two sequences or more have many nans we flag it
                        if missing_period_speed >= 2:
                            time_series_station['validity'][begin:end] = 0
                            time_series_station['last_flagged'][begin:end] = 'excessive_miss_speed'
                            time_series_station['qc_2'][begin:end] = 'excessive_miss_speed'

            # Wind direction
            rolling_direction = time_series_station[wind_direction].isna().rolling('1D').sum()
            for resolution in [10, 5, 1, 0.1]:

                # Select constant sequences
                begins = dict_constant_sequence[station]["date_constant_direction_begin"][str(resolution)]['!=0']
                ends = dict_constant_sequence[station]["date_constant_direction_end"][str(resolution)]['!=0']
                for begin, end in zip(begins, ends):

                    missing_period_direction = 0

                    # If current sequence has many nans
                    if time_series_station[wind_direction][begin:end].isna().sum() > 0.9*time_series_station[wind_direction][begin:end].count():
                        missing_period_direction += 1

                    # If previous sequence has many nans
                    try:
                        if rolling_direction[begin - np.timedelta64(1, 'D')] > 0.9 * 24:
                            missing_period_direction += 1
                    except KeyError:
                        pass

                    # If next sequence has many nans
                    if rolling_direction[end] > 0.9 * 24:
                        missing_period_direction += 1

                    # If two sequences or more have many nans we flag it
                    if missing_period_speed >= 2:
                        time_series_station['validity'][begin:end] = 0
                        time_series_station['last_flagged'][begin:end] = 'excessive_miss_direction'
                        time_series_station['qc_2'][begin:end] = 'excessive_miss_direction'

            # Add station to list of dataframe
            list_dataframe.append(time_series_station)

        self.time_series = pd.concat(list_dataframe)

    def qc_get_stats_cst_seq(self, dict_constant_sequence, amplification_factor=1):
        """
        Quality control

        This function detect compute statistics used to flag constant sequences
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        dict_all_stations = {}
        dict_all_stations["length_constant_speed"] = {}
        dict_all_stations["length_constant_direction"] = {}

        # Wind speed resolution
        for resolution in range(5):
            dict_all_stations["length_constant_speed"][str(resolution)] = {}
            dict_all_stations["length_constant_speed"][str(resolution)]["<1m/s"] = {}
            dict_all_stations["length_constant_speed"][str(resolution)][">=1m/s"] = {}
            dict_all_stations["length_constant_speed"][str(resolution)]["<1m/s"]["stats"] = {}
            dict_all_stations["length_constant_speed"][str(resolution)][">=1m/s"]["stats"] = {}
            dict_all_stations["length_constant_speed"][str(resolution)][">=1m/s"]["values"] = []
            dict_all_stations["length_constant_speed"][str(resolution)]["<1m/s"]["values"] = []

        # Wind direction resolution
        for resolution in [10, 5, 1, 0.1]:
            dict_all_stations["length_constant_direction"][str(resolution)] = {}
            dict_all_stations["length_constant_direction"][str(resolution)]["!=0"] = {}
            dict_all_stations["length_constant_direction"][str(resolution)]["!=0"]["stats"] = {}
            dict_all_stations["length_constant_direction"][str(resolution)]["!=0"]["values"] = []

        # Reconstruct dictionary without station
        for station in list_stations:
            # Speed
            for resolution in range(5):
                array_1 = dict_constant_sequence[station]["length_constant_speed"][str(resolution)]["<1m/s"]
                array_2 = dict_constant_sequence[station]["length_constant_speed"][str(resolution)][">=1m/s"]
                if np.array(array_1).size !=0: dict_all_stations["length_constant_speed"][str(resolution)]["<1m/s"]["values"].extend(array_1)
                if np.array(array_2).size !=0: dict_all_stations["length_constant_speed"][str(resolution)][">=1m/s"]["values"].extend(array_2)

            for resolution in [10, 5, 1, 0.1]:
                array_1 = dict_constant_sequence[station]["length_constant_direction"][str(resolution)]["!=0"]
                if np.array(array_1).size !=0: dict_all_stations["length_constant_direction"][str(resolution)]["!=0"]["values"].extend(array_1)

        # Statistics speed
        for resolution in range(5):
            for speed in ["<1m/s", ">=1m/s"]:
                values = dict_all_stations["length_constant_speed"][str(resolution)][speed]["values"]
                if np.array(values).size != 0:
                    dict_all_stations["length_constant_speed"][str(resolution)][speed]["stats"]["P95"] = np.quantile(values, 0.95)
                    dict_all_stations["length_constant_speed"][str(resolution)][speed]["stats"]["P75"] = np.quantile(values, 0.75)
                    dict_all_stations["length_constant_speed"][str(resolution)][speed]["stats"]["P25"] = np.quantile(values, 0.25)

        # Statistics direction
        for resolution in [10, 5, 1, 0.1]:
            values = dict_all_stations["length_constant_direction"][str(resolution)]["!=0"]["values"]
            if np.array(values).size != 0:
                dict_all_stations["length_constant_direction"][str(resolution)]["!=0"]["stats"]["P95"] = np.quantile(values, 0.95)
                dict_all_stations["length_constant_direction"][str(resolution)]["!=0"]["stats"]["P75"] = np.quantile(values, 0.75)
                dict_all_stations["length_constant_direction"][str(resolution)]["!=0"]["stats"]["P25"] = np.quantile(values, 0.25)

        # Criterion speed
        for resolution in range(5):

            print(f"\nResolution speed: {resolution}")

            # Select values
            values_1 = dict_all_stations["length_constant_speed"][str(resolution)]["<1m/s"]["values"]
            values_2 = dict_all_stations["length_constant_speed"][str(resolution)][">=1m/s"]["values"]

            # Low wind speeds
            if len(values_1) >= 10:
                # Criteria = max(3, P95 + amplification_factor * 7.5 * IQR)
                criteria = np.max((3, np.quantile(values_1, 0.95) + amplification_factor * 7.5 * (np.quantile(values_1, 0.75) - np.quantile(values_1, 0.25))))
                dict_all_stations["length_constant_speed"][str(resolution)]["<1m/s"]["stats"]["criteria"] = criteria
                print("criterion speed <1m/s: ", criteria)
            else:
                dict_all_stations["length_constant_speed"][str(resolution)]["<1m/s"]["stats"]["criteria"] = None

            # High wind speeds
            if len(values_2) >= 10:
                # Criteria = max(3, P95 + amplification_factor * 8 * IQR)
                criteria = np.max((3, np.quantile(values_2, 0.95) + amplification_factor * 8 * (np.quantile(values_2, 0.75) - np.quantile(values_2, 0.25))))
                dict_all_stations["length_constant_speed"][str(resolution)][">=1m/s"]["stats"]["criteria"] = criteria
                print("criterion speed >=1m/s: ", criteria)
            else:
                dict_all_stations["length_constant_speed"][str(resolution)][">=1m/s"]["stats"]["criteria"] = None

        # Criterion direction
        for resolution in [10, 5, 1, 0.1]:
            print(f"\nResolution direction: {resolution}")

            # Constant direction not null
            values_1 = dict_all_stations["length_constant_direction"][str(resolution)]["!=0"]["values"]
            if len(values_1) >= 10:

                # Criteria = max(3, P95 + amplification_factor * 15 * IQR)
                criteria = np.max((3, np.quantile(values_1, 0.95) + amplification_factor * 15 * (np.quantile(values_1, 0.75) - np.quantile(values_1, 0.25))))
                dict_all_stations["length_constant_direction"][str(resolution)]["!=0"]["stats"]["criteria"] = criteria
                print("criterion direction !=0: ", criteria)
            else:
                dict_all_stations["length_constant_direction"][str(resolution)]["!=0"]["stats"]["criteria"] = None
        return(dict_all_stations)

    def qc_apply_stats_cst_seq(self, dict_constant_sequence, dict_all_stations, wind_speed='vw10m(m/s)', wind_direction='winddir(deg)'):
        """
        Quality control

        This function apply criterions to constant sequences
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []
        for station in list_stations:

            # Select station
            filter = time_series["name"] == station
            time_series_station = time_series[filter]

            # qc_cst_sequence
            time_series_station['qc_3'] = np.nan

            # Preferred direction
            pref_direction = time_series_station['cardinal'].value_counts().nlargest(n=1).index

            # Speed
            for resolution in range(5):
                for speed in ['<1m/s', '>=1m/s']:

                    # Get criteria
                    criteria = dict_all_stations['length_constant_speed'][str(resolution)][speed]['stats']['criteria']

                    for index, length in enumerate(dict_constant_sequence[station]['length_constant_speed'][str(resolution)][speed]):

                        # Apply criteria
                        if criteria is not None and (length >= criteria):

                            # Select constant sequence
                            begin = dict_constant_sequence[station]['date_constant_speed_begin'][str(resolution)][speed][index]
                            end = dict_constant_sequence[station]['date_constant_speed_end'][str(resolution)][speed][index]

                            # Flag series
                            time_series_station['validity'][begin:end] = 0
                            time_series_station['last_flagged'][begin:end] = 'cst_sequence_criteria'
                            time_series_station['qc_3'][begin:end] = 'cst_sequence_criteria'

            # Direction
            for resolution in [10, 5, 1, 0.1]:

                # Get criteria
                criteria = dict_all_stations['length_constant_direction'][str(resolution)]['!=0']['stats']['criteria']
                for index, length in enumerate(dict_constant_sequence[station]['length_constant_direction'][str(resolution)]['!=0']):

                    # Apply criteria
                    if criteria is not None and (length >= criteria):

                        # Select constant sequence
                        begin = dict_constant_sequence[station]['date_constant_direction_begin'][str(resolution)]['!=0'][index]
                        end = dict_constant_sequence[station]['date_constant_direction_end'][str(resolution)]['!=0'][index]

                        # Flag series
                        time_series_station['validity'][begin:end] = 0
                        time_series_station['last_flagged'][begin:end] = 'cst_sequence_criteria_not_passed'
                        time_series_station['qc_3'][begin:end] = 'cst_sequence_criteria_not_passed'

                        # Unflag if constant direction is preferred direction
                        direction_cst_seq = time_series_station[wind_direction][begin:end].value_counts().nlargest(n=1).index
                        if direction_cst_seq == pref_direction:
                            time_series_station['validity'][begin:end] = 1
                            time_series_station['last_unflagged'][begin:end] = 'pref_direction'
                            time_series_station['qc_3'][begin:end] = "cst_seq_too_long_but_pref_direction"

            # Add station to list of dataframe
            list_dataframe.append(time_series_station)

        self.time_series = pd.concat(list_dataframe)

    def qc_get_nearest_neigbhors(self):
        """
        Quality control

        This function determines the nearest neighbors of each station.

        We consider the neighbors that are 40km close and with an elevation difference < 500m
        """
        all_neighbors = []
        stations = self.stations
        for name in stations["name"]:

            # Select station
            station = stations[['X', 'Y']][stations["name"] == name].values[0]
            alti_station = stations['alti'][stations["name"] == name].values[0]

            # Select neighbors candidates
            all_stations_except_one = stations[['X', 'Y', 'alti', 'name']][stations["name"] != name]
            tree = cKDTree(all_stations_except_one[['X', 'Y']].values)

            # Distances to station
            distances, indexes_knn = tree.query(station, 50)

            neighbors_station = []
            for distance, index_knn in zip(distances, indexes_knn):
                if distance <= 40_000:
                    if abs(all_stations_except_one['alti'].iloc[index_knn] - alti_station) <= 500:
                        name = all_stations_except_one['name'].iloc[index_knn]
                        neighbors_station.append(name)

            all_neighbors.append(neighbors_station)
        self.stations['neighbors'] = all_neighbors

    def qc_ra(self, dict_constant_sequence, dict_all_stations, wind_speed='vw10m(m/s)', wind_direction='winddir(deg)'):
        """
        Quality control

        This function performs a regional analysis to flag/unflag constant sequences with respect to neighbors.
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []

        for station in list_stations:

            # Select station
            filter = time_series["name"] == station
            time_series_station = time_series[filter]

            # Regional analysis initialisation
            time_series_station["qc_4"] = np.nan
            time_series_station["ra"] = np.nan
            time_series_station["RA"] = np.nan
            time_series_station["qc_neighbors_selected"] = np.nan

            # Select neighbors
            list_neighbors = self.stations['neighbors'][self.stations["name"] == station].values[0]
            print(station)
            print(list_neighbors)

            # Select calm period for each resolution
            for resolution in range(5):
                print(f"Resolution: {resolution}")

                # Load constant sequences
                data = dict_constant_sequence[station]['length_constant_speed'][str(resolution)]['<1m/s']

                if np.size(data) != 0:
                    for index in range(len(data)):

                        # Length of constant time serie
                        length = dict_constant_sequence[station]['length_constant_speed'][str(resolution)]['<1m/s'][index]

                        # Apply criteria
                        criteria = dict_all_stations['length_constant_speed'][str(resolution)]['<1m/s']['stats']['criteria']
                        if criteria is not None and (length >= criteria):

                            # Time of constant sequence
                            begin = dict_constant_sequence[station]['date_constant_speed_begin'][str(resolution)]['<1m/s'][index]
                            end = dict_constant_sequence[station]['date_constant_speed_end'][str(resolution)]['<1m/s'][index]

                            # Ten days time serie
                            begin_ten_days = begin - np.timedelta64(10, 'D')
                            end_ten_days = end + np.timedelta64(10, 'D')
                            ten_days_1 = time_series_station[wind_speed][begin_ten_days: begin]
                            ten_days_2 = time_series_station[wind_speed][end: end_ten_days]
                            ten_days_time_serie = pd.concat((ten_days_1, ten_days_2))

                            # One day time serie
                            begin_one_day = begin - np.timedelta64(1, 'D')
                            end_one_day = end + np.timedelta64(1, 'D')

                            # Construct regional average
                            nb_neighbor_selected = 0
                            selected_neighbors = []
                            ra = []
                            RA = None
                            for neighbor in list_neighbors:

                                neighbor_time_serie = time_series[time_series["name"] == neighbor]
                                neighbor_ten_days_before = neighbor_time_serie[wind_speed][begin_ten_days: begin]
                                neighbor_ten_days_after = neighbor_time_serie[wind_speed][end: end_ten_days]
                                neighbor_ten_days_before_after = pd.concat((neighbor_ten_days_before, neighbor_ten_days_after))

                                # Correlation between ten day time series
                                data = np.array((ten_days_time_serie, neighbor_ten_days_before_after)).T
                                corr_coeff = pd.DataFrame(data).corr().iloc[0, 1]

                                # If stations are correlated
                                if corr_coeff >= 0.4:

                                    # Neighbor candidate accepted
                                    nb_neighbor_selected += 1
                                    selected_neighbors.append(neighbor)

                                    # Normalize neighbors observations
                                    neighbor_one_day = neighbor_time_serie[wind_speed][begin_one_day: end_one_day]
                                    ra.append((neighbor_one_day - np.nanmean(neighbor_one_day)) / np.nanstd(neighbor_one_day))

                            print(f"Number of neighbors selected: {nb_neighbor_selected}")

                            # If we found neighbors
                            if nb_neighbor_selected > 0:

                                # We compute the mean of ra of the neighbors
                                ra = pd.concat(ra).groupby(pd.concat(ra).index).mean() if len(ra) > 1 else ra[0]

                                # RA
                                vmin = np.nanmin(ra[begin:end])
                                vmax = np.nanmax([ra[begin_ten_days: begin], ra[end: end_ten_days]])
                                RA = (ra - vmin) / (vmax - vmin)
                                RA = RA[begin:end]

                                # Store variables
                                time_series_station["ra"][time_series_station.index.isin(ra)] = ra
                                time_series_station["RA"][time_series_station.index.isin(RA)] = RA
                                time_series_station["qc_neighbors_selected"][time_series_station.index.isin(RA)] = [selected_neighbors for k in range(len(RA))]

                            if RA is not None:
                                time_series_station["qc_4"][
                                    time_series_station.index.isin(RA[RA > 0.33].index)] = "qc_ra_suspicious"
                                time_series_station["validity"][time_series_station.index.isin(RA[RA > 0.33].index)] = 0
                                time_series_station["last_flagged"][time_series_station.index.isin(RA[RA > 0.33].index)] = "qc_ra_suspicious"
                                time_series_station["qc_4"][time_series_station.index.isin(RA[RA <= 0.33].index)] = "qc_ra_ok"
                                time_series_station["last_unflagged"][time_series_station.index.isin(RA[RA <= 0.33].index)] = "qc_ra_ok"

            # Add station to list of dataframe
            list_dataframe.append(time_series_station)

        self.time_series = pd.concat(list_dataframe)

    def qc_high_variability(self, wind_speed='vw10m(m/s)'):
        """
        Quality control

        This function detects suspicious high variability in data.
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []

        # Get statistics
        dict_high_variability = {}
        for station in list_stations:
            print(station)
            dict_high_variability[station] = {}

            # Select station
            filter = time_series["name"] == station
            time_series_station = time_series[filter]

            # High variability
            time_series_station["qc_5"] = np.nan
            time_series_station["qc_high_variability_criteria"] = np.nan

            # Delta time
            time = time_series_station.index.to_series()
            time_difference = (time - time.shift()) / np.timedelta64(1, 'h')

            # Increment for several timesteps
            for time_step in range(1, 24):

                # Initialize values
                dict_high_variability[station][str(time_step)] = {}
                dict_high_variability[station][str(time_step)]["values"] = {}
                dict_high_variability[station][str(time_step)]["values"][">=0"] = []
                dict_high_variability[station][str(time_step)]["values"]["<0"] = []

                # Wind_(i+n) - Wind_(i)
                increments = time_series_station[wind_speed].diff(periods=time_step)

                # Postive = increase, negative = decrease
                increments_positive = np.abs(increments[increments >= 0])
                increments_negative = np.abs(increments[increments < 0])

                dict_high_variability[station][str(time_step)]["values"][">=0"].extend(increments_positive.values)
                dict_high_variability[station][str(time_step)]["values"]["<0"].extend(increments_negative.values)

                # Statistics positives
                P95_p = np.nanquantile(increments_positive.values, 0.95)
                P75_p = np.nanquantile(increments_positive.values, 0.75)
                P25_p = np.nanquantile(increments_positive.values, 0.25)
                dict_high_variability[station][str(time_step)]["P95_p"] = P95_p
                dict_high_variability[station][str(time_step)]["P75_p"] = P75_p
                dict_high_variability[station][str(time_step)]["P25_p"] = P25_p

                # Statistics negatives
                P95_n = np.nanquantile(increments_negative.values, 0.95)
                P75_n = np.nanquantile(increments_negative.values, 0.75)
                P25_n = np.nanquantile(increments_negative.values, 0.25)
                dict_high_variability[station][str(time_step)]["P95_p"] = P95_n
                dict_high_variability[station][str(time_step)]["P75_p"] = P75_n
                dict_high_variability[station][str(time_step)]["P25_p"] = P25_n

                criteria_p = P95_p + 8.9 * (P75_p - P25_p)
                criteria_n = P95_n + 8.9 * (P75_n - P25_n)

                if criteria_p < 7.5:
                    criteria_p = 7.5

                if criteria_n < 7.5:
                    criteria_n = 7.5

                # Time resolution
                filter = time_difference == time_step

                time_series_station["qc_5"][(increments >= 0) & (increments >= criteria_p) & filter] = "too high"
                time_series_station["qc_5"][(increments < 0) & (np.abs(increments) >= criteria_n) & filter] = "too high"
                time_series_station["validity"][(increments >= 0) & (increments >= criteria_p) & filter] = 0
                time_series_station["last_flagged"][(increments >= 0) & (increments >= criteria_p) & filter] = "high variation"
                time_series_station["validity"][(increments < 0) & (np.abs(increments) >= criteria_n) & filter] = 0
                time_series_station["last_flagged"][(increments < 0) & (np.abs(increments) >= criteria_n) & filter] = "high variation"
                time_series_station["qc_high_variability_criteria"][(increments >= 0) & (increments >= criteria_p) & filter] = criteria_p
                time_series_station["qc_high_variability_criteria"][(increments < 0) & (np.abs(increments) >= criteria_n) & filter] = criteria_n

                print(f"Resolution: {time_step}")
                print(criteria_p, criteria_n)

            # Add station to list of dataframe
            list_dataframe.append(time_series_station)

        self.time_series = pd.concat(list_dataframe)



"""
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
print("Begin initialization")
BDclim.qc_initilization()
print("End initialization")
print("Begin check_duplicates_in_index")
BDclim.qc_check_duplicates_in_index()
print("End check_duplicates_in_index")
print("Begin qc_resample_index")
BDclim.qc_resample_index()
print("End qc_resample_index")
print("Begin qc_check_duplicates_in_index")
BDclim.qc_check_duplicates_in_index()
print("End qc_check_duplicates_in_index")
print("Begin qc_calm_criteria")
BDclim.qc_calm_criteria()
print("End qc_calm_criteria")
print("Begin qc_true_north")
BDclim.qc_true_north()
print("End qc_true_north")
print("Begin qc_removal_unphysical_values")
BDclim.qc_removal_unphysical_values()
print("End qc_removal_unphysical_values")
print("Begin qc_get_wind_speed_resolution")
BDclim.qc_get_wind_speed_resolution()
print("End qc_get_wind_speed_resolution")
print("Begin qc_get_wind_direction_resolution")
BDclim.qc_get_wind_direction_resolution()
print("End qc_get_wind_direction_resolution")
print("Begin qc_constant_sequences")
dict_constant_sequence = BDclim.qc_constant_sequences()
print("End qc_constant_sequences")
print("Begin qc_excessive_MISS")
BDclim.qc_excessive_MISS(dict_constant_sequence)
print("End qc_excessive_MISS")
print("Begin qc_get_stats_cst_seq")
dict_all_stations = BDclim.qc_get_stats_cst_seq(dict_constant_sequence, amplification_factor=1.5)
print("End qc_get_stats_cst_seq")
print("Begin qc_apply_stats_cst_seq")
BDclim.qc_apply_stats_cst_seq(dict_constant_sequence, dict_all_stations)
print("End qc_apply_stats_cst_seq")
print("Begin qc_get_nearest_neigbhors")
BDclim.qc_get_nearest_neigbhors()
print("End qc_get_nearest_neigbhors")
print("Begin qc_ra")
BDclim.qc_ra(dict_constant_sequence, dict_all_stations)
print("End qc_ra")
print("Begin qc_high_variability")
BDclim.qc_high_variability()
print("End qc_high_variability")



wind_speed = 'vw10m(m/s)'
import matplotlib.pyplot as plt
for station in time_series["name"].unique():
    plt.figure()
    time_serie_station = time_series[time_series["name"] == station]
    time_serie_station[wind_speed].plot(marker='x')
    time_serie_station[wind_speed][time_serie_station['validity'] == 0].plot(marker='x', linestyle='')
    plt.title(station)
"""


















