import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from datetime import datetime
from collections import defaultdict

try:
    from shapely.geometry import Point
    from shapely.geometry import Polygon

    _shapely_geometry = True
except ModuleNotFoundError:
    _shapely_geometry = False

try:
    import concurrent.futures

    _concurrent = True
except:
    _concurrent = False

try:
    import geopandas as gpd

    _geopandas = True
except:
    _geopandas = False

try:
    import dask

    _dask = True
except ModuleNotFoundError:
    _dask = False

from .data_2D import Data_2D
from ..utils.decorators import print_func_executed_decorator, timer_decorator
from ..utils.context_managers import print_all_context


class Observation:
    _shapely_geometry = _shapely_geometry
    _concurrent = _concurrent
    geopandas = _geopandas

    def __init__(self, path_to_list_stations=None, path_to_time_series=None, prm={}):

        select_date_time_serie = prm.get("select_date_time_serie")
        GPU = prm.get("GPU")

        with print_all_context("Observation", level=0, unit="second", verbose=prm.get("verbose", None)):

            # Dates
            self.begin = prm.get("begin")
            self.end = prm.get("end")

            # KNN from NWP
            self._is_updated_with_KNN_from_NWP = False

            # Paths
            self._add_all_stations_paths(path_to_list_stations, path_to_time_series,
                                         prm.get("path_vallot"),
                                         prm.get("path_saint_sorlin"),
                                         prm.get("path_argentiere"),
                                         prm.get("path_Dome_Lac_Blanc"),
                                         prm.get("path_Col_du_Lac_Blanc"),
                                         prm.get("path_Muzelle_Lac_Blanc"),
                                         prm.get("path_Col_de_Porte"),
                                         prm.get("path_Col_du_Lautaret"),
                                         GPU=GPU)

            # Quality control
            self._qc = False
            self._qc_init = False
            self.fast_loading = prm.get("fast_loading")

            # Stations
            self.load_observation_files(type="station", path=path_to_list_stations)

            # Add additional stations
            self._add_all_stations(GPU=GPU)

            # Time series
            if self.fast_loading:
                self.fast_load(path=prm.get("path_fast_loading"), type_file="time_series")
            else:
                self.load_observation_files(type='time_series', path=path_to_time_series)
                self._select_date_time_serie() if select_date_time_serie else None

                # Add additional time series
                self._add_all_time_series(GPU=GPU)

            self._select_date_time_serie() if select_date_time_serie else None

            # Reject stations
            self._assert_equal_station()
            self._reject_stations()

            # float32
            self._downcast_dtype(oldtype='float64', newtype='float32')

    def fast_load(self, path=None, type_file="time_series", verbose=True):
        if type_file == "time_series":
            self.time_series = pd.read_pickle(path)
            print("__Used pd.read_pickle to load time series (fast method)") if verbose else None

    def replace_obs_by_QC_obs(self, prm):
        time_series_qc_all = pd.read_pickle(prm["QC_pkl"])
        filter_validity_speed = (time_series_qc_all['validity_speed'] == 1)
        filter_validity_direction = (time_series_qc_all['validity_direction'] == 1)
        time_series_qc = time_series_qc_all[filter_validity_speed & filter_validity_direction]
        assert len(time_series_qc) != len(time_series_qc_all)
        self.time_series = time_series_qc
        self._qc = True

    def delete_obs_not_passing_QC(self):

        if self._qc:
            filter_qc_speed = (self.time_series['validity_speed'] == 1)
            filter_qc_direction = (self.time_series['validity_direction'] == 1)
            self.time_series = self.time_series[filter_qc_speed & filter_qc_direction]
        else:
            print("Need to apply QC before selecting obserations passing QC")

    def select_bounding_box_around_station(self, station_name, dx, dy):
        stations = self.stations
        x_station, y_station = stations[["X", "Y"]][stations["name"] == station_name].values[0]
        return x_station - dx, y_station + dy, x_station + dx, y_station - dy

    def _assert_equal_station(self, verbose=True):

        for station in self.stations["name"].values:
            if station not in self.time_series["name"].unique():
                self.stations = self.stations[self.stations["name"] != station]

        for station in self.time_series["name"].unique():
            if station not in self.stations["name"].values:
                self.time_series = self.time_series[self.time_series["name"] != station]

        print("__Selected stations that can be found both in stations and time_series") if verbose else None

    def _reject_stations(self, verbose=True):
        stations_to_reject = ['ANTIBES-GAROUPE', 'CANNES', 'SEYNOD-AREA', 'TIGNES_SAPC', 'ST MICHEL MAUR_SAPC',
                              'FECLAZ_SAPC', 'MERIBEL BURGIN', "VAL D'I SOLAISE", 'CAP FERRAT',
                              'ALBERTVILLE', 'FREJUS', "VAL D'I BELLEVA"]

        for station in stations_to_reject:
            self.time_series = self.time_series[self.time_series["name"] != station]
            self.stations = self.stations[self.stations["name"] != station]

        self.time_series = self.time_series[np.logical_not(self.time_series["name"].isna())]
        self.stations = self.stations[np.logical_not(self.stations["name"].isna())]

        print("__Rejected specific stations") if verbose else None

    def _add_all_stations_paths(self, path_to_list_stations, path_to_time_series, path_vallot,
                                path_saint_sorlin, path_argentiere, path_Dome_Lac_Blanc,
                                path_Col_du_Lac_Blanc, path_Muzelle_Lac_Blanc,
                                path_Col_de_Porte, path_Col_du_Lautaret, GPU=False):
        if not GPU:
            self.path_to_list_stations = path_to_list_stations
            self.path_to_time_series = path_to_time_series
            self.path_vallot = path_vallot
            self.path_saint_sorlin = path_saint_sorlin
            self.path_argentiere = path_argentiere
            self.path_Dome_Lac_Blanc = path_Dome_Lac_Blanc
            self.path_Col_du_Lac_Blanc = path_Col_du_Lac_Blanc
            self.path_Muzelle_Lac_Blanc = path_Muzelle_Lac_Blanc
            self.path_Col_de_Porte = path_Col_de_Porte
            self.path_Col_du_Lautaret = path_Col_du_Lautaret

    @staticmethod
    def import_delayed_dask(use_dask=False):

        if _dask and use_dask:
            from dask import delayed

        else:
            def delayed(func):
                return func

        return delayed

    def _downcast_dtype(self, oldtype='float64', newtype='float32'):
        self.time_series.loc[:, self.time_series.dtypes == oldtype] = self.time_series.loc[:,
                                                                      self.time_series.dtypes == oldtype].astype(
            newtype)

    def _add_all_stations(self, GPU=False):
        if not GPU:
            if self.path_vallot is not None: self._add_station(name='Vallot')
            if self.path_saint_sorlin is not None: self._add_station(name='Saint-Sorlin')
            if self.path_argentiere is not None: self._add_station(name='Argentiere')
            if self.path_Dome_Lac_Blanc is not None: self._add_station(name='Dome Lac Blanc')
            if self.path_Col_du_Lac_Blanc is not None: self._add_station(name='Col du Lac Blanc')
            if self.path_Muzelle_Lac_Blanc is not None: self._add_station(name='La Muzelle Lac Blanc')
            if self.path_Col_de_Porte is not None: self._add_station(name='Col de Porte')
            if self.path_Col_du_Lautaret is not None: self._add_station(name='Col du Lautaret')

    def _add_all_time_series(self, GPU=False):
        if not GPU:
            if self.path_vallot is not None: self._add_time_serie_vallot(log_profile=True)
            if self.path_saint_sorlin is not None: self._add_time_serie_glacier(name='Saint-Sorlin', log_profile=False)
            if self.path_argentiere is not None: self._add_time_serie_glacier(name='Argentiere', log_profile=False)
            if self.path_Dome_Lac_Blanc is not None: self._add_time_serie_Col(name='Dome Lac Blanc', log_profile=True)
            if self.path_Col_du_Lac_Blanc is not None: self._add_time_serie_Col(name='Col du Lac Blanc',
                                                                                log_profile=True)
            if self.path_Muzelle_Lac_Blanc is not None: self._add_time_serie_Col(name='La Muzelle Lac Blanc',
                                                                                 log_profile=True)
            if self.path_Col_de_Porte is not None: self._add_time_serie_Col(name='Col de Porte', log_profile=False)
            if self.path_Col_du_Lautaret is not None: self._add_time_serie_Col(name='Col du Lautaret',
                                                                               log_profile=False)

    def load_observation_files(self, type=None, path=None, datetime_index=True, date_column='date', verbose=True):

        if type == 'station':
            if _shapely_geometry:
                self.stations = pd.read_csv(path)
                filter_col_du_lac_blanc = self.stations["name"] != "Col du Lac Blanc"
                filter_col_du_lautaret = self.stations["name"] != "Col du Lautaret"
                self.stations = self.stations[filter_col_du_lac_blanc & filter_col_du_lautaret]
                print(f"__Stations loaded using pd.read_csv") if verbose else None
            else:
                self.stations = pd.read_csv(path)
                list_variables_str = ['AROME_NN_0', 'index_AROME_NN_0_ref_AROME',
                                      'AROME_NN_1', 'index_AROME_NN_1_ref_AROME',
                                      'AROME_NN_2', 'index_AROME_NN_2_ref_AROME',
                                      'AROME_NN_3', 'index_AROME_NN_3_ref_AROME',
                                      'index_IGN_NN_0_cKDTree_ref_IGN', 'IGN_NN_0_cKDTree',
                                      'index_IGN_NN_1_cKDTree_ref_IGN',
                                      'IGN_NN_1_cKDTree',
                                      'index_IGN_NN_2_cKDTree_ref_IGN', 'IGN_NN_2_cKDTree',
                                      'index_IGN_NN_3_cKDTree_ref_IGN',
                                      'IGN_NN_3_cKDTree',
                                      'AROME_NN_0_interpolated',
                                      'index_AROME_NN_0_interpolated_ref_AROME_interpolated',
                                      'AROME_NN_1_interpolated',
                                      'index_AROME_NN_1_interpolated_ref_AROME_interpolated',
                                      'AROME_NN_2_interpolated',
                                      'index_AROME_NN_2_interpolated_ref_AROME_interpolated',
                                      'AROME_NN_3_interpolated',
                                      'index_AROME_NN_3_interpolated_ref_AROME_interpolated',
                                      'index_AROME_NN_0_interpolated_ref_IGN',
                                      'index_AROME_NN_1_interpolated_ref_IGN',
                                      'index_AROME_NN_2_interpolated_ref_IGN',
                                      'index_AROME_NN_3_interpolated_ref_IGN']

                # Check variable that are not present
                variable_to_remove = []
                for variable in list_variables_str:
                    if variable not in list(self.stations.columns):
                        variable_to_remove.append(variable)

                # Remove variable that are not present
                for variable in variable_to_remove:
                    list_variables_str.remove(variable)

                self.stations[list_variables_str] = self.stations[list_variables_str].apply(lambda x: x.apply(eval))
                print(
                    f"__Stations loaded using pd.read_csv and eval function to convert str into tuples") if verbose else None

        if type == 'time_series':
            self.time_series = pd.read_csv(path)
            if datetime_index: self.time_series.index = self.time_series[date_column].apply(lambda x: np.datetime64(x))
            if verbose: print(f"__Time series loaded using pd.read_csv")

    def _select_date_time_serie(self, begin=None, end=None, verbose=True):
        if (begin is None) and (end is None):
            begin = self.begin
            end = self.end
        mask = (self.time_series.index >= begin) & (self.time_series.index <= end)
        self.time_series = self.time_series[mask]
        if verbose: print("__Dates time serie selected")

    def _add_station(self, name=None):

        if name == 'Vallot':  # test2
            X = 998884.573304192
            Y = 6533967.012767595
            numposte = np.nan
            alti = 4360
            lat = 45.83972222
            lon = 6.85222222
            pb_localisation = np.nan

        if name == 'Saint-Sorlin':
            X = 948949.3641216389
            Y = 6457790.489842982
            numposte = np.nan
            alti = 2720
            lat = 45.17444
            lon = 6.17
            pb_localisation = np.nan

        if name == 'Argentiere':
            X = 1007766.7474749532
            Y = 6548636.997793528
            numposte = np.nan
            alti = 2434
            lat = 45.967699
            lon = 6.976024
            pb_localisation = np.nan

        if name == 'Dome Lac Blanc':
            X = 944102.0673463248
            Y = 6452397.4474741975
            numposte = np.nan
            alti = 2808
            lat = 45.1276528
            lon = 6.10564167
            pb_localisation = np.nan

        if name == 'La Muzelle Lac Blanc':
            X = 944534.4482722675
            Y = 6452373.408159107
            numposte = np.nan
            alti = 2722
            lat = 45.1272833
            lon = 6.1111249999999995
            pb_localisation = np.nan

        if name == 'Col du Lac Blanc':
            X = 944566.7122078383
            Y = 6452414.204145856
            numposte = np.nan
            alti = 2720
            lat = 45.1276389
            lon = 6.1115555
            pb_localisation = np.nan

        if name == 'Col de Porte':
            X = 916714.8206076204
            Y = 6469977.074058817
            numposte = np.nan
            alti = 1325
            lat = 45.295
            lon = 5.765333
            pb_localisation = np.nan

        if name == 'Col du Lautaret':
            X = 968490.046994405
            Y = 6444105.79408795
            numposte = 7.0
            alti = 2050
            lat = 45.044
            lon = 0.5
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

    def _add_time_serie_Col(self, name='Dome Lac Blanc', log_profile=False, verbose=True):

        # Select station
        if name == 'Dome Lac Blanc':
            path = self.path_Dome_Lac_Blanc
        if name == 'Col du Lac Blanc':
            path = self.path_Col_du_Lac_Blanc
        if name == 'La Muzelle Lac Blanc':
            path = self.path_Muzelle_Lac_Blanc
        if name == 'Col de Porte':
            path = self.path_Col_de_Porte
        if name == 'Col du Lautaret':
            path = self.path_Col_du_Lautaret

        # Read file
        station_df = pd.read_csv(path)

        # Index
        station_df.index = pd.to_datetime(station_df['date'])

        # Columns to fit BDclim
        if name != 'Col du Lautaret':
            station_df["name"] = name
            station_df["numposte"] = np.nan
            station_df["vwmax_dir(deg)"] = np.nan

            if name != 'Col de Porte':
                station_df["P(mm)"] = np.nan

            for variable in ['quality_speed', 'quality_obs', 'BP_mbar']:
                if variable in self.time_series.columns:
                    station_df[variable] = np.nan

            if name == 'Dome Lac Blanc':
                station_df["HTN(cm)"] = np.nan

            if 'time' in station_df.columns:
                station_df = station_df.drop('time', axis=1)

            if name == 'Dome Lac Blanc':
                alti = 2808
                lat = 45.1276528
                lon = 6.10564167
                z_wind_sensor = 8.5

            if name == 'La Muzelle Lac Blanc':
                alti = 2722
                lat = 45.1272833
                lon = 6.1111249999999995
                z_wind_sensor = 7

            if name == 'Col du Lac Blanc':
                alti = 2720
                lat = 45.1276389
                lon = 6.1115555
                z_wind_sensor = 7.5

            if name == 'Col de Porte':
                alti = 1325
                lat = 45.295
                lon = 5.765333

            station_df["lon"] = lon
            station_df["lat"] = lat
            station_df["alti"] = alti

        for variable in ['vwmax(m/s)', 'vw10m(m/s)', 'winddir(deg)', 'T2m(degC)', 'HTN(cm)']:
            if variable in station_df.columns:
                station_df[variable] = station_df[variable].apply(pd.to_numeric, errors='coerce', downcast='float')

        station_df["date"] = station_df.index

        if name == "Col de Porte":
            station_df["HTN(cm)"] = station_df["HTN(cm)"] * 100
            print("____Snow height expressed in cm at Col de Porte")

        if log_profile:
            Z0_col = 0.054
            log_profile = np.log(10 / Z0_col) / np.log(z_wind_sensor / Z0_col)
            station_df['vw10m(m/s)'] = station_df['vw10m(m/s)'] * log_profile
            print(f"___log profile at {name} obs calculated")

        self.time_series = pd.concat([self.time_series, station_df])
        if verbose: print(f"__{name} time series loaded using pd.read_csv")

    def _add_time_serie_vallot(self, log_profile=True, verbose=True):

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
        for variable in ['quality_speed', 'quality_obs']:
            if variable in self.time_series.columns:
                vallot[variable] = np.nan

        # 45°50’22.93N / 6°51’7.60E, altitude 4360 m
        vallot["lon"] = 45.83972222
        vallot["lat"] = 6.85222222
        vallot["alti"] = 4360

        # Discard duplicates
        vallot = vallot[~vallot.index.duplicated()]

        # Resample to hourly values: keep only top of hour values
        vallot = vallot.resample('1H').first()

        # Change data type
        vallot["vw10m(m/s)"] = vallot["vw10m(m/s)"].astype("float32")
        vallot["winddir(deg)"] = vallot["winddir(deg)"].astype("float32")
        vallot["T2m(degC)"] = vallot["T2m(degC)"].astype("float32")

        # The measurement height is 3m and we apply a log profile to 10m
        if log_profile:
            z0_vallot = 0.00549
            log_profile = np.log(10 / z0_vallot) / np.log(3 / z0_vallot)
            vallot['vw10m(m/s)'] = vallot['vw10m(m/s)'] * log_profile

        self.time_series = pd.concat([self.time_series, vallot])
        if verbose: print("__Vallot time series loaded using pd.read_csv")

    def _add_time_serie_glacier(self, log_profile=False, name=None, verbose=True):

        # Create a file containing all years
        glacier = []

        if name == 'Saint-Sorlin':
            for year in range(2006, 2020):
                glacier_year = pd.read_csv(self.path_saint_sorlin + f"SaintSorlin{year}-halfhourly.csv", sep=';',
                                           header=2)
                glacier.append(glacier_year)

        if name == 'Argentiere':
            for year in range(2007, 2020):
                # Corrected dates in 2018
                if year == 2018:
                    glacier_year = pd.read_csv(self.path_argentiere + f"Argentiere{year}-halfhourly_corrected.csv",
                                               sep=';', header=2)
                else:
                    glacier_year = pd.read_csv(self.path_argentiere + f"Argentiere{year}-halfhourly.csv", sep=';',
                                               header=2)
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
            # print("Found NaNs in dates: " + str(nb_nan))

        # Discard NaNs in dates
        glacier = glacier[glacier["date"].notna()]

        # Columns to fit BDclim
        glacier["name"] = name
        glacier["numposte"] = np.nan
        glacier["vwmax_dir(deg)"] = np.nan
        glacier["P(mm)"] = np.nan
        glacier["HTN(cm)"] = np.nan
        for variable in ['quality_speed', 'quality_obs']:
            if variable in self.time_series.columns:
                glacier[variable] = np.nan

        # 45°10’28.3’’N / 6°10’12.1’’E, altitude 2720 m
        glacier["lon"] = self.stations["lon"][self.stations["name"] == name].values[0]
        glacier["lat"] = self.stations["lat"][self.stations["name"] == name].values[0]
        glacier["alti"] = self.stations["alti"][self.stations["name"] == name].values[0]

        # Dates are converted to np.datetime64
        glacier['date'] = glacier['date'].apply(
            lambda x: np.datetime64(datetime.strptime(x, "%d/%m/%Y %H:%M")))

        # Create the index
        glacier.index = glacier["date"]

        # Discard duplicates
        if verbose:
            nb_duplicate = len(glacier[glacier.index.duplicated()])
            # print("Found date duplicate: " + str(nb_duplicate))
        glacier = glacier[~glacier.index.duplicated()]

        if name == 'Argentiere':
            if verbose:
                # Print number of annotated observations
                nb_annotated_observations = len(glacier[glacier["Unnamed: 7"].notna()])
                # print("Annotated observations: " + str(nb_annotated_observations))

            # Discard annotated observations
            glacier = glacier[glacier["Unnamed: 7"].isna()]
            glacier = glacier.drop("Unnamed: 7", axis=1)

        # Resample to hourly values: keep only top of hour values
        glacier = glacier.resample('1H').first()

        # Change data type
        glacier["vw10m(m/s)"] = glacier["vw10m(m/s)"].astype("float32")
        glacier["winddir(deg)"] = glacier["winddir(deg)"].astype("float32")
        glacier["T2m(degC)"] = glacier["T2m(degC)"].astype("float32")

        if verbose:
            nb_missing_dates = len(glacier.asfreq('1H').index) - len(glacier.index)
            # print("Number missing dates: " + str(nb_missing_dates))

        if log_profile:
            # Apply log profile
            if name == "Saint-Sorlin":
                z0_glacier = 0.0135
            if name == "Argentiere":
                z0_glacier = 1.015
            log_profile = np.log(10 / z0_glacier) / np.log(3 / z0_glacier)
            z0_glacier['Wind speed (m/s)'] = z0_glacier['Wind speed (m/s)'] * log_profile

        self.time_series = pd.concat([self.time_series, glacier])
        if verbose: print(f"__{name} time series loaded using pd.read_csv")

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

    @staticmethod
    def search_neighbors_using_cKDTree(mnt, x_L93, y_l93, number_of_neighbors=4):

        all_MNT_index_x, all_MNT_index_y = mnt.find_nearest_MNT_index(x_L93, y_l93)
        nb_station = len(all_MNT_index_x)

        arrays_nearest_neighbors_l93 = np.zeros((number_of_neighbors, nb_station, 2))
        arrays_nearest_neighbors_index = np.zeros((number_of_neighbors, nb_station, 2))
        arrays_nearest_neighbors_delta_x = np.zeros((number_of_neighbors, nb_station))

        for idx_station in range(nb_station):
            l93_station_x, l93_station_y = x_L93.values[idx_station], y_l93.values[idx_station]
            index_MNT_x = np.intp(all_MNT_index_x[idx_station])
            index_MNT_y = np.intp(all_MNT_index_y[idx_station])

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

        return arrays_nearest_neighbors_l93, arrays_nearest_neighbors_index, arrays_nearest_neighbors_delta_x

    def update_stations_with_KNN_from_NWP(self, nwp=None, number_of_neighbors=4,
                                          data_xr=None, name=None, interpolated=False):
        """
        Update a Observations.station (DataFrame) with index of nearest neighbors in nwp

        ex: BDclim.update_stations_with_KNN_from_NWP(4, AROME) gives information about the 4 KNN at the
        each observation station from AROME
        """

        nwp_data_xr = nwp.data_xr if data_xr is None else data_xr
        name = nwp.name if name is None else name
        height = nwp.height if data_xr is None else nwp_data_xr.yy.shape[0]
        length = nwp.length if data_xr is None else nwp_data_xr.xx.shape[0]
        interp_str = '' if not interpolated else '_interpolated'

        def K_N_N_point(point):
            distance, idx = tree.query(point, k=number_of_neighbors)
            return distance, idx

        # Reference stations
        list_coord_station = zip(self.stations['X'].values, self.stations['Y'].values)

        # Coordinates where to find neighbors
        stacked_xy = Data_2D.x_y_to_stacked_xy(nwp_data_xr["X_L93"], nwp_data_xr["Y_L93"])
        grid_flat = Data_2D.grid_to_flat(stacked_xy)
        tree = cKDTree(grid_flat)

        # Parallel computation of nearest neighbors
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                list_nearest = executor.map(K_N_N_point, list_coord_station)
            print("Parallel computation worked for update_stations_with_KNN_from_NWP\n")
        except:
            print("Parallel computation using concurrent.futures didn't work, "
                  "so update_stations_with_KNN_from_NWP will not be parallelized.\n")
            list_nearest = map(K_N_N_point, list_coord_station)

        # Store results as array
        list_nearest = np.array([np.array(station) for station in list_nearest])
        list_index = [(x, y) for x in range(height) for y in range(length)]

        # Update DataFrame
        for neighbor in range(number_of_neighbors):
            self.stations[f'delta_x_{name}_NN_{neighbor}{interp_str}'] = list_nearest[:, 0, neighbor]
            self.stations[f'{name}_NN_{neighbor}{interp_str}'] = [grid_flat[int(index)] for index in
                                                                  list_nearest[:, 1, neighbor]]
            name_str = f'index_{name}_NN_{neighbor}{interp_str}_ref_{name}{interp_str}'
            self.stations[name_str] = [list_index[int(index)] for index in list_nearest[:, 1, neighbor]]

        self._is_updated_with_KNN_from_NWP = True

    def update_stations_with_KNN_from_MNT(self, mnt):
        index_x_MNT, index_y_MNT = mnt.find_nearest_MNT_index(self.stations["X"], self.stations["Y"])
        self.stations[f"index_X_NN_{mnt.name}_ref_{mnt.name}"] = index_x_MNT
        self.stations[f"index_Y_NN_{mnt.name}_ref_{mnt.name}"] = index_y_MNT

    def update_stations_with_KNN_from_MNT_using_cKDTree(self, mnt, number_of_neighbors=4):

        nn_l93, nn_index, nn_delta_x = self.search_neighbors_using_cKDTree(mnt, self.stations["X"], self.stations["Y"],
                                                                           number_of_neighbors=number_of_neighbors)

        mnt_name = mnt.name
        for neighbor in range(number_of_neighbors):
            name_str = f"index_{mnt_name}_NN_{neighbor}_cKDTree_ref_{mnt_name}"
            self.stations[name_str] = [tuple(index) for index in nn_index[neighbor, :]]
            self.stations[f"{mnt_name}_NN_{neighbor}_cKDTree"] = [tuple(coord) for coord in nn_l93[neighbor, :]]
            self.stations[f"delta_x_{mnt_name}_NN_{neighbor}_cKDTree"] = nn_delta_x[neighbor, :]

    def update_stations_with_KNN_of_NWP_in_MNT_using_cKDTree(self, mnt, nwp,
                                                             interpolated=False, number_of_neighbors=4):

        interp_str = "_interpolated" if interpolated else ""
        mnt_name = mnt.name
        nwp_name = nwp.name

        for neighbor in range(number_of_neighbors):
            x_str = self.stations[f"{nwp_name}_NN_{neighbor}{interp_str}"].str[0]
            y_str = self.stations[f"{nwp_name}_NN_{neighbor}{interp_str}"].str[1]
            _, nn_index, _ = self.search_neighbors_using_cKDTree(mnt, x_str, y_str,
                                                                 number_of_neighbors=number_of_neighbors)

            name_str = f'index_{nwp_name}_NN_{neighbor}{interp_str}_ref_{mnt_name}'
            self.stations[name_str] = [tuple(index) for index in nn_index[neighbor, :]]

    def extract_MNT_around_station(self, station, mnt, nb_pixel_x, nb_pixel_y):
        condition = self.stations["name"] == station
        (index_x, index_y) = self.stations[[f"index_{mnt.name}_NN_0_cKDTree_ref_{mnt.name}"]][condition].values[0][0]
        index_x, index_y = np.int32(index_x), np.int32(index_y)
        MNT_data = mnt.data[index_y - nb_pixel_y:index_y + nb_pixel_y, index_x - nb_pixel_x:index_x + nb_pixel_x]
        MNT_x = mnt.data_xr.x.data[index_x - nb_pixel_x:index_x + nb_pixel_x]
        MNT_y = mnt.data_xr.y.data[index_y - nb_pixel_y:index_y + nb_pixel_y]
        return MNT_data, MNT_x, MNT_y

    def extract_MNT_around_nwp_neighbor(self, station, mnt, nwp, nb_pixel_x, nb_pixel_y, interpolated=False):
        condition = self.stations["name"] == station
        interp_str = "_interpolated" if interpolated else ""
        idx_nwp_in_mnt = f"index_{nwp.name}_NN_0{interp_str}_ref_{mnt.name}"

        (index_x, index_y) = self.stations[idx_nwp_in_mnt][condition].values[0]
        index_x, index_y = int(index_x), int(index_y)
        MNT_data = mnt.data[index_y - nb_pixel_y:index_y + nb_pixel_y, index_x - nb_pixel_x:index_x + nb_pixel_x]
        MNT_x = mnt.data_xr.x.data[index_x - nb_pixel_x:index_x + nb_pixel_x]
        MNT_y = mnt.data_xr.y.data[index_y - nb_pixel_y:index_y + nb_pixel_y]
        return MNT_data, MNT_x, MNT_y

    def extract_MNT(self, mnt, nb_pixel_x, nb_pixel_y, nwp=None, station="Col du Lac Blanc", extract_around="station"):

        if extract_around == "station":
            MNT_data, MNT_x, MNT_y = self.extract_MNT_around_station(station, mnt, nb_pixel_x, nb_pixel_y)

        elif extract_around == "nwp_neighbor":
            MNT_data, MNT_x, MNT_y = self.extract_MNT_around_nwp_neighbor(station, mnt, nwp, nb_pixel_x, nb_pixel_y,
                                                                          interpolated=False)

        elif extract_around == "nwp_neighbor_interp":
            MNT_data, MNT_x, MNT_y = self.extract_MNT_around_nwp_neighbor(station, mnt, nwp, nb_pixel_x, nb_pixel_y,
                                                                          interpolated=True)

        return MNT_data, MNT_x, MNT_y

    @staticmethod
    def _degToCompass(num):
        if np.isnan(num):
            return np.nan

        else:
            val = int((num / 22.5) + .5)
            arr = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
            return arr[(val % 16)]

    @print_func_executed_decorator("initialization")
    @timer_decorator("initialization", unit="minute")
    def qc_initialization(self, wind_direction='winddir(deg)'):

        time_series = self.time_series

        # Create UV_DIR
        time_series["UV_DIR"] = time_series[wind_direction]

        # Create validity
        time_series["validity_speed"] = 1
        time_series["validity_direction"] = 1

        # Create validity
        time_series["last_flagged_speed"] = 0
        time_series["last_flagged_direction"] = 0
        time_series["last_unflagged_speed"] = 0
        time_series["last_unflagged_direction"] = 0

        # Create resolution
        time_series['resolution_speed'] = np.nan
        time_series['resolution_direction'] = np.nan

        # Create qc_2 for excessive_MISS
        time_series['qc_2_speed'] = 1
        time_series['qc_2_direction'] = 1

        # qc_cst_sequence
        time_series['qc_3_speed'] = 1
        time_series['qc_3_direction'] = 1
        time_series['qc_3_direction_pref'] = 0

        # qc_cst_sequence
        time_series['preferred_direction_during_sequence'] = np.nan

        # High variability
        time_series["qc_5_speed"] = 1
        time_series["qc_high_variability_criteria"] = np.nan

        # Bias
        time_series["qc_6_speed"] = 1

        # Bias
        time_series["qc_7_isolated_records_speed"] = 1
        time_series["qc_7_isolated_records_direction"] = 1

        # N, NNE, NE etc
        time_series["cardinal"] = [self._degToCompass(direction) for direction in time_series[wind_direction].values]

        self.time_series = time_series
        self._qc_init = True

    @print_func_executed_decorator("check_duplicates_in_index")
    @timer_decorator("check_duplicates_in_index", unit="minute")
    def qc_check_duplicates_in_index(self, print_duplicated_dates=False):
        """
        Quality control

        This function looks for duplicated dates in observations index
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        nb_problem = 0

        for station in list_stations:
            filter = time_series["name"] == station
            time_series_station = time_series[filter]

            if time_series_station.index.duplicated().sum() > 0:
                print("Found duplicated index")
                print(station)
                nb_problem += 1

                if print_duplicated_dates:
                    print(time_series_station[time_series_station.index.duplicated()].index)
            else:
                pass

        print(f"..Found {nb_problem} duplicated dates")

    def _qc_resample_index(self, time_series, station, frequency):
        filter = time_series["name"] == station
        time_series_station = time_series[filter].asfreq(frequency)
        time_series_station["name"] = station
        return time_series_station

    @print_func_executed_decorator("resample_index")
    @timer_decorator("resample_index", unit="minute")
    def qc_resample_index(self, frequency='1H'):
        """
        Quality control

        This function fill NaN at missing dates in index
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []
        for station in list_stations:
            time_series_station = self._qc_resample_index(time_series, station, frequency)
            list_dataframe.append(time_series_station)
        self.time_series = pd.concat(list_dataframe)

    def _qc_get_wind_speed_resolution(self, time_series, station, wind_speed):

        # Select station
        filter = time_series["name"] == station
        time_series_station = time_series[filter]
        for _, wind_per_day in time_series_station[wind_speed].groupby(pd.Grouper(freq='D')):

            # Check for resolution
            resolution_found = False
            decimal = 0
            while not resolution_found:

                wind_array = wind_per_day.values
                wind_round_array = wind_per_day.round(decimal).values

                if np.allclose(wind_array, wind_round_array, equal_nan=True):
                    in_day = time_series_station.index.isin(wind_per_day.index)
                    time_series_station['resolution_speed'][in_day] = decimal
                    resolution_found = True
                else:
                    decimal += 1

                if decimal >= 4:
                    in_day = time_series_station.index.isin(wind_per_day.index)
                    time_series_station['resolution_speed'][in_day] = decimal
                    resolution_found = True

        # Check that high resolution are not mistaken as low resolution
        resolution = 5
        nb_obs = len(time_series_station['resolution_speed'])

        while resolution >= 0:

            filter = time_series_station['resolution_speed'] == resolution
            nb_not_nans = time_series_station['resolution_speed'][filter].count()

            if nb_not_nans > 0.8 * nb_obs:
                time_series_station['resolution_speed'] = resolution
                resolution = -1
            else:
                resolution = resolution - 1

        return time_series_station

    @print_func_executed_decorator("get_wind_speed_resolution")
    @timer_decorator("get_wind_speed_resolution", unit="minute")
    def qc_get_wind_speed_resolution(self, wind_speed='vw10m(m/s)', use_dask=False, verbose=True):
        """
        Quality control

        This function determines the resolution of wind speed observations
        Possible resolutions are 1m/s, 0.1m/s, 0.01m/s, 0.001m/s, 0.0001m/s
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []
        for station in list_stations:
            time_series_station = self._qc_get_wind_speed_resolution(time_series, station, wind_speed)

            # Add station to list of dataframe
            list_dataframe.append(time_series_station)
        if verbose:
            print("__Speed resolution found")
            print("__Looked for outliers in speed resolution")
        self.time_series = pd.concat(list_dataframe)

    @print_func_executed_decorator("get_wind_direction_resolution")
    @timer_decorator("get_wind_direction_resolution", unit="minute")
    def qc_get_wind_direction_resolution(self, wind_direction='winddir(deg)', verbose=True):
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
                wind_array = wind_per_day[1].values
                resolution_found = False
                resolutions = [10, 5, 1, 0.1]
                index = 0
                while not resolution_found:
                    resolution = resolutions[index]
                    wind_round_array = np.around(wind_array / resolution, decimals=0) * resolution

                    if np.allclose(wind_array, wind_round_array, equal_nan=True):
                        time_series_station['resolution_direction'][
                            time_series_station.index.isin(wind_per_day[1].index)] = resolution
                        resolution_found = True
                    else:
                        index += 1

                    if index >= 3:
                        resolution = resolutions[index]
                        time_series_station['resolution_direction'][
                            time_series_station.index.isin(wind_per_day[1].index)] = resolution
                        resolution_found = True

            # Check that high resolution are not mistaken as low resolution
            # If at a station, a minimum of 80% of wind resolutions are with at the same resolution, we consider
            # that other resolution detected is a misdetection. Consequently, we discard such cases
            resolutions = [10, 5, 1, 0.1]
            index = 0
            nb_obs = len(time_series_station['resolution_direction'])
            while index <= 3:
                resolution = resolutions[index]
                filter = time_series_station['resolution_direction'] == resolution
                nb_not_nans = time_series_station['resolution_direction'][filter].count()
                # todo to improve: check that other resolution are unique or not. If they are uniqe (e.g. if main resolution is 10°
                # check that secondary resolution is indeed fixes). If so look if it is concentrated on a signle period. If so, keep it
                if nb_not_nans > 0.8 * nb_obs:
                    time_series_station['resolution_direction'] = resolution
                    index = 100
                else:
                    index = index + 1

            # Add station to list of dataframe
            list_dataframe.append(time_series_station)
        if verbose:
            print("__Direction resolution found")
            print("__Looked for outliers in direction resolution")
        self.time_series = pd.concat(list_dataframe)

    @print_func_executed_decorator("calm_criteria")
    @timer_decorator("calm_criteria", unit="minute")
    def qc_calm_criteria(self, wind_speed='vw10m(m/s)', verbose=True):
        """
        Quality control

        This function apply calm criteria
        UV = 0m/s => UV_DIR = 0°
        """

        # Calm criteria: UV = 0m/s => UV_DIR = 0°
        self.time_series["UV_DIR"][self.time_series[wind_speed] == 0] = 0

        if verbose: print("__Calm criteria applied. Now if UV=0, UV_DIR=0")

    @print_func_executed_decorator("true_north")
    @timer_decorator("true_north", unit="minute")
    def qc_true_north(self, wind_speed='vw10m(m/s)', wind_direction='winddir(deg)', verbose=True):
        """
        Quality control

        This function apply true north criteria
        UV != 0m/s and UV_DIR=0 => UV_DIR = 360
        """
        # True north criteria: UV != 0m/s and UV_DIR = 0 => UV_DIR = 360
        filter_1 = (self.time_series[wind_speed] != 0)
        filter_2 = (self.time_series[wind_direction] == 0)
        self.time_series["UV_DIR"][filter_1 & filter_2] = 360
        if verbose:
            print("__True north criteria applied. Now if UV!=0 and UV_DIR=0 => UV_DIR=360")
            print("__North=360, no wind=0")

    @print_func_executed_decorator("removal_unphysical_values")
    @timer_decorator("removal_unphysical_values", unit="minute")
    def qc_removal_unphysical_values(self, wind_speed='vw10m(m/s)', wind_direction='winddir(deg)'):
        """
        Quality control

        This function flags unphysical values

        Unphysical speed: UV < 0 or UV > 100
        Unphysical direction: UV_DIR < 0 or UV_DIR > 360
        """

        # Specify result of the test
        self.time_series["qc_1"] = 1
        self.time_series["qc_1_speed"] = 1
        self.time_series["qc_1_direction"] = 1

        # Calm criteria: UV = 0m/s => UV_DIR = 0°
        filter_1 = (self.time_series[wind_speed] < 0)
        filter_2 = (self.time_series[wind_speed] > 100)
        filter_3 = (self.time_series[wind_direction] < 0)
        filter_4 = (self.time_series[wind_direction] > 360)

        self.time_series['validity_speed'][(filter_1 | filter_2)] = 0
        self.time_series['validity_direction'][(filter_3 | filter_4)] = 0
        self.time_series["qc_1_speed"][(filter_1 | filter_2)] = "unphysical_wind_speed"
        self.time_series["last_flagged_speed"][(filter_1 | filter_2)] = "unphysical_wind_speed"
        self.time_series["qc_1_direction"][(filter_3 | filter_4)] = "unphysical_wind_direction"
        self.time_series["last_flagged_direction"][(filter_3 | filter_4)] = "unphysical_wind_direction"
        self.time_series["qc_1"][(filter_1 | filter_2) & (filter_3 | filter_4)] = "unphysical_wind_speed_and_direction"

    @print_func_executed_decorator("constant_sequences")
    @timer_decorator("constant_sequences", unit="minute")
    def qc_constant_sequences(self, wind_speed='vw10m(m/s)', wind_direction='winddir(deg)', tolerance_speed=0.08,
                              tolerance_direction=1, verbose=True):
        """
        Quality control

        This function detect constant sequences and their lengths
        """
        if verbose: print(
            f"__Tolerance for constant sequence detection: speed={tolerance_speed} m/s, direction={tolerance_direction}degree")

        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []

        # Initialize dictionary direction
        dict_cst_seq = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        for station in list_stations:
            # Direction
            for variable in ["length_constant_direction", "date_constant_direction_begin",
                             "date_constant_direction_end"]:
                for resolution in [10, 5, 1, 0.1]:
                    for wind in ["=0", "!=0"]:
                        dict_cst_seq[station][variable][str(resolution)][wind] = []
            # Speed
            for variable in ["length_constant_speed", "date_constant_speed_begin", "date_constant_speed_end"]:
                for resolution in range(5):
                    for wind in ["<1m/s", ">=1m/s"]:
                        dict_cst_seq[station][variable][str(resolution)][wind] = []

        for station in list_stations:

            # Select station
            filter = time_series["name"] == station
            time_series_station = time_series[filter]

            # Inputs as arrays
            speed_values = time_series_station[wind_speed].values
            dir_values = time_series_station[wind_direction].values
            dates = time_series_station.index.values
            resolution_speed = time_series_station["resolution_speed"].values
            resolution_dir = time_series_station["resolution_direction"].values
            nb_values = len(speed_values)

            # Outputs
            constant_speed = np.zeros(nb_values)
            constant_direction = np.zeros(nb_values)

            # Lengths
            lenght_speed = 0
            lenght_direction = 0

            # Begin with non constant sequence
            previous_step_speed_is_constant = False
            previous_step_direction_is_constant = False

            # Tolerance for constant sequence detection
            tolerance_speed = tolerance_speed
            tolerance_direction = tolerance_direction

            resolution_cst_seq = []
            speed_sequence = []
            resolution_cst_seq_direction = []
            direction_sequence = []
            for index in range(nb_values - 1):

                # Constant speed sequences
                nan_detected = np.isnan(speed_values[index])
                if nan_detected:
                    constant_speed[index] = np.nan

                    if previous_step_speed_is_constant:
                        resolution = str(np.bincount(resolution_cst_seq).argmax())
                        key = ">=1m/s" if np.mean(speed_sequence) >= 1 else "<1m/s"
                        dict_cst_seq[station]["length_constant_speed"][resolution][key].append(lenght_speed)
                        dict_cst_seq[station]["date_constant_speed_end"][resolution][key].append(dates[index])
                        dict_cst_seq[station]["date_constant_speed_begin"][resolution][key].append(date_begin)

                    previous_step_speed_is_constant = False
                    resolution_cst_seq = []
                    speed_sequence = []

                # if not nan
                else:

                    not_constant_sequence = np.abs(speed_values[index] - speed_values[index + 1]) > tolerance_speed
                    next_step_is_nan = np.isnan(speed_values[index + 1])

                    if not_constant_sequence or next_step_is_nan:
                        constant_speed[index + 1] = 0

                        # If the previous sequence was constant and is finished
                        if previous_step_speed_is_constant:
                            resolution = str(np.bincount(resolution_cst_seq).argmax())
                            key = ">=1m/s" if np.mean(speed_sequence) >= 1 else "<1m/s"
                            dict_cst_seq[station]["length_constant_speed"][resolution][key].append(lenght_speed)
                            dict_cst_seq[station]["date_constant_speed_end"][resolution][key].append(dates[index])
                            dict_cst_seq[station]["date_constant_speed_begin"][resolution][key].append(date_begin)
                        previous_step_speed_is_constant = False
                        resolution_cst_seq = []
                        speed_sequence = []

                    # If constant sequence
                    else:
                        constant_speed[index:index + 2] = 1
                        resolution_cst_seq.append(resolution_speed[index])
                        speed_sequence.append(speed_values[index])

                        # If the previous sequence was constant and continues
                        lenght_speed = lenght_speed + 1 if previous_step_speed_is_constant else 2

                        if not previous_step_speed_is_constant:
                            date_begin = dates[index]

                        previous_step_speed_is_constant = True

                        # If the time_serie end with a constant sequence
                        if index == (nb_values - 2):
                            resolution = str(np.bincount(resolution_cst_seq).argmax())
                            key = ">=1m/s" if np.mean(speed_sequence) >= 1 else "<1m/s"
                            dict_cst_seq[station]["length_constant_speed"][resolution][key].append(lenght_speed)
                            dict_cst_seq[station]["date_constant_speed_end"][resolution][key].append(dates[index])
                            dict_cst_seq[station]["date_constant_speed_begin"][resolution][key].append(date_begin)

                # Constant direction sequences
                if np.isnan(dir_values[index]):
                    constant_direction[index] = np.nan

                    if previous_step_direction_is_constant:
                        resolutions_mult_by_ten = [10 * value for value in resolution_cst_seq_direction]
                        most_freq_val = np.bincount(resolutions_mult_by_ten).argmax() / 10
                        most_freq_val = str(int(most_freq_val)) if most_freq_val >= 1 else str(most_freq_val)

                        key = "=0" if np.mean(direction_sequence) == 0 else "!=0"
                        dict_cst_seq[station]["length_constant_direction"][most_freq_val][key].append(lenght_direction)
                        dict_cst_seq[station]["date_constant_direction_end"][most_freq_val][key].append(dates[index])
                        dict_cst_seq[station]["date_constant_direction_begin"][most_freq_val][key].append(
                            date_begin_direction)

                    previous_step_direction_is_constant = False
                    resolution_cst_seq_direction = []
                    direction_sequence = []

                not_constant_sequence = np.abs(dir_values[index] - dir_values[index + 1]) > tolerance_direction
                next_step_is_nan = np.isnan(dir_values[index + 1])

                if not_constant_sequence or next_step_is_nan:

                    constant_direction[index + 1] = 0

                    # If the previous sequence was constant and is finished
                    if previous_step_direction_is_constant:
                        resolutions_mult_by_ten = [10 * value for value in resolution_cst_seq_direction]
                        most_freq_val = np.bincount(resolutions_mult_by_ten).argmax() / 10
                        most_freq_val = str(int(most_freq_val)) if most_freq_val >= 1 else str(most_freq_val)

                        key = "=0" if np.mean(direction_sequence) == 0 else "!=0"
                        dict_cst_seq[station]["length_constant_direction"][most_freq_val][key].append(lenght_direction)
                        dict_cst_seq[station]["date_constant_direction_end"][most_freq_val][key].append(dates[index])
                        dict_cst_seq[station]["date_constant_direction_begin"][most_freq_val][key].append(
                            date_begin_direction)

                    previous_step_direction_is_constant = False
                    resolution_cst_seq_direction = []
                    direction_sequence = []
                else:
                    constant_direction[index:index + 2] = 1
                    resolution_cst_seq_direction.append(resolution_dir[index])
                    direction_sequence.append(dir_values[index])

                    # If the previous sequence was constant and continues
                    lenght_direction = lenght_direction + 1 if previous_step_direction_is_constant else 2
                    if not previous_step_direction_is_constant:
                        date_begin_direction = dates[index]

                    # If the time_serie end with a constant sequence
                    if index == nb_values - 2:
                        resolutions_mult_by_ten = [10 * value for value in resolution_cst_seq_direction]
                        most_freq_val = np.bincount(resolutions_mult_by_ten).argmax() / 10
                        most_freq_val = str(int(most_freq_val)) if most_freq_val >= 1 else str(most_freq_val)

                        key = "=0" if np.mean(direction_sequence) == 0 else "!=0"
                        dict_cst_seq[station]["length_constant_direction"][most_freq_val][key].append(lenght_direction)
                        dict_cst_seq[station]["date_constant_direction_end"][most_freq_val][key].append(dates[index])
                        dict_cst_seq[station]["date_constant_direction_begin"][most_freq_val][key].append(
                            date_begin_direction)

                    previous_step_direction_is_constant = True

            # Specify result of the test
            time_series_station["constant_speed"] = constant_speed
            time_series_station["constant_direction"] = constant_direction

            # Add station to list of dataframe
            list_dataframe.append(time_series_station)

        self.time_series = pd.concat(list_dataframe)

        return dict_cst_seq

    @print_func_executed_decorator("excessive_MISS")
    @timer_decorator("excessive_MISS", unit="minute")
    def qc_excessive_MISS(self, dict_constant_sequence, percentage_miss=0.9,
                          wind_speed='vw10m(m/s)', wind_direction='winddir(deg)', verbose=True):
        """
        Quality control

        This function detect suspicious constant sequences based on the number of missing values.

        Wind speed and direction
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []
        for station in list_stations:

            # Select station
            filter = time_series["name"] == station
            time_series_station = time_series[filter]

            # Wind speed
            rolling_speed = time_series_station[wind_speed].isna().rolling('1D').sum()
            for resolution in range(5):
                for speed in ['<1m/s', '>=1m/s']:

                    # Select constant sequences
                    begins = dict_constant_sequence[station]["date_constant_speed_begin"][str(resolution)][speed]
                    ends = dict_constant_sequence[station]["date_constant_speed_end"][str(resolution)][speed]

                    for begin, end in zip(begins, ends):

                        missing_period_speed = 0

                        nb_nan_current_seq = time_series_station[wind_speed][begin:end].isna().sum()
                        nb_obs_current_seq = time_series_station[wind_speed][begin:end].count()

                        # If current sequence has many nans
                        if nb_nan_current_seq > percentage_miss * nb_obs_current_seq:
                            missing_period_speed += 1

                        # If previous sequence has many nans
                        try:
                            if rolling_speed[begin - np.timedelta64(1, 'D')] > percentage_miss * 24:
                                missing_period_speed += 1
                        except KeyError:
                            pass

                        # If next sequence has many nans
                        if rolling_speed[end] > percentage_miss * 24:
                            missing_period_speed += 1

                        # If two sequences or more have many nans we flag it
                        if missing_period_speed >= 2:
                            time_series_station['validity_speed'][begin:end] = 0
                            time_series_station['last_flagged_speed'][begin:end] = 'excessive_miss_speed'
                            time_series_station['qc_2_speed'][begin:end] = 'excessive_miss_speed'

            # Wind direction
            rolling_direction = time_series_station[wind_direction].isna().rolling('1D').sum()
            for resolution in [10, 5, 1, 0.1]:

                # Select constant sequences
                begins = dict_constant_sequence[station]["date_constant_direction_begin"][str(resolution)]['!=0']
                ends = dict_constant_sequence[station]["date_constant_direction_end"][str(resolution)]['!=0']
                for begin, end in zip(begins, ends):

                    missing_period_direction = 0

                    nb_nan_current_seq = time_series_station[wind_direction][begin:end].isna().sum()
                    nb_obs_current_seq = time_series_station[wind_direction][begin:end].count()

                    # If current sequence has many nans
                    if nb_nan_current_seq > percentage_miss * nb_obs_current_seq:
                        missing_period_direction += 1

                    # If previous sequence has many nans
                    try:
                        if rolling_direction[begin - np.timedelta64(1, 'D')] > percentage_miss * 24:
                            missing_period_direction += 1
                    except KeyError:
                        pass

                    # If next sequence has many nans
                    if rolling_direction[end] > percentage_miss * 24:
                        missing_period_direction += 1

                    # If two sequences or more have many nans we flag it
                    if missing_period_speed >= 2:
                        time_series_station['validity_direction'][begin:end] = 0
                        time_series_station['last_flagged_direction'][begin:end] = 'excessive_miss_direction'
                        time_series_station['qc_2_direction'][begin:end] = 'excessive_miss_direction'

            # Add station to list of dataframe
            list_dataframe.append(time_series_station)

        if verbose: print(f"__Excessive miss during cst sequences. Percentage miss: {percentage_miss}")

        self.time_series = pd.concat(list_dataframe)

    @print_func_executed_decorator("get_stats_cst_seq")
    @timer_decorator("get_stats_cst_seq", unit="minute")
    def qc_get_stats_cst_seq(self, dict_constant_sequence, amplification_factor_speed=1,
                             amplification_factor_direction=1, verbose=True):
        """
        Quality control

        This function detect compute statistics used to flag constant sequences
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()

        dict_all_stations = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

        # Wind speed resolution
        for resolution in range(5):
            dict_all_stations["length_constant_speed"][str(resolution)][">=1m/s"]["values"] = []
            dict_all_stations["length_constant_speed"][str(resolution)]["<1m/s"]["values"] = []

        # Wind direction resolution
        for resolution in [10, 5, 1, 0.1]:
            dict_all_stations["length_constant_direction"][str(resolution)]["!=0"]["values"] = []

        # riteria min and max
        criteria_min = 4
        criteria_max = 12

        # Reconstruct dictionary without station
        for station in list_stations:

            # Speed
            for resolution in range(5):
                resolution = str(resolution)
                array_1 = dict_constant_sequence[station]["length_constant_speed"][resolution]["<1m/s"]
                array_2 = dict_constant_sequence[station]["length_constant_speed"][resolution][">=1m/s"]
                if np.array(array_1).size != 0:
                    dict_all_stations["length_constant_speed"][resolution]["<1m/s"]["values"].extend(array_1)
                if np.array(array_2).size != 0:
                    dict_all_stations["length_constant_speed"][resolution][">=1m/s"]["values"].extend(array_2)

            # Direction
            for resolution in [10, 5, 1, 0.1]:
                resolution = str(resolution)
                array_1 = dict_constant_sequence[station]["length_constant_direction"][resolution]["!=0"]
                if np.array(array_1).size != 0: dict_all_stations["length_constant_direction"][resolution]["!=0"][
                    "values"].extend(array_1)

        # Statistics speed
        for resolution in range(5):
            resolution = str(resolution)
            for speed in ["<1m/s", ">=1m/s"]:
                values = dict_all_stations["length_constant_speed"][resolution][speed]["values"]

                P95 = np.quantile(values, 0.95) if np.array(values).size != 0 else None
                P75 = np.quantile(values, 0.75) if np.array(values).size != 0 else None
                P25 = np.quantile(values, 0.25) if np.array(values).size != 0 else None

                dict_all_stations["length_constant_speed"][resolution][speed]["stats"]["P95"] = P95
                dict_all_stations["length_constant_speed"][resolution][speed]["stats"]["P75"] = P75
                dict_all_stations["length_constant_speed"][resolution][speed]["stats"]["P25"] = P25

        # Statistics direction
        for resolution in [10, 5, 1, 0.1]:
            resolution = str(resolution)
            values = dict_all_stations["length_constant_direction"][resolution]["!=0"]["values"]
            P95 = np.quantile(values, 0.95) if np.array(values).size != 0 else None
            P75 = np.quantile(values, 0.75) if np.array(values).size != 0 else None
            P25 = np.quantile(values, 0.25) if np.array(values).size != 0 else None
            dict_all_stations["length_constant_direction"][resolution]["!=0"]["stats"]["P95"] = P95
            dict_all_stations["length_constant_direction"][resolution]["!=0"]["stats"]["P75"] = P75
            dict_all_stations["length_constant_direction"][resolution]["!=0"]["stats"]["P25"] = P25

        # Criterion speed
        for resolution in range(5):
            resolution = str(resolution)

            # Select values
            values_1 = dict_all_stations["length_constant_speed"][resolution]["<1m/s"]["values"]
            values_2 = dict_all_stations["length_constant_speed"][resolution][">=1m/s"]["values"]
            try:
                P95_low = dict_all_stations["length_constant_speed"][resolution]["<1m/s"]["stats"]["P95"]
            except:
                print("Stats")
                print(dict_all_stations["length_constant_speed"][resolution]["<1m/s"]["stats"])
                print("Values")
                print(dict_all_stations["length_constant_speed"][resolution]["<1m/s"]["values"])
                print("resolution")
                print(resolution)
                print('All')
                print(dict_all_stations["length_constant_speed"][resolution]["<1m/s"])
                raise
            P95_high = dict_all_stations["length_constant_speed"][resolution][">=1m/s"]["stats"]["P95"]
            P75_low = dict_all_stations["length_constant_speed"][resolution]["<1m/s"]["stats"]["P75"]
            P75_high = dict_all_stations["length_constant_speed"][resolution][">=1m/s"]["stats"]["P75"]
            P25_low = dict_all_stations["length_constant_speed"][resolution]["<1m/s"]["stats"]["P25"]
            P25_high = dict_all_stations["length_constant_speed"][resolution][">=1m/s"]["stats"]["P25"]

            # Low wind speeds
            if len(values_1) >= 10:

                # Criteria
                criteria = P95_low + amplification_factor_speed * 7.5 * (P75_low - P25_low)

                # Criteria = max(3, criteria)
                criteria = np.max((3, criteria))

                # Criteria = min(12, criteria)
                criteria = np.min((12, criteria))

                dict_all_stations["length_constant_speed"][resolution]["<1m/s"]["stats"]["criteria"] = criteria
            else:
                dict_all_stations["length_constant_speed"][resolution]["<1m/s"]["stats"]["criteria"] = 12

            # High wind speeds
            if len(values_2) >= 10:

                # Criteria
                criteria = P95_high + amplification_factor_speed * 7.5 * (P75_high - P25_high)

                # Criteria = max(4, P95 + amplification_factor * 8 * IQR)
                criteria = np.max((4, criteria))

                # Criteria = max(12, criteria)
                criteria = np.min((12, criteria))

                dict_all_stations["length_constant_speed"][resolution][">=1m/s"]["stats"]["criteria"] = criteria
            else:
                dict_all_stations["length_constant_speed"][resolution][">=1m/s"]["stats"]["criteria"] = 12

        # Criterion direction
        for resolution in [10, 5, 1, 0.1]:
            resolution = str(resolution)

            # Constant direction not null
            values_1 = dict_all_stations["length_constant_direction"][resolution]["!=0"]["values"]
            P95 = dict_all_stations["length_constant_direction"][resolution]["!=0"]["stats"]["P95"]
            P75 = dict_all_stations["length_constant_direction"][resolution]["!=0"]["stats"]["P75"]
            P25 = dict_all_stations["length_constant_direction"][resolution]["!=0"]["stats"]["P25"]

            if len(values_1) >= 10:

                # Criteria
                criteria = P95 + amplification_factor_direction * 15 * (P75 - P25)

                # Criteria = max(4, criteria)
                criteria = np.max((4, criteria))

                # Criteria = max(12, criteria)
                criteria = np.min((12, criteria))

                dict_all_stations["length_constant_direction"][resolution]["!=0"]["stats"]["criteria"] = criteria
            else:
                dict_all_stations["length_constant_direction"][resolution]["!=0"]["stats"]["criteria"] = 12

        if verbose:
            print("__Criterion speed and direction calculated")
            print(f"__Amplification factor speed: {amplification_factor_speed}")
            print(f"__Amplification factor direction: {amplification_factor_direction}")
            print(f"__Minimum length of suspect constant sequence: {criteria_min}")
            print(f"__Maximum length of suspect constant sequence: {criteria_max}")

        return dict_all_stations

    @print_func_executed_decorator("apply_stats_cst_seq")
    @timer_decorator("apply_stats_cst_seq", unit="minute")
    def qc_apply_stats_cst_seq(self, dict_constant_sequence, dict_all_stations, wind_speed='vw10m(m/s)',
                               wind_direction='winddir(deg)'):
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

            # Preferred direction
            pref_direction = time_series_station['cardinal'].value_counts().nlargest(n=1).index

            time_series_station['preferred_direction'] = pref_direction[0] if np.size(pref_direction) != 0 else np.nan

            # Speed
            for resolution in range(5):
                resolution = str(resolution)
                for speed in ['<1m/s', '>=1m/s']:

                    # Get criteria
                    criteria = dict_all_stations['length_constant_speed'][resolution][speed]['stats']['criteria']

                    for index, length in enumerate(
                            dict_constant_sequence[station]['length_constant_speed'][resolution][speed]):

                        # Apply criteria
                        assert criteria is not None

                        # Select constant sequence
                        begin = dict_constant_sequence[station]['date_constant_speed_begin'][resolution][speed][
                            index]
                        end = dict_constant_sequence[station]['date_constant_speed_end'][resolution][speed][index]

                        if length >= criteria:
                            # Flag series
                            time_series_station['validity_speed'][begin:end] = 0
                            time_series_station['last_flagged_speed'][
                            begin:end] = 'cst_sequence_criteria_not_passed_speed'
                            time_series_station['qc_3_speed'][begin:end] = 'cst_sequence_criteria_not_passed_speed'

            # Direction
            for resolution in [10, 5, 1, 0.1]:
                resolution = str(resolution)

                # Get criteria
                criteria = dict_all_stations['length_constant_direction'][resolution]['!=0']['stats']['criteria']
                lengths_sequences = dict_constant_sequence[station]['length_constant_direction'][resolution]['!=0']
                for index, length in enumerate(lengths_sequences):

                    # Apply criteria
                    assert criteria is not None

                    # Select constant sequence
                    begin = dict_constant_sequence[station]['date_constant_direction_begin'][resolution]['!=0'][index]
                    end = dict_constant_sequence[station]['date_constant_direction_end'][resolution]['!=0'][index]

                    if length >= criteria:

                        # Flag series
                        time_series_station['validity_direction'][begin:end] = 0
                        time_series_station['last_flagged_direction'][
                        begin:end] = 'cst_sequence_criteria_not_passed_direction'
                        time_series_station['qc_3_direction'][begin:end] = 'cst_sequence_criteria_not_passed_direction'

                        # Unflag if constant direction is preferred direction
                        count_cardinals = time_series_station['cardinal'][begin:end].value_counts()
                        direction_cst_seq = count_cardinals.nlargest(n=1).index[0]
                        time_series_station['preferred_direction_during_sequence'][begin:end] = direction_cst_seq

                        if direction_cst_seq == pref_direction:
                            time_series_station['validity_direction'][begin:end] = 1
                            time_series_station['last_unflagged_direction'][begin:end] = 'pref_direction'
                            time_series_station['qc_3_direction_pref'][begin:end] = 1
                    else:
                        pass

            # Add station to list of dataframe
            list_dataframe.append(time_series_station)

        self.time_series = pd.concat(list_dataframe)

    @print_func_executed_decorator("get_nearest_neigbhors")
    @timer_decorator("get_nearest_neigbhors", unit="minute")
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

    @print_func_executed_decorator("ra")
    @timer_decorator("ra", unit="minute")
    def qc_ra(self, dict_constant_sequence, dict_all_stations, wind_speed='vw10m(m/s)', wind_direction='winddir(deg)'):
        """
        Quality control

        This function performs a regional analysis to flag/unflag constant sequences with respect to neighbors.

        Specific to wind speed.
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []

        for station in list_stations:

            # Select station
            filter = time_series["name"] == station
            time_series_station = time_series[filter]

            # Regional analysis initialisation
            time_series_station["qc_4"] = 1
            time_series_station["ra"] = np.nan
            time_series_station["RA"] = np.nan
            time_series_station["qc_neighbors_selected"] = np.nan

            # Select neighbors
            list_neighbors = self.stations['neighbors'][self.stations["name"] == station].values
            print(station)
            print(list_neighbors)

            if np.size(list_neighbors) != 0:
                # Select calm period for each resolution
                for resolution in range(5):

                    # Load constant sequences
                    data = dict_constant_sequence[station]['length_constant_speed'][str(resolution)]['<1m/s']

                    if np.size(data) != 0:
                        for index in range(len(data)):

                            # Length of constant time serie
                            length = dict_constant_sequence[station]['length_constant_speed'][str(resolution)]['<1m/s'][
                                index]

                            # Apply criteria
                            criteria = dict_all_stations['length_constant_speed'][str(resolution)]['<1m/s']['stats'][
                                'criteria']
                            if criteria is not None and (length >= criteria):

                                # Time of constant sequence
                                begin = \
                                    dict_constant_sequence[station]['date_constant_speed_begin'][str(resolution)][
                                        '<1m/s'][
                                        index]
                                end = \
                                    dict_constant_sequence[station]['date_constant_speed_end'][str(resolution)][
                                        '<1m/s'][
                                        index]

                                # Ten days time serie
                                begin_ten_days = begin - np.timedelta64(10, 'D')
                                end_ten_days = end + np.timedelta64(10, 'D')
                                try:
                                    ten_days_1 = time_series_station[wind_speed][begin_ten_days: begin]
                                    ten_days_2 = time_series_station[wind_speed][end: end_ten_days]
                                except IndexError:
                                    pass
                                ten_days_time_serie = pd.concat((ten_days_1, ten_days_2))

                                # One day time serie
                                begin_one_day = begin - np.timedelta64(1, 'D')
                                end_one_day = end + np.timedelta64(1, 'D')

                                # Construct regional average
                                nb_neighbor_selected = 0
                                selected_neighbors = []
                                ra = []
                                RA = None
                                if np.size(list_neighbors) != 0:
                                    for neighbor in list_neighbors[0]:

                                        neighbor_time_serie = time_series[time_series["name"] == neighbor]
                                        try:
                                            neighbor_ten_days_before = neighbor_time_serie[wind_speed][
                                                                       begin_ten_days: begin]
                                            neighbor_ten_days_after = neighbor_time_serie[wind_speed][end: end_ten_days]
                                            neighbor_ten_days_before_after = pd.concat(
                                                (neighbor_ten_days_before, neighbor_ten_days_after))

                                            # Correlation between ten day time series
                                            data = np.array((ten_days_time_serie, neighbor_ten_days_before_after)).T
                                            corr_coeff = pd.DataFrame(data).corr().iloc[0, 1]
                                        except IndexError:
                                            corr_coeff = 0
                                            pass

                                        # If stations are correlated
                                        if corr_coeff >= 0.4:

                                            # Neighbor candidate accepted
                                            nb_neighbor_selected += 1
                                            selected_neighbors.append(neighbor)

                                            # Normalize neighbors observations
                                            try:
                                                neighbor_one_day = neighbor_time_serie[wind_speed][
                                                                   begin_one_day: end_one_day]
                                            except IndexError:
                                                pass
                                            ra.append((neighbor_one_day - np.nanmean(neighbor_one_day)) / np.nanstd(
                                                neighbor_one_day))

                                    # If we found neighbors
                                    if nb_neighbor_selected > 0:
                                        # We compute the mean of ra of the neighbors
                                        ra = pd.concat(ra).groupby(pd.concat(ra).index).mean() if len(ra) > 1 else ra[0]

                                        # RA
                                        vmin = np.nanmin(ra[begin:end])
                                        vmax_begin = np.nanmax(ra[begin_ten_days: begin])
                                        vmax_end = np.nanmax(ra[end: end_ten_days])
                                        vmax = np.nanmax((vmax_begin, vmax_end))

                                        RA = (ra - vmin) / (vmax - vmin)
                                        RA = RA[begin:end]

                                        # Store variables
                                        time_series_station["ra"][time_series_station.index.isin(ra)] = ra
                                        time_series_station["RA"][time_series_station.index.isin(RA)] = RA
                                        time_series_station["qc_neighbors_selected"][
                                            time_series_station.index.isin(RA)] = [
                                            selected_neighbors for k in range(len(RA))]

                                    if RA is not None:
                                        time_series_station["qc_4"][
                                            time_series_station.index.isin(RA[RA > 0.33].index)] = "qc_ra_suspicious"
                                        time_series_station["validity_speed"][
                                            time_series_station.index.isin(RA[RA > 0.33].index)] = 0
                                        time_series_station["last_flagged_speed"][
                                            time_series_station.index.isin(RA[RA > 0.33].index)] = "qc_ra_suspicious"
                                        time_series_station["qc_4"][
                                            time_series_station.index.isin(RA[RA <= 0.33].index)] = "qc_ra_ok"
                                        time_series_station["last_unflagged_speed"][
                                            time_series_station.index.isin(RA[RA <= 0.33].index)] = "qc_ra_ok"

                # Add station to list of dataframe
                list_dataframe.append(time_series_station)

        self.time_series = pd.concat(list_dataframe)

    @print_func_executed_decorator("high_variability")
    @timer_decorator("high_variability", unit="minute")
    def qc_high_variability(self, wind_speed='vw10m(m/s)'):
        """
        Quality control

        This function detects suspicious high variability in data.

        Specific to wind speed.
        """
        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []

        # Get statistics
        dict_high_variability = {}
        for station in list_stations:
            dict_high_variability[station] = {}

            # Select station
            filter = time_series["name"] == station
            time_series_station = time_series[filter]

            # Delta time
            time = time_series_station.index.to_series()
            time_difference = (time - time.shift()) / np.timedelta64(1, 'h')

            # Increment for several timesteps
            for time_step in range(1, 24):
                time_step = str(time_step)

                # Initialize values
                dict_high_variability[station][time_step] = {}
                dict_high_variability[station][time_step]["values"] = {}
                dict_high_variability[station][time_step]["values"][">=0"] = []
                dict_high_variability[station][time_step]["values"]["<0"] = []

                # Wind_(i+n) - Wind_(i)
                increments = time_series_station[wind_speed].diff(periods=time_step)

                # Postive = increase, negative = decrease
                increments_positive = np.abs(increments[increments >= 0])
                increments_negative = np.abs(increments[increments < 0])

                dict_high_variability[station][time_step]["values"][">=0"].extend(increments_positive.values)
                dict_high_variability[station][time_step]["values"]["<0"].extend(increments_negative.values)

                # Statistics positives
                P95_p = np.nanquantile(increments_positive.values, 0.95)
                P75_p = np.nanquantile(increments_positive.values, 0.75)
                P25_p = np.nanquantile(increments_positive.values, 0.25)
                dict_high_variability[station][time_step]["P95_p"] = P95_p
                dict_high_variability[station][time_step]["P75_p"] = P75_p
                dict_high_variability[station][time_step]["P25_p"] = P25_p

                # Statistics negatives
                P95_n = np.nanquantile(increments_negative.values, 0.95)
                P75_n = np.nanquantile(increments_negative.values, 0.75)
                P25_n = np.nanquantile(increments_negative.values, 0.25)
                dict_high_variability[station][time_step]["P95_p"] = P95_n
                dict_high_variability[station][time_step]["P75_p"] = P75_n
                dict_high_variability[station][time_step]["P25_p"] = P25_n

                criteria_p = P95_p + 8.9 * (P75_p - P25_p)
                criteria_n = P95_n + 8.9 * (P75_n - P25_n)

                if criteria_p < 7.5:
                    criteria_p = 7.5

                if criteria_n < 7.5:
                    criteria_n = 7.5

                # Time resolution
                delta_t = time_difference == time_step
                too_high_positive = (increments >= 0) & (increments >= criteria_p) & delta_t
                too_high_negative = (increments < 0) & (np.abs(increments) >= criteria_n) & delta_t

                time_series_station["qc_5_speed"][too_high_positive] = "too high"
                time_series_station["qc_5_speed"][too_high_negative] = "too high"
                time_series_station["validity_speed"][too_high_positive] = 0
                time_series_station["validity_speed"][too_high_negative] = 0
                time_series_station["last_flagged_speed"][too_high_positive] = "high variation"
                time_series_station["last_flagged_speed"][too_high_negative] = "high variation"
                time_series_station["qc_high_variability_criteria"][too_high_positive] = criteria_p
                time_series_station["qc_high_variability_criteria"][too_high_negative] = criteria_n

            # Add station to list of dataframe
            list_dataframe.append(time_series_station)

        self.time_series = pd.concat(list_dataframe)

    @print_func_executed_decorator("bias")
    @timer_decorator("bias", unit="minute")
    def qc_bias(self, stations='all', wind_speed='vw10m(m/s)', wind_direction='winddir(deg)',
                correct_factor_mean=1.5, correct_factor_std=1, correct_factor_coeff_var=4, update_file=True):
        pd.options.mode.chained_assignment = None  # default='warn'

        time_series = self.time_series
        if stations == 'all':
            stations = time_series["name"].unique()

        list_dataframe = []
        for station in stations:
            time_serie_station = time_series[time_series["name"] == station]

            # Select wind speed
            wind = time_serie_station[wind_speed]

            # Daily wind
            wind = wind.resample('1D').mean()
            result = wind.copy(deep=True) * 0 + 1

            # No outliers
            # We use rolling means to detect outliers
            rol_mean = wind.rolling('15D').mean()
            rol_std = wind.rolling('15D').std()
            no_outliers = wind.copy(deep=True)

            # Very high values
            filter_1 = (no_outliers > rol_mean + 2 * rol_std)

            # Very low values
            filter_2 = (no_outliers < rol_mean - 2 * rol_std)

            no_outliers[filter_1 | filter_2] = np.nan

            # Seasonal mean based on each day
            seasonal = no_outliers.groupby([no_outliers.index.month, no_outliers.index.day]).mean()

            # Arbitrary index
            try:
                seasonal.index = pd.date_range(start='1904-01-01', freq='D', periods=366)
            except ValueError:
                print(f"__qc_bias: Not enough data at {station} to perform bias analysis")
                continue

            # Rolling mean
            seasonal_rolling = seasonal.rolling('15D').mean()

            # Interpolate missing values
            seasonal_rolling = seasonal_rolling.interpolate()

            # Divide two datasets by seasonal
            for month in range(1, 13):
                for day in range(1, 32):

                    # Filters
                    filter_wind = (wind.index.month == month) & (wind.index.day == day)
                    filter_no_outlier = (no_outliers.index.month == month) & (no_outliers.index.day == day)
                    filter_seasonal = (seasonal_rolling.index.month == month) & (seasonal_rolling.index.day == day)

                    # Normalize daily values by seasonal means
                    try:
                        wind[filter_wind] = wind[filter_wind] / seasonal_rolling[filter_seasonal].values[0]
                    except IndexError:
                        wind[filter_wind] = wind / 1
                    try:
                        no_outliers[filter_wind] = no_outliers[filter_no_outlier] / \
                                                   seasonal_rolling[filter_seasonal].values[0]
                    except IndexError:
                        no_outliers[filter_wind] = no_outliers / 1

            # Rolling
            wind_rolling = wind.rolling('15D').mean()
            no_outliers_rolling = no_outliers.rolling('15D').mean()

            # Wind speed
            P95 = no_outliers.rolling('15D').quantile(0.95)
            P25 = no_outliers.rolling('15D').quantile(0.25)
            P75 = no_outliers.rolling('15D').quantile(0.75)
            criteria_high = (wind_rolling > (P95 + 3.7 * (P75 - P25)))
            criteria_low = (wind_rolling < 0.5 / correct_factor_mean)
            criteria_mean = (criteria_high | criteria_low)

            # Standard deviation
            standard_deviation = np.abs(wind - wind.mean())
            standard_deviation_rolling = standard_deviation.rolling('15D').mean()
            standard_deviation_no_outliers = np.abs(no_outliers - no_outliers.mean())
            P95 = standard_deviation_no_outliers.rolling('15D').quantile(0.95)
            P25 = standard_deviation_no_outliers.rolling('15D').quantile(0.25)
            P75 = standard_deviation_no_outliers.rolling('15D').quantile(0.75)
            criteria_high = (standard_deviation_rolling > (P95 + 7.5 * (P75 - P25)))
            criteria_low = (standard_deviation_rolling < (0.044 / correct_factor_std))
            criteria_std = (criteria_high | criteria_low)

            # Coefficient of variation
            coeff_variation = standard_deviation / wind_rolling.mean()
            coeff_variation_rolling = coeff_variation.rolling('15D').mean()
            coeff_variation_no_outliers = standard_deviation_no_outliers / no_outliers.mean()
            P95 = coeff_variation_no_outliers.rolling('15D').quantile(0.95)
            P25 = coeff_variation_no_outliers.rolling('15D').quantile(0.25)
            P75 = coeff_variation_no_outliers.rolling('15D').quantile(0.75)
            criteria_high = (coeff_variation_rolling > (P95 + 7.5 * (P75 - P25)))
            criteria_low = (coeff_variation_rolling < 0.22 / correct_factor_coeff_var)
            criteria_coeff_var = (criteria_high | criteria_low)

            # Criteria number of nans during rolling mean
            condition_nb_nans = wind.rolling('15D').count() < 7

            # Result
            result[criteria_mean | criteria_std | criteria_coeff_var] = 0
            result[condition_nb_nans] = 1
            result = result.resample('1H').pad()

            if self._qc_init:
                time_serie_station["validity_speed"] = result
                time_serie_station['qc_6_speed'] = result
                time_serie_station["last_flagged_speed"][time_serie_station["qc_6_speed"] == 0] = "bias speed"

            # Add station to list of dataframe
            list_dataframe.append(time_serie_station)
        if update_file:
            self.time_series = pd.concat(list_dataframe)
        else:
            return time_serie_station

    def _qc_isolated_records(self, time_series_station, variable, max_time=24, min_time=12, type="speed", verbose=True):

        # to detect isolated records after the qc process, we need to apply the result of the qc to the time series
        wind = time_series_station.copy()
        filter_last_flagged = wind["last_flagged_speed"] != 0
        wind[variable][filter_last_flagged] = np.nan
        wind = pd.DataFrame(wind[variable].values, columns=[variable])
        wind.index = time_series_station.index

        is_na = wind[variable].isna()
        wind["group"] = is_na.diff().ne(0).cumsum()
        wind["is_nan"] = is_na * wind["group"]

        groups = [group for _, group in wind.groupby("group")]

        groups_to_discard = []
        for index in range(1, len(groups) - 1):

            previous_nan = (groups[index - 1].is_nan.mean() != 0)
            next_nan = (groups[index + 1].is_nan.mean() != 0)
            current_not_nan = (groups[index].is_nan.mean() == 0)

            previous_len = len(groups[index - 1]) >= 12
            next_len = len(groups[index + 1]) >= 12
            current_len = len(groups[index]) <= 24

            if previous_nan & next_nan & previous_len & next_len & current_not_nan & current_len:
                groups_to_discard.append(index + 1)

        filter = wind["group"].isin(groups_to_discard)
        if type == "speed":
            time_series_station["qc_7_isolated_records_speed"][filter] = 0
            time_series_station["validity_speed"][filter] = 0
            time_series_station["last_flagged_speed"][filter] = "Isolated records"

        if type == "direction":
            time_series_station["qc_7_isolated_records_direction"][filter] = 0
            time_series_station["validity_direction"][filter] = 0
            time_series_station["last_flagged_direction"][filter] = "Isolated records"

        if verbose:
            if type == "speed": print("__Isolated records speed calculated")
            if type == "direction": print("__Isolated records direction calculated")
            print(f"__Isolated record max  duration: {max_time} hours")
            print(f"__Nan periods before isolated records: min {min_time} hours")

        return time_series_station

    @print_func_executed_decorator("isolated records")
    @timer_decorator("isolated_records", unit="minute")
    def qc_isolated_records(self, wind_speed='vw10m(m/s)', wind_direction='winddir(deg)', verbose=True):

        time_series = self.time_series
        list_stations = time_series["name"].unique()
        list_dataframe = []

        # Get statistics
        for station in list_stations:
            # Select station
            filter = time_series["name"] == station
            time_series_station = time_series[filter]

            time_series_station = self._qc_isolated_records(time_series_station, wind_speed, type="speed",
                                                            verbose=verbose)
            time_series_station = self._qc_isolated_records(time_series_station, wind_direction, type="direction",
                                                            verbose=verbose)
            list_dataframe.append(time_series_station)

        self.time_series = pd.concat(list_dataframe)

    @timer_decorator("qc", unit="minute")
    def qc(self, compare_calm_long_sequences_to_neighbors=False):
        """
        52 minutes on LabIA cluster
        """

        pd.options.mode.chained_assignment = None  # default='warn'

        self.qc_initialization()

        self.qc_check_duplicates_in_index()

        self.qc_resample_index()

        self.qc_check_duplicates_in_index()

        self.qc_calm_criteria()

        self.qc_true_north()

        self.qc_bias()

        self.qc_removal_unphysical_values()

        self.qc_get_wind_speed_resolution()

        self.qc_get_wind_direction_resolution()

        dict_constant_sequence = self.qc_constant_sequences()

        self.qc_excessive_MISS(dict_constant_sequence)

        dict_all_stations = self.qc_get_stats_cst_seq(dict_constant_sequence,
                                                      amplification_factor_speed=1.5,
                                                      amplification_factor_direction=1.5)

        self.qc_apply_stats_cst_seq(dict_constant_sequence, dict_all_stations)

        if compare_calm_long_sequences_to_neighbors:
            self.qc_get_nearest_neigbhors()

            self.qc_ra(dict_constant_sequence, dict_all_stations)

        self.qc_high_variability()

        self.qc_isolated_records()

        self._qc = True

        return dict_constant_sequence, dict_all_stations

    # Multiprocessing
    def qc_bias_station(self, time_series=None, station=None, wind_speed=None,
                        correct_factor_mean=None, correct_factor_std=None, correct_factor_coeff_var=None):
        pd.options.mode.chained_assignment = None
        time_serie_station = time_series[time_series["name"] == station]

        # Select wind speed
        wind = time_serie_station[wind_speed]

        # Daily wind
        wind = wind.resample('1D').mean()
        result = wind.copy(deep=True) * 0

        # No outliers
        # We use rolling means to detect outliers
        rol_mean = wind.rolling('15D').mean()
        rol_std = wind.rolling('15D').std()
        no_outliers = wind.copy(deep=True)

        # Very high values
        filter_1 = (no_outliers > rol_mean + 2 * rol_std)

        # Very low values
        filter_2 = (no_outliers < rol_mean - 2 * rol_std)

        no_outliers[filter_1 | filter_2] = np.nan

        # Seasonal mean based on each day
        seasonal = no_outliers.groupby([no_outliers.index.month, no_outliers.index.day]).mean()

        # Arbitrary index
        try:
            seasonal.index = pd.date_range(start='1904-01-01', freq='D', periods=366)
        except ValueError:
            print(f"__qc_bias: Not enough data at {station} to perform bias analysis")
            return None

        # Rolling mean
        seasonal_rolling = seasonal.rolling('15D').mean()

        # Interpolate missing values
        seasonal_rolling = seasonal_rolling.interpolate()

        # Divide two datasets by seasonal
        for month in range(1, 13):
            for day in range(1, 32):

                # Filters
                filter_wind = (wind.index.month == month) & (wind.index.day == day)
                filter_no_outlier = (no_outliers.index.month == month) & (no_outliers.index.day == day)
                filter_seasonal = (seasonal_rolling.index.month == month) & (seasonal_rolling.index.day == day)

                # Normalize daily values by seasonal means
                try:
                    wind[filter_wind] = wind[filter_wind] / seasonal_rolling[filter_seasonal].values[0]
                except IndexError:
                    wind[filter_wind] = wind / 1
                try:
                    no_outliers[filter_wind] = no_outliers[filter_no_outlier] / \
                                               seasonal_rolling[filter_seasonal].values[0]
                except IndexError:
                    no_outliers[filter_wind] = no_outliers / 1

        # Rolling
        wind_rolling = wind.rolling('15D').mean()

        # Wind speed
        P95 = no_outliers.rolling('15D').quantile(0.95)
        P25 = no_outliers.rolling('15D').quantile(0.25)
        P75 = no_outliers.rolling('15D').quantile(0.75)

        criteria_high = (wind_rolling > (P95 + 3.7 * (P75 - P25)))
        criteria_low = (wind_rolling < 0.5 / correct_factor_mean)
        criteria_mean = (criteria_high | criteria_low)

        # Standard deviation
        standard_deviation = np.abs(wind - wind.mean())
        standard_deviation_rolling = standard_deviation.rolling('15D').mean()
        standard_deviation_no_outliers = np.abs(no_outliers - no_outliers.mean())
        P95 = standard_deviation_no_outliers.rolling('15D').quantile(0.95)
        P25 = standard_deviation_no_outliers.rolling('15D').quantile(0.25)
        P75 = standard_deviation_no_outliers.rolling('15D').quantile(0.75)
        criteria_high = (standard_deviation_rolling > (P95 + 7.5 * (P75 - P25)))
        criteria_low = (standard_deviation_rolling < (0.044 / correct_factor_std))
        criteria_std = (criteria_high | criteria_low)

        # Coefficient of variation
        coeff_variation = standard_deviation / wind_rolling.mean()
        coeff_variation_rolling = coeff_variation.rolling('15D').mean()
        coeff_variation_no_outliers = standard_deviation_no_outliers / no_outliers.mean()
        P95 = coeff_variation_no_outliers.rolling('15D').quantile(0.95)
        P25 = coeff_variation_no_outliers.rolling('15D').quantile(0.25)
        P75 = coeff_variation_no_outliers.rolling('15D').quantile(0.75)
        criteria_high = (coeff_variation_rolling > (P95 + 7.5 * (P75 - P25)))
        criteria_low = (coeff_variation_rolling < 0.22 / correct_factor_coeff_var)
        criteria_coeff_var = (criteria_high | criteria_low)

        # Result
        result[criteria_mean | criteria_std | criteria_coeff_var] = 1
        result = result.resample('1H').pad()
        time_serie_station['qc_bias_observation_speed'] = result

        if self._qc_init:
            time_serie_station["validity_speed"] = result
            time_serie_station["last_flagged_speed"][result == 1] = "high variation"

        # Add station to list of dataframe
        return time_serie_station

    def _qc_bias_station(self, args):
        return self.qc_bias_station(**args)

    def qc_bias_multiprocessing(self, nb_cpu=2, stations='all', wind_speed='vw10m(m/s)', wind_direction='winddir(deg)',
                                correct_factor_mean=1.5, correct_factor_std=1, correct_factor_coeff_var=4,
                                update_file=True):
        import multiprocessing
        pd.options.mode.chained_assignment = None  # default='warn'

        time_series = self.time_series
        if stations == 'all':
            stations = time_series["name"].unique()

        # list_dataframe = []

        todo = []
        for station in stations:
            todo.append({'time_series': time_series, 'station': station, 'wind_speed': wind_speed,
                         'correct_factor_mean': correct_factor_mean,
                         'correct_factor_std': correct_factor_std,
                         'correct_factor_coeff_var': correct_factor_coeff_var})

        with multiprocessing.Pool(nb_cpu) as p:
            list_dataframe = p.map(self._qc_bias_station, todo)

        if update_file:
            self.time_series = pd.concat(list_dataframe)
