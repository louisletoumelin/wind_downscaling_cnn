import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from scipy.ndimage import rotate
import random  # Warning
from time import time as t
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d  # Warning
import seaborn as sns

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
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _cartopy = True
except:
    _cartopy = False

from MidpointNormalize import MidpointNormalize


class Visualization:
    _shapely_geometry = _shapely_geometry
    _geopandas = _geopandas
    _cartopy = _cartopy

    def __init__(self, p):
        t0 = t()
        self.p = p
        if _cartopy:
            self.l93 = ccrs.epsg(2154)
        t1 = t()
        print(f"\nVizualisation created in {np.round(t1-t0, 2)} seconds\n")

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
        ax.plot(stations[f"{nwp_name}_NN_0"].values[0][0],
                stations[f"{nwp_name}_NN_0"].values[0][1],
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
        x_nwp, y_nwp = stations[f"{nwp_name}_NN_0"][stations["name"] == station_name].values[0]

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

    def plot_predictions_2D(self, array_xr=None, stations_name=['Col du Lac Blanc']):

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

    def _plot_3D_topography(self, XX, YY, ZZ, station_name, alpha=1, figsize=(30, 30), new_figure=True,
                            cmap='gist_earth', linewidth=2):
        if new_figure:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.gcf()
        ax = fig.gca(projection='3d')
        im = ax.plot_surface(XX, YY, ZZ, cmap=cmap, lw=0.5, rstride=1, cstride=1, alpha=alpha, linewidth=linewidth)
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

    def plot_3D_topography_NWP(self, station_name='Col du Lac Blanc', length_x_km=30, length_y_km=30, alpha=1,
                               new_figure=True, cmap='gist_earth', linewidth=2):
        """
        Display the topographic map of the area in 3D

        Input:
                array_xr : file with predictions
                station_name : ex: 'Col du Lac Blanc'
        """
        stations = self.p.observation.stations
        nwp_name = self.p.nwp.name
        idx_nwp_x, idx_nwp_y = stations[f"index_{nwp_name}_NN_0"][stations["name"] == station_name].values[0]

        shift_x = (length_x_km // 1.3) + 1
        shift_y = (length_y_km // 1.3) + 1

        XX = self.p.nwp.data_xr.X_L93.data[int(idx_nwp_x - shift_x):int(idx_nwp_x + shift_x),
             int(idx_nwp_y - shift_y):int(idx_nwp_y + shift_y)]
        YY = self.p.nwp.data_xr.Y_L93.data[int(idx_nwp_x - shift_x):int(idx_nwp_x + shift_x),
             int(idx_nwp_y - shift_y):int(idx_nwp_y + shift_y)]
        ZZ = self.p.nwp.data_xr.ZS.data[int(idx_nwp_x - shift_x):int(idx_nwp_x + shift_x),
             int(idx_nwp_y - shift_y):int(idx_nwp_y + shift_y)]

        self._plot_3D_topography(XX, YY, ZZ, station_name, alpha=alpha, new_figure=new_figure, cmap=cmap,
                                 linewidth=linewidth)

    def plot_3D_topography_large_domain(self, station_name='Col du Lac Blanc', nb_pixel_x=400, nb_pixel_y=400,
                                        new_figure=True, cmap="gist_earth", alpha=1):
        """
        Display the topographic map of a large area in 3D
        """

        MNT_data, MNT_x, MNT_y = self.p.observation.extract_MNT_around_station(station_name, self.p.mnt, nb_pixel_x,
                                                                               nb_pixel_y)
        XX, YY = np.meshgrid(MNT_x, MNT_y)
        self._plot_3D_topography(XX, YY, MNT_data, station_name, new_figure=new_figure, cmap=cmap, alpha=alpha)

    def plot_comparison_topography_MNT_NWP(self, station_name='Col du Lac Blanc', length_x_km=10, length_y_km=10,
                                           alpha_nwp=1, alpha_mnt=1, linewidth=2,
                                           new_figure=True, cmap_nwp='cividis', cmap_mnt='gist_earth'):
        # Converting km to number of pixel for mnt
        nb_pixel_x = int(length_x_km * 1000 // self.p.mnt.resolution_x)
        nb_pixel_y = int(length_y_km * 1000 // self.p.mnt.resolution_y)

        self.plot_3D_topography_large_domain(station_name=station_name, nb_pixel_x=nb_pixel_x, nb_pixel_y=nb_pixel_y,
                                             new_figure=new_figure, cmap=cmap_mnt, alpha=alpha_mnt)

        self.plot_3D_topography_NWP(station_name=station_name, length_x_km=length_x_km, length_y_km=length_y_km,
                                    alpha=alpha_nwp, new_figure=new_figure, cmap=cmap_nwp, linewidth=linewidth)

    def _plot_3D_variable(self, XX, YY, ZZ, variable, display, time, midpoint, station_name, alpha=1,
                          rstride=1, cstride=1, new_figure=True):

        if new_figure:
            fig = plt.figure()
        else:
            fig = plt.gcf()
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

    def plot_predictions_3D(self, array_xr=None, stations_name=['Col du Lac Blanc'], arrow_length_ratio=0.3, length_arrow_modification=1, arrow=True, alpha=1):
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
        self._plot_3D_variable(XX, YY, ZZ, variable, UVW, time, nwp_speed, random_station_name, alpha=alpha)
        if arrow:
            self._plot_arrows_for_CNN_3D(XX, YY, ZZ, U, V, W, None, nwp_speed, arrow_length_ratio=arrow_length_ratio, length_arrow_modification=length_arrow_modification)

    def _plot_arrows_for_CNN_3D(self, XX, YY, ZZ, U, V, W, variable_color, midpoint, cmap="coolwarm",
                                figsize=(30, 30), colors='black', linewidth=1, length=10,
                                arrow_length_ratio=0.3, length_arrow_modification=1):
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
                      length=length*length_arrow_modification,
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
        x_nwp, y_nwp = stations[f"{nwp_name}_NN_0"][stations["name"] == station_name].values[0]
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

    def qc_plot_validity(self, wind_speed='vw10m(m/s)', wind_direction='winddir(deg)'):

        # Select time series
        time_series = self.p.observation.time_series


        for station in time_series["name"].unique():

            # Select station
            time_serie_station = time_series[time_series["name"] == station]

            # Wind speed
            plt.figure()
            try:
                plt.subplot(211)
                time_serie_station[wind_speed].plot(marker='x')
                time_serie_station[wind_speed][time_serie_station['validity_speed'] == 0].plot(marker='x', linestyle='')
            except:
                pass

            # Wind direction
            try:
                plt.subplot(212)
                time_serie_station[wind_direction].plot(marker='x')
                time_serie_station[wind_direction][time_serie_station['validity_direction'] == 0].plot(marker='x', linestyle='')
            except:
                pass

            plt.title(station)

    def qc_plot_resolution(self, wind_speed='vw10m(m/s)', wind_direction='winddir(deg)'):

        # Select time series
        time_series = self.p.observation.time_series

        for station in time_series["name"].unique():

            # Select station
            time_serie_station = time_series[time_series["name"] == station]

            plt.figure()
            try:
                plt.subplot(211)

                cmap = matplotlib.cm.viridis
                bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
                norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

                x = time_serie_station[wind_speed].index
                y = time_serie_station[wind_speed].values
                c = time_serie_station['resolution_speed'].values
                im = plt.scatter(x, y, c=c, marker='x', norm=norm, cmap=cmap)
                plt.colorbar(im)
            except:
                pass

            try:
                plt.subplot(212)

                cmap = matplotlib.cm.magma
                bounds = [0, 0.2, 1.5, 7, 12]
                norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
                x = time_serie_station[wind_direction].index
                y = time_serie_station[wind_direction].values
                c = time_serie_station['resolution_direction'].values
                im = plt.scatter(x, y, c=c, marker='x', norm=norm, cmap=cmap)
                plt.colorbar(im)
            except:
                pass

            plt.title(station)
            plt.tight_layout()

    def qc_plot_constant(self, wind_speed='vw10m(m/s)', wind_direction='winddir(deg)'):

        # Select time series
        time_series = self.p.observation.time_series

        for station in time_series["name"].unique():

            # Select station
            time_serie_station = time_series[time_series["name"] == station]

            plt.figure()
            try:
                plt.subplot(211)

                cmap = matplotlib.cm.viridis
                bounds = [-0.5, 0.5, 1.5]
                norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

                x = time_serie_station[wind_speed].index
                y = time_serie_station[wind_speed].values
                c = time_serie_station['constant_speed'].values
                im = plt.scatter(x, y, c=c, marker='x', norm=norm, cmap=cmap)
                plt.colorbar(im)
            except:
                pass

            try:
                plt.subplot(212)

                cmap = matplotlib.cm.magma
                bounds = [-0.5, 0.5, 1.5]
                norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
                x = time_serie_station[wind_direction].index
                y = time_serie_station[wind_direction].values
                c = time_serie_station['constant_direction'].values
                im = plt.scatter(x, y, c=c, marker='x', norm=norm, cmap=cmap)
                plt.colorbar(im)
            except:
                pass

            plt.title(station)
            plt.tight_layout()

    def qc_plot_last_flagged(self, stations='all', wind_speed='vw10m(m/s)', wind_direction='winddir(deg)'):

        # Select time series
        time_series = self.p.observation.time_series

        if stations == 'all':
            stations = time_series["name"].unique()


        for station in stations:

            # Select station
            time_serie_station = time_series[time_series["name"] == station]
            unflagged_data_speed = time_serie_station[time_serie_station['last_unflagged_speed'] != 0]
            unflagged_data_direction = time_serie_station[time_serie_station['last_unflagged_direction'] != 0]

            plt.figure()

            plt.subplot(211)
            sns.scatterplot(x=time_serie_station.index, y=wind_speed, data=time_serie_station, hue='last_flagged_speed')
            sns.scatterplot(x=unflagged_data_speed.index, y=wind_speed, data=unflagged_data_speed, hue='last_unflagged_speed', s=20, palette="husl")

            plt.subplot(212)
            sns.scatterplot(x=time_serie_station.index, y=wind_direction, data=time_serie_station, hue='last_flagged_direction')
            sns.scatterplot(x=unflagged_data_direction.index, y=wind_direction, data=unflagged_data_direction, hue='last_unflagged_direction', s=20, palette="husl")

            plt.title(station)
            plt.tight_layout()
