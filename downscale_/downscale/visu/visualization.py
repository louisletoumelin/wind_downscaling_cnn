import uuid

import numpy as np
import pandas as pd
import random  # Warning
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LightSource

try:
    import seaborn as sns
    sns.set(font_scale=2)
    sns.set_style("white")
    _seaborn = True
except ModuleNotFoundError:
    _seaborn = False

try:
    from shapely.geometry import Point
    from shapely.geometry import Polygon
    _shapely_geometry = True
except ModuleNotFoundError:
    _shapely_geometry = False

try:
    import geopandas as gpd

    _geopandas = True
except ModuleNotFoundError:
    _geopandas = False

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    _cartopy = True
except ModuleNotFoundError:
    _cartopy = False

from downscale.visu.MidpointNormalize import MidpointNormalize
from downscale.operators.devine import Devine
from downscale.utils.context_managers import print_all_context
from downscale.utils.utils_func import save_figure


class Visualization:

    def __init__(self, p=None, prm={"verbose": True}):

        with print_all_context("Vizualization", level=0, unit="second", verbose=prm.get("verbose")):
            self.p = p
            if _cartopy:
                self.l93 = ccrs.epsg(2154)


    @staticmethod
    def density_scatter(x, y, ax=None, sort=True, use_power_norm=1, bins=20, colorbar=True, **kwargs):
        """
        https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib/53865762#53865762
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import PowerNorm
        from scipy.interpolate import interpn
        if ax is None:
            fig, ax = plt.subplots()
        data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
        z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T,
                    method="splinef2d", bounds_error=False)
        # To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0
        # Sort the points by density, so that the densest points are plotted last
        if sort:
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
        if use_power_norm == 1:
            im = ax.scatter(x, y, c=z, norm=PowerNorm(gamma=0.25, vmin=0), **kwargs)
        elif use_power_norm == 2:
            im = ax.scatter(x, y, c=z, norm=PowerNorm(gamma=0.25, vmin=0), **kwargs)
        else:
            im = ax.scatter(x, y, c=z, **kwargs)
        if colorbar:
            plt.colorbar(im)
        # norm = Normalize(vmin=np.min(z), vmax=np.max(z))
        # cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
        # cbar.ax.set_ylabel('Density')
        return ax

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

    def plot_model(self):
        # Load model
        self.p.load_cnn(dependencies=True)

        import visualkeras
        from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D, \
            Cropping2D, InputLayer, Concatenate
        from collections import defaultdict
        import matplotlib
        import matplotlib.pylab as pl
        from PIL import ImageFont
        color_map = defaultdict(dict)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        colors = pl.cm.gist_earth(norm(np.linspace(0, 1, 20)), bytes=True)

        color_map[Conv2D]['fill'] = tuple(colors[19])
        color_map[ZeroPadding2D]['fill'] = tuple(colors[15])
        color_map[Concatenate]['fill'] = tuple(colors[10])
        color_map[MaxPooling2D]['fill'] = tuple(colors[8])
        color_map[Dense]['fill'] = tuple(colors[6])
        color_map[Flatten]['fill'] = tuple(colors[4])
        color_map[Cropping2D]['fill'] = tuple(colors[2])
        color_map[InputLayer]['fill'] = tuple(colors[0])

        print(color_map)

        font = ImageFont.truetype("arial.ttf", 35)  # using comic sans is strictly prohibited!
        visualkeras.layered_view(self.p.model, color_map=color_map, legend=True, draw_volume=True, draw_funnel=True,
                                 shade_step=0, font=font, scale_xy=2, scale_z=0.5, to_file='output86.png')
        # tf.keras.utils.plot_model(self.model, to_file='Model1.png')
        # tf.keras.utils.plot_model(self.model, to_file='Model2.png', show_shapes=True)

    def plot_area(self, set_lim=True):
        """Updated 16/11/2021"""
        fig = plt.figure()

        ax = fig.add_subplot(111, projection=self.l93)

        ax.coastlines(resolution='10m', alpha=0.1)
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        if set_lim:
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
        """Updated 16/11/2021"""
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
        rotated_topo_large = self.rotate_topography(MNT_data, wind_direction)
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

    def _plot_arrow_for_NWP(self, array_xr, time_index, station_name='Col du Lac Blanc', vmin=1, vmax=5, scale_cmap=2,
                            scale_nwp=1/0.04):
        # Arrow for NWP wind
        ax = plt.gca()
        stations = self.p.observation.stations
        nwp_name = self.p.nwp.name
        x_nwp, y_nwp = stations[f"{nwp_name}_NN_0"][stations["name"] == station_name].values[0]
        y_nwp = y_nwp

        if isinstance(time_index, int) or isinstance(time_index, float):
            NWP_wind_speed = array_xr.NWP_wind_speed.sel(station=station_name).isel(time=time_index).data
            NWP_wind_DIR = array_xr.NWP_wind_DIR.sel(station=station_name).isel(time=time_index).data
        elif isinstance(time_index, np.datetime64):
            NWP_wind_speed = array_xr.NWP_wind_speed.sel(station=station_name).sel(time=time_index).data
            NWP_wind_DIR = array_xr.NWP_wind_DIR.sel(station=station_name).sel(time=time_index).data

        U_nwp = -np.sin((np.pi / 180) * NWP_wind_DIR) * NWP_wind_speed
        V_nwp = -np.cos((np.pi / 180) * NWP_wind_DIR) * NWP_wind_speed

        divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=NWP_wind_speed, vmax=vmax)
        print(station_name)
        print(NWP_wind_speed)
        ax.quiver(x_nwp, y_nwp, U_nwp, V_nwp, NWP_wind_speed, cmap="coolwarm", norm=divnorm,
                  edgecolor="C1", linewidths=1, scale=scale_nwp)
        #ax.text(x_nwp - 100, y_nwp + 100, str(np.round(NWP_wind_speed, 1)) + " m/s", color='red')
        #ax.text(x_nwp + 100, y_nwp - 100, str(np.round(NWP_wind_DIR)) + '°', color='red')

    def _plot_arrow_for_observation_station(self, array_xr, time_index, station_name='Col du Lac Blanc', vmin=1, vmax=5, nwp_speed=None, scale_cmap=2):
        # Arrow for station wind
        ax = plt.gca()
        stations = self.p.observation.stations
        time_series = self.p.observation.time_series
        x_observation = stations["X"][stations["name"] == station_name].values[0]
        y_observation = stations["Y"][stations["name"] == station_name].values[0]
        if isinstance(time_index, float) or isinstance(time_index, int):
            dates = pd.DatetimeIndex([array_xr.time.isel(time=time_index).values])
        elif isinstance(time_index, np.datetime64):
            dates = pd.DatetimeIndex([array_xr.time.sel(time=time_index).values])
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
        from matplotlib import colors
        divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=nwp_speed, vmax=vmax)
        print("Observation")
        print(station_name)
        print(observed_UV)
        print(observed_UV_DIR)
        ax.quiver(x_observation, y_observation, U_observed, V_observed, observed_UV, cmap="coolwarm",
                  norm=divnorm, edgecolor="black", linewidths=1)

    def _plot_arrows_for_CNN(self, array_xr, time_index, station_name='Col du Lac Blanc', scale=1 / 0.005, n = 1,
                             cmap="coolwarm", size_proportionnal=False, scale_cmap=2, vmin=1, vmax=5, fontsize=20, **kwargs):

        ax = plt.gca()
        if isinstance(time_index, int) or isinstance(time_index, float):
            U = array_xr.U.sel(station=station_name).isel(time=time_index).data
            V = array_xr.V.sel(station=station_name).isel(time=time_index).data
            UV = array_xr.UV.sel(station=station_name).isel(time=time_index).data
            NWP_wind_speed = array_xr.NWP_wind_speed.sel(station=station_name).isel(time=time_index).data
        elif isinstance(time_index, np.datetime64):
            U = array_xr.U.sel(station=station_name).sel(time=time_index).data
            V = array_xr.V.sel(station=station_name).sel(time=time_index).data
            UV = array_xr.UV.sel(station=station_name).sel(time=time_index).data
            NWP_wind_speed = array_xr.NWP_wind_speed.sel(station=station_name).sel(time=time_index).data

        x = array_xr["XX"].sel(station=station_name).data
        y = array_xr["YY"].sel(station=station_name).data
        XX, YY = np.meshgrid(x, y)
        from matplotlib import colors
        divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=NWP_wind_speed, vmax=vmax)
        if size_proportionnal:
            arrows = ax.quiver(XX[::n, ::n], YY[::n, ::n], U[::n, ::n], V[::n, ::n], UV[::n, ::n],
                               scale=scale,
                               cmap=cmap,
                               norm=divnorm,
                               **kwargs)
        else:
            arrows = ax.quiver(XX[::n, ::n], YY[::n, ::n], U[::n, ::n]/UV[::n, ::n], V[::n, ::n]/UV[::n, ::n], UV[::n, ::n],
                               scale=scale,
                               cmap=cmap,
                               norm=divnorm,
                               #norm=MidpointNormalize(midpoint=NWP_wind_speed),
                               **kwargs)
        ax = plt.gca()
        cbar = plt.colorbar(arrows, ax=ax, extend="both", orientation='vertical')
        cbar.ax.tick_params(labelsize=fontsize)

    def plot_predictions_2D(self, array_xr=None, stations_name=['Col du Lac Blanc'],
                            random_selection=True, date_selected=None, scale_cmap=2, fontsize=20, vmin=1, vmax=5,
                            fraction=0.01, vert_exag=10_000, scale_nwp=1/0.04, **kwargs):

        if array_xr is None:
            array_xr = self.predict_UV_with_CNN(self, stations_name)

        if random_selection:
            time_selected = random.choice(list(range(len(array_xr.time.data))))
            station_name = random.choice(stations_name)
        elif len(stations_name) == 1:
            time_selected = date_selected
            station_name = stations_name[0]

        # Plot topography
        #self.plot_topography_around_station_2D(station_name)

        # NWP wind speed and direction
        if random_selection:
            nwp_speed = array_xr.NWP_wind_speed.sel(station=station_name).isel(time=time_selected).data
            nwp_DIR = array_xr.NWP_wind_DIR.sel(station=station_name).isel(time=time_selected).data
        else:
            nwp_speed = array_xr.NWP_wind_speed.sel(station=station_name).sel(time=time_selected).data
            nwp_DIR = array_xr.NWP_wind_DIR.sel(station=station_name).sel(time=time_selected).data
        print(f"NWP wind speed: {nwp_speed}")
        print(f"NWP wind direction: {nwp_DIR}")

        # Experience time
        if random_selection:
            time = array_xr.time.isel(time=time_selected)
        else:
            time = array_xr.time.sel(time=time_selected)

        # Plot results
        for variable in ["U", "V", "W", "UV", "UVW", "UV_DIR_deg", "alpha_deg"]:

            # Define midpoint for color normalization
            if variable in ["U", "UV", "UVW"]:
                midpoint = nwp_speed
            elif variable in ["V", "W", "alpha_deg"]:
                midpoint = 0
            else:
                midpoint = nwp_DIR
            """
            plt.figure(figsize=(10, 20))
            if random_selection:
                display = array_xr[variable].sel(station=station_name).isel(time=time_selected).data
            else:
                display = array_xr[variable].sel(station=station_name).sel(time=time_selected).data
            plt.contourf(XX_station, YY_station, display, cmap='bwr', norm=MidpointNormalize(midpoint=midpoint))
            plt.title(variable + '\n' + station_name + '\n' + str(time))
            """
            if variable in ["UV"]:
                plt.figure(figsize=(20, 30))
                mnt_small = self.p.observation.extract_MNT_around_station(stations_name[0], self.p.mnt, 100, 100)
                topo = mnt_small[0][100 - 79 // 2:100 + 79 // 2 + 1, 100 - 80 // 2: 100 + 80 // 2 + 1]
                XX = mnt_small[1][100 - 80 // 2: 100 + 80 // 2 + 1]
                YY = mnt_small[2][100 - 79 // 2:100 + 79 // 2 + 1]
                x_min = np.min(XX)
                y_max = np.max(YY)
                stations = self.p.observation.stations
                x_dome, y_dome = stations[["X", "Y"]][stations["name"] == "Dome Lac Blanc"].values[0]
                x_lac, y_lac = stations[["X", "Y"]][stations["name"] == "Col du Lac Blanc"].values[0]
                x_muzelle, y_muzelle = stations[["X", "Y"]][stations["name"] == "La Muzelle Lac Blanc"].values[0]
                XX_station = array_xr["XX"].sel(station=stations_name).data
                YY_station = array_xr["YY"].sel(station=stations_name).data
                x_min = np.nanmin(XX_station)
                y_max = np.nanmax(YY_station)

                idx_dome = np.intp((x_dome-x_min) // 30)
                idx_lac = np.intp((x_lac-x_min) // 30)
                idx_muzelle = np.intp((x_muzelle-x_min) // 30)

                idy_dome = np.intp((y_max-y_dome) // 30)
                idy_lac = np.intp((y_max-y_lac) // 30)
                idy_muzelle = np.intp((y_max-y_muzelle) // 30)

                speed_dome = array_xr.sel(time=time_selected, station=stations_name).isel(x=idx_dome, y=idy_dome).UV.values
                direction_dome = array_xr.sel(time=time_selected, station=stations_name).isel(x=idx_dome, y=idy_dome).UV_DIR_deg.values
                print("DEVINE: wind speed dome")
                print(speed_dome)
                print("DEVINE: wind direction dome")
                print(direction_dome)
                speed_lac = array_xr.sel(time=time_selected, station=stations_name).isel(x=idx_lac, y=idy_lac).UV.values
                direction_lac = array_xr.sel(time=time_selected, station=stations_name).isel(x=idx_lac, y=idy_lac).UV_DIR_deg.values
                print("DEVINE: wind speed lac")
                print(speed_lac)
                print("DEVINE: wind direction lac")
                print(direction_lac)
                speed_muzelle = array_xr.sel(time=time_selected, station=stations_name).isel(x=idx_muzelle, y=idy_muzelle).UV.values
                direction_muzelle = array_xr.sel(time=time_selected, station=stations_name).isel(x=idx_muzelle, y=idy_muzelle).UV_DIR_deg.values
                print("DEVINE: wind speed muzelle")
                print(speed_muzelle)
                print("DEVINE: wind direction muzelle")
                print(direction_muzelle)

                print("Domain")
                print(np.min(XX))
                print(np.max(XX))
                print(np.min(YY))
                print(np.max(YY))

                ls = LightSource(azdeg=270, altdeg=45)
                plt.contourf(XX, YY,
                             ls.hillshade(topo, fraction=fraction, vert_exag=vert_exag, dx=35, dy=35),
                             cmap=plt.cm.gray, vmin=0.48, vmax=0.51)
                CS = plt.contour(XX, YY, topo, colors="dimgrey", levels=15, alpha=0.75)
                ax = plt.gca()
                ax.clabel(CS, CS.levels, fmt="%d", inline=True, fontsize=10)

                ax = plt.gca()
                ax.set_aspect('equal', 'box')
                ax.xaxis.set_tick_params(labelsize=fontsize)
                ax.yaxis.set_tick_params(labelsize=fontsize)
                #ax.set_xlim((943_500, 945_500))

                self._plot_arrows_for_CNN(array_xr, time_selected, station_name=station_name, scale_cmap=scale_cmap,
                                          fontsize=fontsize, vmin=vmin, vmax=vmax, **kwargs)
                self._plot_arrow_for_NWP(array_xr, time_selected, station_name=station_name, vmin=vmin, vmax=vmax,
                                         scale_cmap=scale_cmap, scale_nwp=scale_nwp)
                self._plot_arrow_for_observation_station(array_xr, time_selected,
                                                         station_name=station_name,
                                                         nwp_speed=nwp_speed,
                                                         scale_cmap=scale_cmap,
                                                         vmin=vmin,
                                                         vmax=vmax)
                if station_name == "Col du Lac Blanc":
                    self._plot_arrow_for_observation_station(array_xr, time_selected,
                                                             station_name="La Muzelle Lac Blanc", nwp_speed=nwp_speed, scale_cmap=scale_cmap,
                                                             vmin=vmin,
                                                             vmax=vmax)
                    self._plot_arrow_for_observation_station(array_xr, time_selected,
                                                             station_name="Dome Lac Blanc", nwp_speed=nwp_speed, scale_cmap=scale_cmap,
                                                             vmin=vmin,
                                                             vmax=vmax)

        return station_name, time

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

    def plot_predictions_3D(self, array_xr=None, stations_name=['Col du Lac Blanc'], arrow_length_ratio=0.3,
                            length_arrow_modification=1, arrow=True, alpha=1):
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
            self._plot_3D_variable(XX, -YY, ZZ, variable, display, time, midpoint, random_station_name)
            if variable in ["U", "UV", "UVW"]:
                self._plot_arrow_for_NWP_3D(array_xr, random_time_idx, station_name=random_station_name)

        U = array_xr["U"].sel(station=random_station_name).isel(time=random_time_idx).data
        V = array_xr["V"].sel(station=random_station_name).isel(time=random_time_idx).data
        W = array_xr["W"].sel(station=random_station_name).isel(time=random_time_idx).data
        UVW = array_xr["UVW"].sel(station=random_station_name).isel(time=random_time_idx).data
        self._plot_3D_variable(XX, -YY, ZZ, variable, UVW, time, nwp_speed, random_station_name, alpha=alpha)
        if arrow:
            self._plot_arrows_for_CNN_3D(XX, -YY, ZZ, U, V, W, None, nwp_speed, arrow_length_ratio=arrow_length_ratio,
                                         length_arrow_modification=length_arrow_modification)

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
                      length=length * length_arrow_modification,
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

        if isinstance(time_index, int) or isinstance(time_index, float):
            NWP_wind_speed = array_xr.NWP_wind_speed.sel(station=station_name).isel(time=time_index).data
            NWP_wind_DIR = array_xr.NWP_wind_DIR.sel(station=station_name).isel(time=time_index).data
        elif isinstance(time_index, np.datetime64):
            NWP_wind_speed = array_xr.NWP_wind_speed.sel(station=station_name).sel(time=time_index).data
            NWP_wind_DIR = array_xr.NWP_wind_DIR.sel(station=station_name).sel(time=time_index).data

        U_nwp = -np.sin((np.pi / 180) * NWP_wind_DIR) * NWP_wind_speed
        V_nwp = -np.cos((np.pi / 180) * NWP_wind_DIR) * NWP_wind_speed

        ax.quiver(x_nwp, -y_nwp, z_nwp, U_nwp, V_nwp, 0, color='green', length=length)

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

    def qc_plot_validity(self, stations='all', wind_speed='vw10m(m/s)', wind_direction='winddir(deg)',
                         markersize_valid=10, markersize_not_valid=20):

        # Select time series
        time_series = self.p.observation.time_series

        if stations == 'all':
            stations = time_series["name"].unique()

        for station in stations:

            # Select station
            time_serie_station = time_series[time_series["name"] == station]

            # Wind speed
            plt.figure()
            try:
                plt.subplot(211)

                #time_serie_station[wind_speed].plot(marker='x', markersize=markersize_valid)
                #time_serie_station[wind_speed][time_serie_station['validity_speed'] == 0].plot(marker='x',
                #                                                                               linestyle='',
                #                                                                               markersize=markersize_not_valid)

                sns.scatterplot(x=time_serie_station.index,
                                y=wind_speed,
                                data=time_serie_station,
                                hue='validity_speed')
            except:
                pass

            # Wind direction
            try:
                plt.subplot(212)
                #time_serie_station[wind_direction].plot(marker='x', markersize=markersize_valid)
                #time_serie_station[wind_direction][time_serie_station['validity_direction'] == 0].plot(marker='x',
                #                                                                                       linestyle='',
                #                                                                                       markersize=markersize_not_valid)
                sns.scatterplot(x=time_serie_station.index,
                                y=wind_direction,
                                data=time_serie_station,
                                hue='validity_direction')
            except:
                pass

            plt.title(station)

    def qc_plot_resolution(self, stations='all', wind_speed='vw10m(m/s)', wind_direction='winddir(deg)'):

        # Select time series
        time_series = self.p.observation.time_series

        if stations == 'all':
            stations = time_series["name"].unique()

        for station in stations:

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

    def qc_plot_constant(self, stations='all', wind_speed='vw10m(m/s)', wind_direction='winddir(deg)'):

        # Select time series
        time_series = self.p.observation.time_series

        if stations == 'all':
            stations = time_series["name"].unique()

        for station in stations:

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

    def qc_plot_last_flagged(self, stations='all', wind_speed='vw10m(m/s)', wind_direction='winddir(deg)', legend=True,
                             markersize_large=20, markersize_small=1, fig_to_plot=["speed", "direction_1", "direction_2"]):
        # Select time series
        time_series = self.p.observation.time_series

        if stations == 'all':
            stations = time_series["name"].unique()

        nb_subplots = len(fig_to_plot)

        for station in stations:

            # Select station
            time_serie_station = time_series[time_series["name"] == station]

            fig = plt.figure()
            if "speed" in fig_to_plot:
                fig.add_subplot(nb_subplots, 1, 1)
                nb_categories = len(time_serie_station['last_flagged_speed'].dropna().unique())
                index_0 = np.argwhere(time_serie_station['last_flagged_speed'].dropna().unique() == 0)[0][0]
                list_marker_size = [markersize_large for i in range(nb_categories)]
                list_marker_size[index_0] = markersize_small
                sns.scatterplot(x=time_serie_station.index, y=wind_speed, data=time_serie_station, hue='last_flagged_speed',
                                size='last_flagged_speed', sizes=list_marker_size)
                if not legend:
                    ax = plt.gca()
                    ax.legend().set_visible(False)

            if "direction_1" in fig_to_plot:
                fig.add_subplot(nb_subplots, 1, 2)
                nb_categories = len(time_serie_station['last_flagged_direction'].dropna().unique())
                index_0 = np.argwhere(time_serie_station['last_flagged_direction'].dropna().unique() == 0)[0][0]
                list_marker_size = [markersize_large for i in range(nb_categories)]
                list_marker_size[index_0] = markersize_small
                sns.scatterplot(x=time_serie_station.index, y=wind_direction, data=time_serie_station,
                                hue='last_flagged_direction', size='last_flagged_direction', sizes=list_marker_size)
                if not legend:
                    ax = plt.gca()
                    ax.legend().set_visible(False)
            if "direction_2" in fig_to_plot:
                fig.add_subplot(nb_subplots, 1, 3)
                nb_categories = len(time_serie_station['last_unflagged_direction'].dropna().unique())
                index_0 = np.argwhere(time_serie_station['last_unflagged_direction'].dropna().unique() == 0)[0][0]
                list_marker_size = [markersize_large for i in range(nb_categories)]
                list_marker_size[index_0] = markersize_small
                sns.scatterplot(x=time_serie_station.index, y=wind_direction, data=time_serie_station,
                                hue='last_unflagged_direction', size='last_unflagged_direction', sizes=list_marker_size)
                if not legend:
                    ax = plt.gca()
                    ax.legend().set_visible(False)
            plt.title(station)
            plt.tight_layout()

    def qc_plot_bias_speed(self, stations='all', wind_speed='vw10m(m/s)', wind_direction='winddir(deg)',
                           figsize=(10, 10), fontsize=12, update_file=True, df=None):

        # Select time series
        time_series = self.p.observation.time_series

        if stations == 'all':
            stations = time_series["name"].unique()

        for station in stations:

            # Select station
            time_serie_station = time_series[time_series["name"] == station]

            if not (update_file):
                time_serie_station = df

            plt.figure(figsize=figsize)
            time_serie_station[wind_speed].plot(marker='x', linestyle='')
            filter = time_serie_station['qc_6_speed'] != 1
            time_serie_station[wind_speed][filter].plot(marker='d', linestyle='')
            time_serie_station[wind_speed].rolling('30D').mean().plot()
            plt.legend(('Hourly observation', 'Suspicious observation: bias', '30 days rolling mean'), fontsize=fontsize)
            plt.ylabel("Wind speed [m/s]", fontsize=fontsize)

    def qc_plot_bias_correction_factors(self, station='Col du Lac Blanc', metric='mean', list_correct_factor=[1, 2, 4]):

        for correct_factor in list_correct_factor:
            if metric == 'mean':
                time_serie_station = self.p.observation.qc_bias(stations=[station], correct_factor_mean=correct_factor,
                                                                update_file=False)
                self.qc_plot_bias_speed(stations=[station], update_file=False, df=time_serie_station)
                plt.title(f'{metric} threshold divided by {correct_factor}')
            elif metric == 'std':
                time_serie_station = self.p.observation.qc_bias(stations=[station], correct_factor_std=correct_factor,
                                                                update_file=False)
                self.qc_plot_bias_speed(stations=[station], update_file=False, df=time_serie_station)
                plt.title(f'{metric} threshold divided by {correct_factor}')
            else:
                time_serie_station = self.p.observation.qc_bias(stations=[station],
                                                                correct_factor_coeff_var=correct_factor,
                                                                update_file=False)
                self.qc_plot_bias_speed(stations=[station], update_file=False, df=time_serie_station)
                plt.title(f'{metric} threshold divided by {correct_factor}')

    def qc_sankeyy_diagram_speed(self, stations='all', wind_speed='vw10m(m/s)', scale=10):
        from matplotlib.sankey import Sankey
        import matplotlib.pylab as pl

        time_series = self.p.observation.time_series

        if stations == 'all':
            scale = 0.0000001
        else:
            time_series = time_series[time_series["name"].isin(stations)]
            scale = 0.0000001 * scale

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                             title=None)

        sankey = Sankey(ax=ax, scale=scale, offset=0.2, head_angle=135, format="%i", unit=' ')

        all_speeds = time_series[wind_speed].count()
        qc_1 = time_series[wind_speed][time_series["qc_1_speed"] != 1].dropna().count()
        qc_2 = time_series[wind_speed][
            (time_series["qc_1_speed"] == 1) & (time_series["qc_2_speed"] != 1)].dropna().count()
        qc_3 = time_series[wind_speed][
            (time_series["qc_1_speed"] == 1) & (time_series["qc_2_speed"] == 1) & (
                    time_series["qc_3_speed"] != 1)].dropna().count()
        qc_5 = time_series[wind_speed][
            (time_series["qc_1_speed"] == 1) & (time_series["qc_2_speed"] == 1) & (time_series["qc_3_speed"] == 1) & (
                    time_series["qc_5_speed"] != 1)].dropna().count()
        qc_6 = time_series[wind_speed][
            (time_series["qc_1_speed"] == 1) & (time_series["qc_2_speed"] == 1) & (time_series["qc_3_speed"] == 1) & (
                    time_series["qc_5_speed"] == 1) & (time_series["qc_6_speed"] != 1)].dropna().count()

        colors = pl.cm.cividis(np.linspace(0, 1, 5))

        result_1 = (all_speeds - qc_1)
        sankey.add(flows=[all_speeds, -qc_1, -result_1],
                   labels=['All data', 'Unphysical values', None],
                   orientations=[0, -1, 0],
                   facecolor=colors[0])

        # Arguments to matplotlib.patches.PathPatch
        result_2 = result_1 - qc_2
        sankey.add(flows=[result_1, -qc_2, -result_2],
                   labels=[None, "Excessive miss", None],
                   orientations=[0, 1, 0],
                   prior=0,
                   connect=(2, 0),
                   facecolor=colors[1])

        result_3 = result_2 - qc_3
        sankey.add(flows=[result_2, -qc_3, -result_3],
                   labels=[None, "Constant sequences", None],
                   orientations=[0, -1, 0],
                   prior=1,
                   connect=(2, 0),
                   facecolor=colors[2])

        result_4 = result_3 - qc_5
        sankey.add(flows=[result_3, -qc_5, -result_4],
                   labels=[None, "High variability", None],
                   orientations=[0, 1, 0],
                   prior=2,
                   connect=(2, 0),
                   facecolor=colors[3])

        result_5 = result_4 - qc_6
        sankey.add(flows=[result_4, -qc_6, -result_5],
                   labels=[None, "Bias", "Valid observations"],
                   orientations=[0, -1, 0],
                   prior=3,
                   connect=(2, 0),
                   facecolor=colors[4])

        diagrams = sankey.finish()
        diagrams[0].texts[-1].set_color('r')
        diagrams[0].text.set_fontweight('bold')

    def plot_wind_arrows_on_grid(self, nwp=None, U="U", V="V", UV="Wind", UV_DIR="Wind_DIR", x="X_L93", y="Y_L93",
                                 ax=None, time=None, idx_time=None, size_proportionnal=False, **kwargs):

        nwp = self.nwp if nwp is None else nwp

        U_in_keys = U in nwp.keys()
        V_in_keys = V in nwp.keys()
        UV_in_keys = UV in nwp.keys()

        if not U_in_keys and not V_in_keys and not UV_in_keys:
            raise Exception("Wind speed or wind components need to be specified")

        if not (U_in_keys and V_in_keys):
            nwp = self.horizontal_wind_component(library="xarray", xarray_data=nwp, wind_name=UV,
                                                   wind_dir_name=UV_DIR)
        if not UV_in_keys:
            nwp = self.compute_speed_and_direction_xarray(xarray_data=nwp, u_name="U", v_name="V")

        if ax is None:
            plt.figure()
            ax = plt.gca()

        if time is not None and idx_time is not None:
            raise Exception("Time and idx_time can not be specified at the same time")

        if time is not None:
            nwp = nwp.sel(time=time)
        elif idx_time is not None:
            nwp = nwp.isel(time=idx_time)

        x_coord = nwp[x].values
        y_coord = nwp[y].values
        u = nwp[U].values
        v = nwp[V].values
        uv = nwp[UV].values

        if not size_proportionnal:
            u = u/uv
            v = v/uv

        plt.quiver(x_coord, y_coord, u, v, uv, **kwargs)

    def figure_maps_large_domain(self, wind_xr, config=None, prm=None):
        scale = config.get("scale_arrow")
        width = config.get("width_arrow")
        lateral_extent = config.get("lateral_extent")
        idx_cross_section = config.get("idx_cross_section")
        scale_cross_section = config.get("scale_cross_section")
        width_cross_section = config.get("width_cross_section")
        nb_arrows = config.get("nb_arrows_cross_section")
        color_cross_section = config.get("color_cross_section")
        height_arrows_above_cross_section = config.get("height_arrows_above_cross_section")
        cmap_wind_speed = "magma"
        cmap_nwp_arrows = "viridis"
        cmap_alpha = "coolwarm"
        add_station_arrow = config.get("add_station_arrow")
        idx_y=idx_cross_section
        fontsize =  config.get("fontsize")

        time_idx = 0
        time_value = wind_xr.isel(time=time_idx).time.values
        print("Time value")
        print(time_value)
        mnt_values = self.p.mnt.data_xr.values[0, 860:-1310, 2150:-2005]
        wind_values = wind_xr.isel(time=time_idx).Wind.values[860:-1310, 2150:-2005]
        alpha_values = wind_xr.isel(time=time_idx).alpha.values[860:-1310, 2150:-2005]
        U = wind_xr.isel(time=time_idx).U.values[860:-1310, 2150:-2005]
        V = wind_xr.isel(time=time_idx).V.values[860:-1310, 2150:-2005]
        W = wind_xr.isel(time=time_idx).W.values[860:-1310, 2150:-2005]
        print("Shape downscale wind")
        print(U.shape)
        x = wind_xr.x.values[2150:-2005]
        y = wind_xr.y.values[860:-1310]
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)
        print("All domain size")
        print(x_min)
        print(x_max)
        print(y_min)
        print(y_max)
        print("y")
        print(wind_xr.y.max()-wind_xr.y.min())
        print("x")
        print(wind_xr.x.max()-wind_xr.x.min())
        print("Size domain figure")
        print("y")
        print(y_max-y_min)
        print("x")
        print(x_max-x_min)
        dx = (x[1] - x[0]) / 2.
        dy = (y[1] - y[0]) / 2.
        extent = [x[0] - dx, x[-1] + dx, y[-1] - dy, y[0] + dy]
        # Select coordinates MNT
        xx, yy = np.meshgrid(x, y)

        #
        #
        # Figure: MNT
        #
        #
        plt.figure(figsize=(20, 20))
        ax = plt.gca()
        plt.imshow(mnt_values, extent=extent, cmap="viridis")
        plt.plot(xx[idx_y, 650:1250], yy[idx_y, 650:1250], color="black", linewidth=5)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        cbar = plt.colorbar(orientation="horizontal")
        cbar.ax.tick_params(labelsize=fontsize)
        save_figure(f"mnt_{time_value}", prm)

        #
        #
        # Figure: Wind speed
        #
        #
        plt.figure(figsize=(20, 20))

        # Select boundaries
        im = plt.imshow(wind_values, extent=extent, cmap=cmap_wind_speed)
        cbar = plt.colorbar(im, orientation="horizontal")
        cbar.ax.tick_params(labelsize=fontsize)

        NWP1 = self.p.nwp.data_xr.isel(time=time_idx)
        NWP1 = self.p.horizontal_wind_component(xarray_data=NWP1,
                                                library="xarray",
                                                wind_name="Wind",
                                                wind_dir_name="Wind_DIR",
                                                unit_direction="degree",
                                                verbose=True)
        NWP1 = NWP1.where(NWP1.X_L93 <= x_max + lateral_extent).where(NWP1.X_L93 >= x_min - lateral_extent)
        NWP1 = NWP1.where(NWP1.Y_L93 >= y_min - lateral_extent).where(NWP1.Y_L93 <= y_max + lateral_extent)
        NWP1 = NWP1.dropna(dim="xx", how="all")
        print("Max wind speed on domain")
        print(NWP1.Wind.max())
        print("Max speeds of DEVINE domain")
        print(wind_values.max())
        print("Max and min deviations of DEVINE domain")
        print(alpha_values.max())
        print(alpha_values.min())
        X_L93 = NWP1.X_L93.values
        Y_L93 = NWP1.Y_L93.values
        U_NWP = NWP1.U.values
        V_NWP = NWP1.V.values
        Wind_NWP = NWP1.Wind.values

        ax = plt.gca()
        arrows = ax.quiver(X_L93, Y_L93, U_NWP, V_NWP, Wind_NWP,
                           units="dots", cmap=cmap_nwp_arrows, width=width, scale=scale)
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        cbar = plt.colorbar(arrows, orientation="horizontal")
        cbar.ax.tick_params(labelsize=fontsize)

        if add_station_arrow:
            list_x_stations, list_y_stations, stations_in_domain = self.p.find_stations_in_domain(x, y)
            time_series = self.p.observation.time_series
            time_series = time_series[time_series["name"].isin(stations_in_domain)]
            U_stations, V_stations = self.p.horizontal_wind_component(UV=time_series["wind_corrected"].values,
                                                                 UV_DIR=time_series["UV_DIR"].values,
                                                                 unit_direction="degree",
                                                                 verbose=True)
            arrows = ax.quiver(list_x_stations.values, list_y_stations.values, U_stations, V_stations,
                               color="white", units="dots", width=width, scale=scale)

        plt.plot(xx[idx_y, 650:1250], yy[idx_y, 650:1250], color="black", linewidth=5)

        save_figure(f"map_speed_{time_value}", prm)

        #
        #
        # Figure: alpha
        #
        #
        plt.figure(figsize=(20, 20))
        im = plt.imshow(alpha_values, extent=extent, cmap=cmap_alpha)
        cbar = plt.colorbar(im, orientation="horizontal")
        cbar.ax.tick_params(labelsize=fontsize)
        ax = plt.gca()
        arrows = ax.quiver(X_L93,
                           Y_L93,
                           U_NWP,
                           V_NWP,
                           Wind_NWP,
                           units="dots",
                           cmap=cmap_nwp_arrows,
                           width=width,
                           scale=scale)
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
        cbar = plt.colorbar(arrows, orientation="horizontal")
        cbar.ax.tick_params(labelsize=fontsize)
        plt.plot(xx[idx_y, 650:1250], yy[idx_y, 650:1250], color="black", linewidth=5)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        save_figure(f"map_alpha_{time_value}", prm)

        #
        #
        # Figure: cross section
        #
        #
        plt.figure(figsize=(50, 15))
        cross_section = xx[idx_y, 650:1250]
        plt.fill_between(cross_section,
                         mnt_values[idx_y, 650:1250],
                         color=color_cross_section)
        ax = plt.gca()
        ax.quiver(cross_section[::nb_arrows],
                  mnt_values[idx_y, 650:1250:nb_arrows] + height_arrows_above_cross_section,
                  U[idx_y, 650:1250:nb_arrows],
                  W[idx_y, 650:1250:nb_arrows],
                  scale=scale_cross_section,
                  width=width_cross_section)
        plt.plot(xx[idx_y, 1250-115:1250-90], np.full_like(xx[idx_y, 1250-115:1250-90], 2550), linewidth=5, color='red')
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.ylim(500, 4000)
        save_figure(f"cross_section_idx_y_{idx_y}_{time_value}", prm)


        #
        #
        # Figure: ZOOM cross section
        #
        #
        plt.figure(figsize=(25, 15))
        cross_section = xx[idx_y, 1250-115:1250-93]
        print("Size")
        print(len(cross_section)*30)
        plt.fill_between(cross_section,
                         mnt_values[idx_y, 1250-115:1250-93],
                         color=color_cross_section)
        ax = plt.gca()
        nb_arrows=1
        ax.quiver(cross_section[::nb_arrows],
                  mnt_values[idx_y, 1250-115:1250-93:nb_arrows] + 15,
                  U[idx_y, 1250-115:1250-93:nb_arrows],
                  W[idx_y, 1250-115:1250-93:nb_arrows],
                  scale=scale_cross_section/2.5,
                  width=width_cross_section*2)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.ylim(2550, 2730)
        save_figure(f"ZOOM_cross_section_idx_y_{idx_y}_{time_value}", prm)

    def plot_features_maps(self, dem, station="Col du Lac Blanc", dependencies=True):
        import tensorflow as tf

        self.load_cnn(dependencies=dependencies)
        model = self.model
        model.trainable = False
        successive_outputs = [layer.output for layer in model.layers[1:]]
        visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)

        observation = self.p.observation
        x = observation.extract_MNT_around_station("Col du Lac Blanc", station, dem, 69//2+1, 79//2+1)[0][:79, :69]
        x = x.reshape((1, 79, 69, 1))
        successive_feature_maps = visualization_model.predict(x)
        layer_names = np.array([layer.name for layer in model.layers[1:]])

        for index, (layer_name, feature_map) in enumerate(zip(layer_names, successive_feature_maps)):

            # if (("conv" in layer_name) or ("pool" in layer_name)) and (index <= 100) and (index >= 0):
            #print(index, feature_map.shape, layer_name)

            n_features = feature_map.shape[-1]

            fig = plt.figure()
            plt.title(f"{layer_name}")
            for i in range(n_features):
                ax = plt.gca()
                fig.add_subplot(np.int32(np.sqrt(n_features)) + 1, np.int32(np.sqrt(n_features)) + 1, i + 1)
                x = feature_map[0, :, :, i]
                x = x - np.nanmean(x)
                x = x / np.where(np.std(x) == 0, 0.01, np.std(x))
                plt.imshow(x)
                ax.set_xticks([])
                ax.set_yticks([])



