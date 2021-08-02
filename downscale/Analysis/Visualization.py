import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from scipy.ndimage import rotate
import random  # Warning
from time import time as t
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d  # Warning

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

from downscale.Analysis.MidpointNormalize import MidpointNormalize


class Visualization:
    _shapely_geometry = _shapely_geometry
    _geopandas = _geopandas
    _cartopy = _cartopy

    def __init__(self, p=None):
        t0 = t()
        self.p = p
        if _cartopy:
            self.l93 = ccrs.epsg(2154)
        t1 = t()
        print(f"\nVizualisation created in {np.round(t1 - t0, 2)} seconds\n")

    @staticmethod
    def density_scatter(x, y, ax=None, sort=True, use_power_norm=1, bins=20, **kwargs):
        """
        https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib/53865762#53865762
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.colors import Normalize, PowerNorm
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
            plt.scatter(x, y, c=z, vmin=0, norm=PowerNorm(gamma=0.25), **kwargs)
        elif use_power_norm == 2:
            plt.scatter(x, y, c=z, norm=PowerNorm(gamma=0.25), **kwargs)
        else:
            plt.scatter(x, y, c=z, **kwargs)
        plt.colorbar()
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
        self.p.load_model(dependencies=True)

        import visualkeras
        from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D, \
            Cropping2D, InputLayer
        from collections import defaultdict
        import matplotlib
        import matplotlib.pylab as pl
        from PIL import ImageFont
        color_map = defaultdict(dict)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        colors = pl.cm.ocean(norm(np.linspace(0, 1, 9)), bytes=True)

        color_map[Conv2D]['fill'] = tuple(colors[7])
        color_map[ZeroPadding2D]['fill'] = tuple(colors[6])
        color_map[Dropout]['fill'] = tuple(colors[5])
        color_map[MaxPooling2D]['fill'] = tuple(colors[4])
        color_map[Dense]['fill'] = tuple(colors[3])
        color_map[Flatten]['fill'] = tuple(colors[2])
        color_map[Cropping2D]['fill'] = tuple(colors[1])
        color_map[InputLayer]['fill'] = tuple(colors[0])

        font = ImageFont.truetype("arial.ttf", 35)  # using comic sans is strictly prohibited!
        visualkeras.layered_view(self.model, color_map=color_map, legend=True, draw_volume=True, draw_funnel=True,
                                 shade_step=0, font=font, scale_xy=2, scale_z=0.5, to_file='output85.png')
        # tf.keras.utils.plot_model(self.model, to_file='Model1.png')
        # tf.keras.utils.plot_model(self.model, to_file='Model2.png', show_shapes=True)

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
            self._plot_3D_variable(XX, YY, ZZ, variable, display, time, midpoint, random_station_name)
            if variable in ["U", "UV", "UVW"]:
                self._plot_arrow_for_NWP_3D(array_xr, random_time_idx, station_name=random_station_name)

        U = array_xr["U"].sel(station=random_station_name).isel(time=random_time_idx).data
        V = array_xr["V"].sel(station=random_station_name).isel(time=random_time_idx).data
        W = array_xr["W"].sel(station=random_station_name).isel(time=random_time_idx).data
        UVW = array_xr["UVW"].sel(station=random_station_name).isel(time=random_time_idx).data
        self._plot_3D_variable(XX, YY, ZZ, variable, UVW, time, nwp_speed, random_station_name, alpha=alpha)
        if arrow:
            self._plot_arrows_for_CNN_3D(XX, YY, ZZ, U, V, W, None, nwp_speed, arrow_length_ratio=arrow_length_ratio,
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
                time_serie_station[wind_speed].plot(marker='x', markersize=markersize_valid)
                time_serie_station[wind_speed][time_serie_station['validity_speed'] == 0].plot(marker='x',
                                                                                               linestyle='',
                                                                                               markersize=markersize_not_valid)
            except:
                pass

            # Wind direction
            try:
                plt.subplot(212)
                time_serie_station[wind_direction].plot(marker='x', markersize=markersize_valid)
                time_serie_station[wind_direction][time_serie_station['validity_direction'] == 0].plot(marker='x',
                                                                                                       linestyle='',
                                                                                                       markersize=markersize_not_valid)
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

    def qc_plot_last_flagged(self, stations='all', wind_speed='vw10m(m/s)', wind_direction='winddir(deg)',
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

            if "direction_1" in fig_to_plot:
                fig.add_subplot(nb_subplots, 1, 2)
                nb_categories = len(time_serie_station['last_flagged_direction'].dropna().unique())
                index_0 = np.argwhere(time_serie_station['last_flagged_direction'].dropna().unique() == 0)[0][0]
                list_marker_size = [markersize_large for i in range(nb_categories)]
                list_marker_size[index_0] = markersize_small
                sns.scatterplot(x=time_serie_station.index, y=wind_direction, data=time_serie_station,
                                hue='last_flagged_direction', size='last_flagged_direction', sizes=list_marker_size)

            if "direction_2" in fig_to_plot:
                fig.add_subplot(nb_subplots, 1, 3)
                nb_categories = len(time_serie_station['last_unflagged_direction'].dropna().unique())
                index_0 = np.argwhere(time_serie_station['last_unflagged_direction'].dropna().unique() == 0)[0][0]
                list_marker_size = [markersize_large for i in range(nb_categories)]
                list_marker_size[index_0] = markersize_small
                sns.scatterplot(x=time_serie_station.index, y=wind_direction, data=time_serie_station,
                                hue='last_unflagged_direction', size='last_unflagged_direction', sizes=list_marker_size)

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
            filter = time_serie_station['qc_bias_observation_speed'] == 1
            time_serie_station[wind_speed][filter].plot(marker='d', linestyle='')
            time_serie_station[wind_speed].rolling('30D').mean().plot()
            plt.legend(('Hourly observation', 'Suspicious obseration: bias', '30 days rolling mean'), fontsize=fontsize)

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

    def qc_sankey_diagram_speed(self, stations='all', wind_speed='vw10m(m/s)', scale=10):
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


class VisualizationGaussian(Visualization):

    def __init__(self, p=None):
        super().__init__(p=p)

    @staticmethod
    def plot_gaussian_wind_arrows(U, V, UV, nb_col=69, nb_lin=79, midpoint=3, new_fig=True):

        x = np.arange(nb_col)
        y = np.arange(nb_lin)

        XX, YY = np.meshgrid(x, y)

        plt.figure() if new_fig else None

        ax = plt.gca()
        arrows = ax.quiver(XX, YY, U, V, UV,
                           scale=1 / 0.005,
                           cmap='coolwarm',
                           norm=MidpointNormalize(midpoint=midpoint))
        plt.colorbar(arrows, orientation='horizontal')

    def plot_gaussian_topo_and_wind_2D(self, topo, u, v, w, uv, alpha, idx_simu, arrows=True, cmap="mako"):

        u = u[idx_simu, :, :]
        v = v[idx_simu, :, :]
        w = w[idx_simu, :, :]
        uv = uv[idx_simu, :, :]
        alpha = alpha[idx_simu, :, :]
        topo = topo[idx_simu, :, :]

        plt.figure()
        plt.imshow(topo, cmap=cmap)
        self.plot_gaussian_wind_arrows(u, v, uv, new_fig=False) if arrows else None
        plt.colorbar()
        plt.title("Topography")

        plt.figure()
        plt.imshow(uv, cmap=cmap)
        self.plot_gaussian_wind_arrows(u, v, uv, new_fig=False) if arrows else None
        plt.colorbar()
        plt.title("Wind speed")

        plt.figure()
        plt.imshow(u, cmap=cmap)
        self.plot_gaussian_wind_arrows(u, v, uv, new_fig=False) if arrows else None
        plt.colorbar()
        plt.title("U")

        plt.figure()
        plt.imshow(v, cmap=cmap)
        self.plot_gaussian_wind_arrows(u, v, uv, new_fig=False) if arrows else None
        plt.colorbar()
        plt.title("V")

        plt.figure()
        plt.imshow(w, cmap=cmap)
        self.plot_gaussian_wind_arrows(u, v, uv, new_fig=False) if arrows else None
        plt.colorbar()
        plt.title("W")

        plt.figure()
        plt.imshow(alpha, cmap=cmap)
        self.plot_gaussian_wind_arrows(u, v, uv, new_fig=False) if arrows else None
        plt.colorbar()
        plt.title("Angular deviation")

    @staticmethod
    def plot_gaussian_distrib(data_1=None, data_2=None, type_plot="acceleration"):

        if type_plot == "acceleration":
            xlabel = "Acceleration distribution []"
        elif type_plot == "wind speed":
            xlabel = "Wind speed distribution []"
        elif type_plot == "angular deviation":
            xlabel = "Wind speed distribution []"
        else:
            xlabel=""

        if type_plot in ["acceleration", "wind speed"]:
            # Flatten matrixes
            data_1_flat = data_1.flatten() if data_1 is not None else None
            data_2_flat = data_2.flatten() if data_2 is not None else None

            # Create DataFrame
            df = pd.DataFrame()
            if data_1_flat is not None:
                df["UV"] = data_1_flat
            if data_2_flat is not None:
                df["UVW"] = data_2_flat

            # melt DataFrame
            UV_in_df = "UV" in list(df.columns.values)
            UVW_in_df = "UVW" in list(df.columns.values)

            if UV_in_df and UVW_in_df:
                df = df.melt(value_vars=["UV", "UVW"])
                print(df)
                sns.displot(data=df, x="value", hue="variable", kind='kde', fill=True)
                plt.xlabel(xlabel)
            elif UV_in_df:
                sns.displot(data_1_flat, kde=True)
                plt.xlabel(xlabel)
            else:
                sns.displot(data_2_flat, kde=True)
                plt.xlabel(xlabel)

        if type_plot == "angular deviation":
            sns.displot(data_1.flatten(), kde=True)
            plt.xlabel(xlabel)

    def plot_gaussian_distrib_by_degree_or_xi(self, dict_gaussian, degree_or_xi="degree", type_plot="acceleration",
                                              type_of_wind="uv", fill=True, fontsize=20, cmap="viridis"):
        """
        To obtain the input dict_gaussian DataFrame, use the command:
        gaussian_topo_instance.load_data_by_degree_or_xi(prm, degree_or_xi="degree")
        """
        dict_all_degree_or_xi = pd.DataFrame(columns=["value", degree_or_xi])
        for deg_or_xi in dict_gaussian["wind"].keys():

            # Wind components
            u = dict_gaussian["wind"][deg_or_xi][:, :, 0]
            v = dict_gaussian["wind"][deg_or_xi][:, :, 1]
            w = dict_gaussian["wind"][deg_or_xi][:, :, 2]

            # Compute wind speed and reformat
            uv = self.p.compute_wind_speed(u, v)
            uvw = self.p.compute_wind_speed(u, v, w)
            uv_flat = uv.flatten()
            uvw_flat = uvw.flatten()

            if type_plot == "acceleration" and type_of_wind == "uv":
                acc_uv = self.p.wind_speed_ratio(num=uv_flat, den=np.full(uv_flat.shape, 3))
                var = acc_uv
                label = "Acceleration distribution"

            if type_plot == "acceleration" and type_of_wind == "uvw":
                acc_uvw = self.p.wind_speed_ratio(num=uv_flat, den=np.full(uv_flat.shape, 3))
                var = acc_uvw
                label = "Acceleration distribution"

            if type_plot == "wind speed" and type_of_wind == "uv":
                var = uv_flat
                label = "Wind speed distribution"

            if type_plot == "wind speed" and type_of_wind == "uvw":
                var = uvw_flat
                label = "Wind speed distribution"

            if type_plot == "angular deviation":
                var = np.rad2deg(self.p.angular_deviation(u, v)).flatten()
                label = "Angular deviation"

            if type_plot == "test":
                var = uv_flat[:1000]
                label = "Wind speed distribution test"

            # List of degrees to append to DataFrame
            list_deg_or_xi = [deg_or_xi] * len(var)
            df_deg_or_xi_i = pd.DataFrame(np.transpose([var, list_deg_or_xi]), columns=dict_all_degree_or_xi.columns)
            dict_all_degree_or_xi = dict_all_degree_or_xi.append(df_deg_or_xi_i, ignore_index=True)

        sns.displot(data=dict_all_degree_or_xi, x="value", hue=degree_or_xi, kind='kde', fill=fill, palette=cmap)
        plt.xlabel(label, fontsize=fontsize)

