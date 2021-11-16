import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from downscale.visu.visualization import Visualization

class VisualizationGaussian(Visualization):

    def __init__(self, p=None, prm={"verbose": True}):
        super().__init__(p=p, prm=prm)

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
        plt.colorbar(arrows, orientation='vertical')

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
            uv = self.compute_wind_speed(u, v)
            uvw = self.compute_wind_speed(u, v, w)
            uv_flat = uv.flatten()
            uvw_flat = uvw.flatten()

            if type_plot == "acceleration" and type_of_wind == "uv":
                acc_uv = self.wind_speed_ratio(num=uv_flat, den=np.full(uv_flat.shape, 3))
                var = acc_uv
                label = "Acceleration distribution"

            if type_plot == "acceleration" and type_of_wind == "uvw":
                acc_uvw = self.wind_speed_ratio(num=uv_flat, den=np.full(uv_flat.shape, 3))
                var = acc_uvw
                label = "Acceleration distribution"

            if type_plot == "wind speed" and type_of_wind == "uv":
                var = uv_flat
                label = "Wind speed distribution"

            if type_plot == "wind speed" and type_of_wind == "uvw":
                var = uvw_flat
                label = "Wind speed distribution"

            if type_plot == "angular deviation":
                var = np.rad2deg(self.angular_deviation(u, v)).flatten()
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