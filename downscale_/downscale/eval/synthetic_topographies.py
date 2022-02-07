import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from time import time as t

from downscale.utils.utils_func import print_current_line, save_figure
from downscale.operators.wind_utils import Wind_utils
from downscale.operators.helbig import DwnscHelbig
from downscale.operators.micro_met import MicroMet
from downscale.operators.processing import Processing
from downscale.visu.visu_gaussian import VisualizationGaussian
from downscale.visu.MidpointNormalize import MidpointNormalize
from downscale.utils.utils_func import save_figure


class GaussianTopo(Wind_utils, DwnscHelbig, MicroMet):

    def __init__(self):
        super().__init__()

    def filenames_to_array(self, df_all, input_dir, name):
        """
        Reading data from filename stored in dataframe. Returns an array

        For degree:
        e.g. dem/5degree/gaussiandem_N5451_dx30_xi200_sigma18_r000.txt

        For wind:
        e.g.
        Wind/ucompwind/5degree/gaussianu_N5451_dx30_xi200_sigma18_r000.txt
        Wind/vcompwind/5degree/gaussianv_N5451_dx30_xi200_sigma18_r000.txt
        Wind/wcompwind/5degree/gaussianw_N5451_dx30_xi200_sigma18_r000.txt

        """
        print("__Begin loading synthetic topographies") if name == 'topo_name' else None
        print("__Begin loading winds on synthetic topographies") if name == 'wind_name' else None
        all_data = []

        for simu in range(len(df_all.values)):

            degree_folder = str(df_all['degree'].values[simu]) + 'degree/'

            # Topographies
            if name == 'topo_name':
                path = input_dir + 'dem/' + degree_folder
                donnees, data = self.read_values(path + df_all[name].values[simu])

            # Wind maps
            if name == 'wind_name':
                u_v_w = []

                # Wind components
                for index, wind_component in enumerate(['ucompwind/', 'vcompwind/', 'wcompwind/']):
                    path = input_dir + 'Wind/U_V_W/' + wind_component + degree_folder
                    uvw_simu = df_all[name].values[simu].split('(')[1].split(')')[0].split(', ')
                    donnees, data = self.read_values(path + uvw_simu[index].split("'")[1])
                    u_v_w.append(data.reshape(79 * 69))

                u_v_w = np.array(list(zip(*u_v_w)))
                assert np.shape(u_v_w) == (79 * 69, 3)
                data = u_v_w

            # Store data
            all_data.append(data)

            # Print execution
            print_current_line(simu, len(df_all.values), 5)

        print("__End loading synthetic topographies") if name == 'topo_name' else None
        print("__End loading winds on synthetic topographies") if name == 'wind_name' else None

        return np.array(all_data)

    def read_values(self, filename):
        """
        Reading ARPS (from Nora) .txt files

        Input:
               path file

        Output:
                1. Data description as described in the files (entête)
                2. Data (2D matrix)
        """

        # opening the file, type TextIOWrapper
        with open(filename, 'r') as fichier:
            entete = []

            # Reading info
            for _ in range(6):
                entete.append(fichier.readline())

            donnees = []
            # Data
            for string in entete:
                donnees = np.loadtxt(entete, dtype=int, usecols=[1])

            ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value = donnees

            # Data begins at line 7
            data = np.zeros((nrows, ncols), dtype=float)
            for i in range(nrows):
                ligne = fichier.readline()
                ligne = ligne.split()
                data[i] = [float(ligne[j]) for j in range(ncols)]

            return donnees, data

    def load_data_all(self, prm, verbose=True, name_topo='topo_name', name_wind='wind_name'):
        """
        Load all synthetic data

        Read a .csv file with filenames and load corresponding data
        """

        # Loading data
        path = f'{prm["gaussian_topo_path"]}df_all_{0}.csv'
        df_all = pd.read_csv(path)

        # Training data
        print("Loading topographies") if verbose else None
        synthetic_topo = self.filenames_to_array(df_all, prm["gaussian_topo_path"], name_topo)
        print("Loading wind") if verbose else None
        synthetic_wind = self.filenames_to_array(df_all, prm["gaussian_topo_path"], name_wind)

        return synthetic_topo, synthetic_wind

    def load_data_by_degree_or_xi(self, prm, degree_or_xi="degree"):

        path = f'{prm["gaussian_topo_path"]}df_all_{0}.csv'
        df_all = pd.read_csv(path)

        dict_gaussian = defaultdict(lambda: defaultdict(dict))

        for degree in df_all[degree_or_xi].unique():

            print(degree)
            filter_degree = df_all[degree_or_xi] == degree
            df_all_degree = df_all[filter_degree]

            dict_gaussian["topo"][degree] = self.filenames_to_array(df_all_degree,
                                                                    prm["gaussian_topo_path"],
                                                                    'topo_name')

            dict_gaussian["wind"][degree] = self.filenames_to_array(df_all_degree,
                                                                    prm["gaussian_topo_path"],
                                                                    'wind_name')
        return dict_gaussian

    @staticmethod
    def classify(df, variable="mu", quantile=False):

        df[f"class_{variable}"] = np.nan

        if quantile:
            q25 = df[variable].quantile(0.25)
            q50 = df[variable].quantile(0.5)
            q75 = df[variable].quantile(0.75)

            filter_1 = (df[variable] <= q25)
            filter_2 = (df[variable] > q25) & (df[variable] <= q50)
            filter_3 = (df[variable] > q50) & (df[variable] <= q75)
            filter_4 = (df[variable] > q75)

            df[f"class_{variable}"][filter_1] = "$x \leq q_{25}$"
            df[f"class_{variable}"][filter_2] = "$q_{25}<x \leq q_{50}$"
            df[f"class_{variable}"][filter_3] = "$q_{50}<x \leq q_{75}$"
            df[f"class_{variable}"][filter_4] = "$q_{75}<x$"

            print(f"Quantiles {variable}: ", q25, q50, q75)

        else:
            if variable == "mu":

                filter_1 = (df[variable] <= 0.5)
                filter_2 = (df[variable] > 0.5) & (df[variable] <= 1)
                filter_3 = (df[variable] > 1) & (df[variable] <= 1.5)
                filter_4 = (df[variable] > 1.5)

                df[f"class_{variable}"][filter_1] = "mu <= 0.5"
                df[f"class_{variable}"][filter_2] = "0.5 < mu <= 1"
                df[f"class_{variable}"][filter_3] = "1 < mu <= 1.5"
                df[f"class_{variable}"][filter_4] = "1.5 < mu"

            elif variable == "tpi_500":

                filter_1 = (df[variable] >= 0) & (df[variable] <= 100)
                filter_2 = (df[variable] < 0) & (df[variable] >= -100)
                filter_3 = (df[variable] > 100)
                filter_4 = (df[variable] < -100)

                df[f"class_{variable}"][filter_1] = "0 <= tpi <= 100"
                df[f"class_{variable}"][filter_2] = "-100 <= tpi < 0"
                df[f"class_{variable}"][filter_3] = "100 < tpi"
                df[f"class_{variable}"][filter_4] = "tpi < -100"

            elif variable == "sx_300":

                filter_1 = (df[variable] >= 0) & (df[variable] <= 50 * 0.001)
                filter_2 = (df[variable] < 0) & (df[variable] >= -50 * 0.001)
                filter_3 = (df[variable] > 50 * 0.001) & (df[variable] <= 100 * 0.001)
                filter_4 = (df[variable] <= -50 * 0.001) & (df[variable] > -100 * 0.001)
                filter_5 = (df[variable] > 100 * 0.001)
                filter_6 = (df[variable] < -100 * 0.001)

                df[f"class_{variable}"][filter_1] = "0 < sx <= 0.05"
                df[f"class_{variable}"][filter_2] = "-0.05 < sx < 0"
                df[f"class_{variable}"][filter_3] = "0.05 < sx <= 0.1"
                df[f"class_{variable}"][filter_4] = "-0.1 < sx <= -0.05"
                df[f"class_{variable}"][filter_5] = "0.1 < sx"
                df[f"class_{variable}"][filter_6] = "sx <= -0.1"

        return df

    def load_initial_and_predicted_data_in_df(self, prm):
        p = Processing(prm=prm)
        p.load_model(dependencies=True)
        _, std = p._load_norm_prm()

        df_all = pd.read_csv(prm["path_to_synthetic_topo"]).drop(columns=["degree_xi", 'Unnamed: 0'])
        topos = self.filenames_to_array(df_all, prm["gaussian_topo_path"], 'topo_name')
        winds = self.filenames_to_array(df_all, prm["gaussian_topo_path"], 'wind_name').reshape(len(topos), 79, 69, 3)

        # Normalize data
        mean_topos = np.mean(topos, axis=(1, 2)).reshape(len(topos), 1, 1)
        std = std.reshape(1, 1, 1)
        topo_norm = self.normalize_topo(topos, mean_topos, std)

        # Predict test data
        predictions = p.model.predict(topo_norm)

        for type_of_data in ["pred", "test"]:
            for variable in [f"U_{type_of_data}", f"V_{type_of_data}", f"W_{type_of_data}"]:
                df_all[variable] = ""

        # Define U_test, V_test, W_test, U_pred etc
        for index in range(len(df_all)):
            df_all["topo_name"].iloc[index] = topos[index, :, :]
            df_all["U_test"].iloc[index] = winds[index, :, :, 0]
            df_all["V_test"].iloc[index] = winds[index, :, :, 1]
            df_all["W_test"].iloc[index] = winds[index, :, :, 2]
            df_all["U_pred"].iloc[index] = predictions[index, :, :, 0]
            df_all["V_pred"].iloc[index] = predictions[index, :, :, 1]
            df_all["W_pred"].iloc[index] = predictions[index, :, :, 2]

        # DataFrame to array
        U_test, V_test, W_test = df_all["U_test"].values, df_all["V_test"].values, df_all["W_test"].values
        U_pred, V_pred, W_pred = df_all["U_pred"].values, df_all["V_pred"].values, df_all["W_pred"].values

        # Compute wind and direction
        df_all["UV_test"] = [self.compute_wind_speed(U_test[index], V_test[index], verbose=False) for index in range(len(U_test))]
        df_all["UVW_test"] = [self.compute_wind_speed(U_test[index], V_test[index], W_test[index], verbose=False)
                              for index in range(len(U_test))]
        df_all["alpha_test"] = [self.angular_deviation(U_test[index], V_test[index], verbose=False) for index in
                                range(len(U_test))]

        df_all["UV_pred"] = [self.compute_wind_speed(U_pred[index], V_pred[index], verbose=False) for index in
                             range(len(U_pred))]
        df_all["UVW_pred"] = [self.compute_wind_speed(U_pred[index], V_pred[index], W_pred[index], verbose=False)
                              for index in range(len(U_pred))]
        df_all["alpha_pred"] = [self.angular_deviation(U_pred[index], V_pred[index], verbose=False) for index in
                                range(len(U_pred))]

        return df_all

    @staticmethod
    def unnesting(df, explode):
        idx = df.index.repeat(df[explode[0]].str.len())
        df1 = pd.concat([
            pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
        df1.index = idx
        return df1.join(df.drop(explode, 1), how='left')

    def df_with_list_in_rows_to_flatten_df(self, df_all, list_variables=['topo_name', 'U_test', 'V_test',
                          'W_test', 'U_pred', 'V_pred', 'W_pred', 'UV_test', 'UVW_test',
                          'alpha_test', 'UV_pred', 'UVW_pred', 'alpha_pred']):

        for variable in list_variables:
            df_all[variable] = df_all[variable].apply(lambda x: np.array(x).flatten())

        df_all_flat = self.unnesting(df_all, list_variables)

        return df_all_flat


    def figure_example_topo_wind_gaussian(self, config, prm):

        topo, wind = self.load_data_all(prm)
        topo = topo.reshape((len(topo), 79, 69))
        print("Nb synthetic topographies: ", len(topo))
        visu = VisualizationGaussian()
        wind = wind.reshape((len(wind), 79, 69, 3))

        #
        #
        # 2D and 3D maps on gaussian grid
        #
        #

        cmap_topo = config.get("cmap_topo")
        cmap_arrow = config.get("cmap_arrow")
        midpoint = config.get("midpoint")
        scale = config.get("scale")
        range_idx_to_plot = config.get("range_idx_to_plot")
        n = config.get("n")
        vmin = config.get("vmin")
        vmax = config.get("vmax")
        fontsize = config.get("fontsize")
        fontsize_3D = config.get("fontsize_3D")

        u, v, w = wind[:, :, :, 0], wind[:, :, :, 1], wind[:, :, :, 2]
        uv = self.compute_wind_speed(U=u, V=v)
        path = f'{prm["gaussian_topo_path"]}df_all_{0}.csv'
        df_all = pd.read_csv(path)

        for idx_simu in range_idx_to_plot:
            degree = np.int32(df_all["degree"].iloc[idx_simu])
            xi = np.int32(df_all["xi"].iloc[idx_simu])
            u1 = u[idx_simu, :, :]
            v1 = v[idx_simu, :, :]
            uv1 = uv[idx_simu, :, :]
            topo1 = topo[idx_simu, :, :]

            #
            #
            # 3D map
            #
            #
            plt.figure(figsize=(20,20))
            nb_col = 69
            nb_lin = 79
            x = np.arange(nb_col)
            y = np.arange(nb_lin)

            XX, YY = np.meshgrid(x, y)
            XX = XX * 30
            YY = YY * 30
            fig = plt.gcf()
            ax = fig.gca(projection='3d')
            im = ax.plot_surface(XX, YY, topo1, cmap=cmap_topo, lw=0.5, rstride=1, cstride=1, alpha=1, linewidth=1)
            visu._set_axes_equal(ax)
            ax.xaxis.set_tick_params(labelsize=fontsize_3D)
            ax.yaxis.set_tick_params(labelsize=fontsize_3D)
            #ax.zaxis.set_tick_params(labelsize=fontsize_3D)

            ax.zaxis.set_ticklabels([])
            for line in ax.zaxis.get_ticklines():
                line.set_visible(False)

            cbar = plt.colorbar(im, orientation="vertical")
            cbar.ax.tick_params(labelsize=fontsize)

            ax = plt.gca()
            ii = 270
            ax.view_init(elev=35, azim=ii)
            save_figure(f"3D_map_{idx_simu}_degree_{degree}_xi_{xi}", prm)

            #
            #
            # 2D map with arrows
            #
            #
            plt.figure()
            ax = plt.gca()
            arrows = ax.quiver(XX[::n, ::n], YY[::n, ::n], u1[::n, ::n] / uv1[::n, ::n], v1[::n, ::n] / uv1[::n, ::n],
                               uv1[::n, ::n],
                               cmap=cmap_arrow,
                               scale=scale,
                               norm=MidpointNormalize(midpoint=midpoint, vmin=vmin, vmax=vmax))
            plt.axis("square")
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            cbar = plt.colorbar(arrows)
            cbar.ax.tick_params(labelsize=fontsize)
            save_figure(f"2D_map_{idx_simu}_degree_{degree}_xi_{xi}", prm)


    def figure_tpi_vs_sx(self, dem, obs, config, prm):

        down_helb = DwnscHelbig()

        # Select domain dem
        mnt = dem.data[config["min_y_dem"]:config["max_y_dem"], config["min_x_dem"]:config["max_x_dem"]]
        print("DEM domain selected")

        # Load sunthetic topo and winds
        gaussian_topo = GaussianTopo()
        topo, _ = gaussian_topo.load_data_all(prm)
        topo = topo.reshape((len(topo), 79, 69)).astype(np.float32)
        print("Nb synthetic topographies: ", len(topo))

        if config["working_on_a_small_example"]:
            topo = topo[:100, :, :]
            mnt = mnt[1000:1200, 1300:1400]

        # TPI and Sx on real topographies
        print("Begin computing TPI on real topographies")
        t0 = t()
        tpi_real = down_helb.tpi_map(mnt, config["distance_tpi"], resolution=config["resolution_dem"]).astype(np.float32)
        print(f"Time in minutes: {np.round((t() - t0) / 60)}")
        print("End computing TPI on real topographies")

        print("Begin computing Sx on real topographies")
        t0 = t()
        sx_real_1 = down_helb.sx_map(mnt, config["resolution_dem"], config["distance_sx"], config["angle_sx"]).astype(np.float32)
        print(f"Time in minutes: {np.round((t() - t0) / 60)}")
        print("End computing Sx on real topographies")

        # Remove borders where TPI and Sx are not defined
        tpi_real[:17, :] = np.nan
        tpi_real[-17:, :] = np.nan
        tpi_real[:, :17] = np.nan
        tpi_real[:, -17:] = np.nan
        sx_real_1[:10, :] = np.nan
        sx_real_1[-10:, :] = np.nan
        sx_real_1[:, :10] = np.nan
        sx_real_1[:, -10:] = np.nan
        print("Border Sx and TPI removed")

        tpi_real_flat = tpi_real.flatten()
        sx_real_flat_1 = sx_real_1.flatten()
        print("TPI and sx real flat calculated")

        # TPI and Sx on gaussian topographies
        print("Begin computing TPI on gaussian topographies")
        t0 = t()
        tpi_gau = np.array([down_helb.tpi_map(topo[i], config["distance_tpi"], config["resolution_dem"]) for i in range(len(topo))]).astype(np.float32)
        print(f"Time in minutes: {np.round((t() - t0) / 60)}")
        print("End computing TPI on gaussian topographies")

        print("Begin computing Sx on gaussian topographies")
        t0 = t()
        sx_gau = np.array([down_helb.sx_map(topo[i], config["resolution_dem"], config["distance_sx"], config["angle_sx"]) for i in range(len(topo))]).astype(np.float32)
        print(f"Time in minutes: {np.round((t() - t0) / 60)}")
        print("End computing Sx on gaussian topographies")

        print("tpi_gau shape")
        print(np.shape(tpi_gau))
        tpi_gau[:17, :] = np.nan
        tpi_gau[-17:, :] = np.nan
        tpi_gau[:, :17] = np.nan
        tpi_gau[:, -17:] = np.nan
        print("tpi_gau shape")
        print(np.shape(sx_gau))
        sx_gau[:10, :] = np.nan
        sx_gau[-10:, :] = np.nan
        sx_gau[:, :10] = np.nan
        sx_gau[:, -10:] = np.nan
        print("Border Sx and TPI removed on gaussian topographies")

        tpi_gau_flat = tpi_gau.flatten()
        sx_gau_flat = sx_gau.flatten()
        print("TPI and sx gau flat calculated")

        df_topo_gaussian = pd.DataFrame(np.transpose([tpi_gau_flat, sx_gau_flat]), columns=["tpi", "sx"])
        df_topo_gaussian["topography"] = "gaussian"
        df_topo_real_1 = pd.DataFrame(np.transpose([tpi_real_flat, sx_real_flat_1]), columns=["tpi", "sx"])
        df_topo_real_1["topography"] = "real"
        df_topo = pd.concat([df_topo_gaussian, df_topo_real_1])
        print("Dataframe created")
        del tpi_real_flat
        del sx_gau_flat

        # Observations
        obs.update_stations_with_KNN_from_MNT_using_cKDTree(dem)
        idx_x = np.array([np.intp(idx_x) for (idx_x, _) in
                          obs.stations[f"index_{prm['name_mnt']}_NN_0_cKDTree_ref_{prm['name_mnt']}"].to_list()])
        idx_y = np.array([np.intp(idx_y) for (_, idx_y) in
                          obs.stations[f"index_{prm['name_mnt']}_NN_0_cKDTree_ref_{prm['name_mnt']}"].to_list()])
        #names = obs.stations["name"].values
        tpi_stations = down_helb.tpi_idx(dem.data, idx_x=idx_x, idx_y=idx_y, radius=config["distance_tpi"], resolution=config["resolution_dem"]).astype(np.float32)
        sx_stations = down_helb.sx_idx(dem.data, idx_x, idx_y, config["resolution_dem"], config["distance_sx"], config["angle_sx"], 5, 30).astype(np.float32)
        print("TPI and sx stations calculated")

        # Unique values for scatter plot
        df_topo1 = df_topo.dropna().drop_duplicates()
        tpi_gau_unique = df_topo1["tpi"][df_topo1["topography"] == "gaussian"]
        sx_gau_unique = df_topo1["sx"][df_topo1["topography"] == "gaussian"]
        print("TPI and sx unique calculated")
        del df_topo1

        tpi_gau_unique = tpi_gau_unique.astype(np.float32)
        sx_gau_unique = sx_gau_unique.astype(np.float32)
        df_topo["tpi"] = df_topo["tpi"].astype(np.float32)
        df_topo["sx"] = df_topo["sx"].astype(np.float32)
        print("Data transformed to float32")

        print("Begin plot figure tpi vs sx")
        plt.figure(figsize=(15, 15))
        df_topo.index = list(range(len(df_topo)))
        result = sns.jointplot(data=df_topo.dropna(), x="tpi", y="sx", hue="topography",
                               palette=[config["color_real"], config["color_gaussian"]], marker="o", s=3,
                               linewidth=0, edgecolor=None, hue_order=["real", "gaussian"], legend=False,
                               marginal_kws=dict(bw=0.8))
        del df_topo
        ax = result.ax_joint
        ax.scatter(tpi_gau_unique, sx_gau_unique, s=3, alpha=0.75, color=config["color_gaussian"])
        ax.scatter(tpi_stations, sx_stations, s=7, alpha=1, color=config["color_station"], zorder=10)
        ax.grid(True)
        ax = result.ax_marg_x
        ax.scatter(tpi_stations, np.zeros_like(tpi_stations), s=5, alpha=1, color=config["color_station"], zorder=10)
        ax.grid(True)
        ax = result.ax_marg_y
        ax.scatter(np.zeros_like(sx_stations), sx_stations, s=5, alpha=1, color=config["color_station"], zorder=10)
        ax.grid(True)
        save_figure("tpi_vs_sx", prm, svg=config["svg"])
        print("End plot figure tpi vs sx")

    def figure_laplacian_vs_mu(self, dem, obs, config, prm):
        down_helb = DwnscHelbig()

        # Select domain dem
        mnt = dem.data[config["min_y_dem"]:config["max_y_dem"], config["min_x_dem"]:config["max_x_dem"]]
        print("DEM domain selected")

        # Load sunthetic topo and winds
        gaussian_topo = GaussianTopo()
        topo, _ = gaussian_topo.load_data_all(prm)
        topo = topo.reshape((len(topo), 79, 69))
        print("Nb synthetic topographies: ", len(topo))

        if config["working_on_a_small_example"]:
            topo = topo[:100, :, :]
            mnt = mnt[:100, :150]

        print("Begin computing Laplacian on gaussian topographies")
        t0 = t()
        laplacian_gaussian = down_helb.laplacian_map(topo, config["resolution_dem"], helbig=config["laplacian_helbig"], verbose=False)
        print(f"Time in minutes: {np.round((t() - t0) / 60)}")
        print("End computing Laplacian on gaussian topographies")

        print("Begin computing Mu on gaussian topographies")
        t0 = t()
        mu_gaussian = down_helb.mu_helbig_map(topo, config["resolution_dem"], verbose=False)
        print(f"Time in minutes: {np.round((t() - t0) / 60)}")
        print("End computing Mu on gaussian topographies")

        laplacian_gaussian_flat = laplacian_gaussian[:, 1:-1, 1:-1].flatten()
        mu_gaussian_flat = mu_gaussian[:, 1:-1, 1:-1].flatten()
        print("Gaussian topographies flatten")

        print("Begin computing Laplacian on real topographies")
        t0 = t()
        laplacian_real = down_helb.laplacian_map(mnt, config["resolution_dem"], helbig=config["laplacian_helbig"], verbose=False)
        print(f"Time in minutes: {np.round((t() - t0) / 60)}")
        print("End computing Laplacian on real topographies")

        print("Begin computing Mu on real topographies")
        t0 = t()
        mu_real = down_helb.mu_helbig_map(mnt, config["resolution_dem"], verbose=False)
        print(f"Time in minutes: {np.round((t() - t0) / 60)}")
        print("End computing Mu on real topographies")

        laplacian_real_flat = laplacian_real[1:-1, 1:-1].flatten()
        mu_real_flat = mu_real[1:-1, 1:-1].flatten()
        print("Real topographies flatten")

        df_topo_gaussian = pd.DataFrame(np.transpose([laplacian_gaussian_flat, mu_gaussian_flat]),
                                        columns=["laplacian", "mu"])
        df_topo_gaussian["topography"] = "gaussian"
        df_topo_real = pd.DataFrame(np.transpose([laplacian_real_flat, mu_real_flat]), columns=["laplacian", "mu"])
        df_topo_real["topography"] = "real"
        df_topo = pd.concat([df_topo_gaussian, df_topo_real])
        print("Dataframe created")
        df_topo1 = df_topo.drop_duplicates()
        lapl_gau_unique = df_topo1["laplacian"][df_topo1["topography"] == "gaussian"]
        mu_gau_unique = df_topo1["mu"][df_topo1["topography"] == "gaussian"]

        obs.update_stations_with_KNN_from_MNT_using_cKDTree(dem)
        idx_x = np.array([np.intp(idx_x) for (idx_x, _) in
                          obs.stations[f"index_{prm['name_mnt']}_NN_0_cKDTree_ref_{prm['name_mnt']}"].to_list()])
        idx_y = np.array([np.intp(idx_y) for (_, idx_y) in
                          obs.stations[f"index_{prm['name_mnt']}_NN_0_cKDTree_ref_{prm['name_mnt']}"].to_list()])
        names = obs.stations["name"].values
        mu_stations = down_helb.mu_helbig_idx(dem.data, dx=config["resolution_dem"], idx_x=idx_x, idx_y=idx_y)
        laplacian_stations = down_helb.laplacian_idx(dem.data, dx=config["resolution_dem"], idx_x=idx_x, idx_y=idx_y, helbig=config["laplacian_helbig"])

        df_topo.index = list(range(len(df_topo)))
        result = sns.jointplot(data=df_topo.dropna(), x="laplacian", y="mu", hue="topography", marker="o", s=5,
                               palette=[config["color_real"], config["color_gaussian"]], linewidth=0, edgecolor=None, hue_order=["real", "gaussian"],
                               legend=False)
        ax = result.ax_joint
        ax.scatter(lapl_gau_unique, mu_gau_unique, s=5, alpha=0.75, color=config["color_gaussian"])
        ax.scatter(laplacian_stations, mu_stations, s=5, alpha=1, color=config["color_station"], zorder=10)
        ax.grid(True)
        ax = result.ax_marg_x
        ax.scatter(laplacian_stations, np.zeros_like(laplacian_stations), s=5, alpha=1, color=config["color_station"], zorder=10)
        ax.grid(True)
        ax = result.ax_marg_y
        ax.scatter(np.zeros_like(mu_stations), mu_stations, s=5, alpha=1, color=config["color_station"], zorder=10)
        ax.grid(True)
        save_figure("laplacian_vs_mu", prm, svg=config["svg"])
