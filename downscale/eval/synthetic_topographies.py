import numpy as np
import pandas as pd
from collections import defaultdict

from downscale.utils.utils_func import print_current_line
from downscale.operators.wind_utils import Wind_utils
from downscale.operators.helbig import DwnscHelbig
from downscale.operators.micro_met import MicroMet
from downscale.operators.processing import Processing


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

    def load_data_all(self, prm, verbose=True):
        """
        Load all synthetic data

        Read a .csv file with filenames and load corresponding data
        """

        # Loading data
        path = f'{prm["gaussian_topo_path"]}df_all_{0}.csv'
        df_all = pd.read_csv(path)

        # Training data
        print("Loading topographies") if verbose else None
        synthetic_topo = self.filenames_to_array(df_all, prm["gaussian_topo_path"], 'topo_name')
        print("Loading wind") if verbose else None
        synthetic_wind = self.filenames_to_array(df_all, prm["gaussian_topo_path"], 'wind_name')

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

            df[f"class_{variable}"][filter_1] = f"{variable} <= q25"
            df[f"class_{variable}"][filter_2] = f"q25 < {variable} <= q50"
            df[f"class_{variable}"][filter_3] = f"q50 < {variable} <= q75"
            df[f"class_{variable}"][filter_4] = f"q75 < {variable}"

            print("Quantiles: ", q25, q50, q75)
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

    """
    def load_data_train_test_by_fold(self, prm):

        for fold_nb in range(10):
            path = f"C:/Users/louis/git/wind_downscaling_CNN/Data/2_Pre_processed/ARPS/fold/fold{fold_nb}/df_all_{fold_nb}.csv"

            df_all = pd.read_csv(path)
            df_all['wind_name'] = [x.strip('()').split(',') for x in df_all['wind_name']]

    dict_gaussian = defaultdict(lambda: defaultdict(dict))


    dict_gaussian["wind"][degree] = self.filenames_to_array(df_all_degree,
                                                            prm["gaussian_topo_path"],
                                                            'wind_name')

    df = pd.DataFrame([(k,k1,value) for k,v in results.items() for k1,v1 in v.items() for value in v1], columns = ['Col1','Col2','Val'])
    """
