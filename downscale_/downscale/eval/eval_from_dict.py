import uuid

import numpy as np
import pandas as pd
import matplotlib
from scipy.stats import ttest_ind

matplotlib.use('Agg')
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ModuleNotFoundError:
    pass

from downscale.visu.MidpointNormalize import MidpointNormalize
from downscale.eval.evaluation import Evaluation
from downscale.utils.utils_func import save_figure
from downscale.eval.synthetic_topographies import GaussianTopo

class EvaluationFromDict(Evaluation):

    def __init__(self, v, prm={"verbose": True}):
        super().__init__(v, prm=prm)

    @staticmethod
    def _drop_nans_and_duplicates_at_station_results(df):

        # Drop Nans
        df = df.dropna()

        # Drop duplicates
        df = df[~df.index.duplicated(keep='first')]

        # Convert dictionary values to DataFrame (if necessary)
        df = df.to_frame() if isinstance(df, pd.core.series.Series) else df

        return df

    def intersection_model_obs_on_results(self, results, variables=["UV"],
                                          use_QC=False, time_series_qc=None, variable_qc="vw10m(m/s)", verbose=True):
        """
        Keep only data were we have simultaneously model and observed data

        Input = dictionary
        Output = dictionary
        """

        for variable in variables:

            if variable == "UV_DIR_deg":
                variable_qc = "winddir(deg)"

            for station in results[variable]["cnn"].keys():
                nwp, cnn, obs = [self._drop_nans_and_duplicates_at_station_results(results[variable][data_key][station])
                                 for data_key in ["nwp", "cnn", "obs"]]

                # Keep index intersection
                index_intersection = nwp.index.intersection(cnn.index)
                if use_QC:
                    qc_filter_station = time_series_qc["name"] == station
                    qc = self._drop_nans_and_duplicates_at_station_results(
                        time_series_qc[variable_qc][qc_filter_station])
                    index_intersection = index_intersection.intersection(qc.index)
                    _, nwp, cnn, qc = [df[df.index.isin(index_intersection)] for df in [obs, nwp, cnn, qc]]
                    results[variable]["obs"][station] = qc
                else:
                    index_intersection = index_intersection.intersection(obs.index)
                    obs, nwp, cnn = [df[df.index.isin(index_intersection)] for df in [obs, nwp, cnn]]
                    results[variable]["obs"][station] = obs

                results[variable]["nwp"][station] = nwp
                results[variable]["cnn"][station] = cnn

                assert len(results[variable]["nwp"][station]) == len(results[variable]["obs"][station])
                assert len(results[variable]["cnn"][station]) == len(results[variable]["obs"][station])

        return results

    def create_df_from_dict(self, results, data_type="cnn", variables=["UV"]):
        """
        Creates a DataFrame with results at each observation station

        result = create_df_from_dict(results, data_type="cnn", variables="UV")
        result =
                                UV             name
        time
        2017-09-01 00:00:00  0.470177    BARCELONNETTE
        2017-09-01 01:00:00  0.746627    BARCELONNETTE
                               ...              ...
        2020-05-31 19:00:00  3.009761  Col du Lautaret
        2020-05-31 20:00:00  2.814069  Col du Lautaret

        """

        result_df = pd.DataFrame()

        result_var = []
        for index, variable in enumerate(variables):

            result_df_var_i = []
            list_stations = []
            for station in results[variable][data_type].keys():

                df_station = self._drop_nans_and_duplicates_at_station_results(results[variable][data_type][station])

                if len(result_df) != 0:
                    index_intersection = df_station.index.intersection(result_df[result_df["name"] == station].index)
                    df_station = df_station[df_station.index.isin(index_intersection)]

                result_df_var_i.append(df_station)
                list_stations.extend([station] * len(df_station))

            result_df_var_i = pd.concat(result_df_var_i)

            if isinstance(result_df_var_i, pd.core.series.Series):
                result_df_var_i = result_df_var_i.to_frame()

            result_df_var_i.columns = [variable]
            if "name" not in result_df_var_i.columns:
                result_df_var_i["name"] = list_stations

            if index == 0:
                result_df = result_df_var_i
            else:
                result_df1 = []
                for station in results[variable][data_type].keys():

                    index_intersection = result_df_var_i.index.intersection(result_df.index)
                    filter_station_2 = result_df_var_i["name"] == station
                    filter_station_1 = result_df["name"] == station

                    var1 = result_df[result_df.index.isin(index_intersection) & filter_station_1]
                    var2 = result_df_var_i[result_df_var_i.index.isin(index_intersection) & filter_station_2]
                    var2 = var2.drop(columns="name")
                    result_df1.append(pd.concat([var1, var2], axis=1))

                result_df = pd.concat(result_df1, axis=0)

        # result_df = pd.concat(result_df, axis=1)

        return result_df

    def update_df_with_topo_carac(self, df, variables=["laplacian", "alti", "mu", "tpi_2000", "tpi_500", "curvature"]):
        """update result DataFrame with topographical information"""
        if not self.v.p.is_updated_with_topo_characteristics:
            self.v.p.update_station_with_topo_characteristics()

        stations = self.v.p.observation.stations

        for variable in variables:

            df[variable] = np.nan

            for station in stations["name"].values:
                filter_station_df = df["name"] == station
                filter_station = stations["name"] == station
                df[variable][filter_station_df] = stations[variable][filter_station].values[0]

        return df

    @staticmethod
    def _group_metric_result_by_alti_category(metric_result, z0=1000, z1=1700, z2=2500):
        metric_result["group_alti"] = np.nan

        filter_low = metric_result["alti"] < z0
        metric_result["group_alti"][filter_low] = f"Station altitude <= {z0}m"
        print(f"Nb station in category Station altitude <= {z0}m")
        print(len(metric_result["name"][filter_low].unique()))
        print(metric_result["name"][filter_low].unique())

        filter_low = z0 < metric_result["alti"]
        filter_high = metric_result["alti"] <= z1
        metric_result["group_alti"][filter_low & filter_high] = f"{z0}m < Station altitude <= {z1}m"
        print(f"Nb station in category {z0}m < Station altitude <= {z1}m")
        print(len(metric_result["name"][filter_low & filter_high].unique()))
        print(metric_result["name"][filter_low & filter_high].unique())

        filter_low = z1 < metric_result["alti"]
        filter_high = metric_result["alti"] <= z2
        metric_result["group_alti"][filter_low & filter_high] = f"{z1}m < Station altitude <= {z2}m"
        print(f"Nb station in category {z1}m < Station altitude <= {z2}m")
        print(len(metric_result["name"][filter_low & filter_high].unique()))
        print(metric_result["name"][filter_low & filter_high].unique())

        filter_low = metric_result["alti"] > z2
        metric_result["group_alti"][filter_low] = f"{z2}m < Station altitude"
        print(f"Nb station in category {z2}m < Station altitude")
        print(len(metric_result["name"][filter_low].unique()))
        print(metric_result["name"][filter_low].unique())

        return metric_result

    @staticmethod
    def _group_metric_result_by_wind_category(metric_result, variable, w1=5, w2=10, w3=15):
        metric_result[f"group_{variable}_nwp"] = np.nan

        filter_low = metric_result[f"nwp_{variable}"] < w1
        metric_result[f"group_{variable}_nwp"][filter_low] = f"NWP wind speed < {w1} m/s"
        print(f"NWP wind speed < {w1} m/s: {len(metric_result[f'group_{variable}_nwp'][filter_low])} ")

        filter_low = metric_result[f"nwp_{variable}"] >= w1
        filter_high = metric_result[f"nwp_{variable}"] < w2
        metric_result[f"group_{variable}_nwp"][filter_low & filter_high] = f"{w1} m/s <= NWP wind speed < {w2} m/s"
        print(
            f"{w1} m/s <= NWP wind speed < {w2} m/s: "
            f"{len(metric_result[f'group_{variable}_nwp'][filter_low & filter_high])} ")

        filter_low = metric_result[f"nwp_{variable}"] >= w2
        filter_high = metric_result[f"nwp_{variable}"] < w3
        metric_result[f"group_{variable}_nwp"][filter_low & filter_high] = f"{w2} m/s <= NWP wind speed < {w3} m/s"
        print(f"{w2} m/s <= NWP wind speed < {w3} m/s: "
              f"{len(metric_result[f'group_{variable}_nwp'][filter_low & filter_high])} ")

        filter_low = metric_result[f"nwp_{variable}"] >= w3
        metric_result[f"group_{variable}_nwp"][filter_low] = f"{w3} m/s <= NWP wind speed"
        print(f"{w3} m/s <= NWP wind speed: {len(metric_result[f'group_{variable}_nwp'][filter_low])} ")

        return metric_result

    @staticmethod
    def prepare_direction_data_for_evaluation(obs, nwp, cnn, metric_result, variable="UV_DIR_deg", speed_min=0.5):

        assert len((obs["UV"] >= speed_min).values) == len((nwp["UV"] >= speed_min).values)
        assert len((cnn["UV"] >= speed_min).values) == len((nwp["UV"] >= speed_min).values)

        condition_speed = (obs["UV"] >= speed_min).values & (nwp["UV"] >= speed_min).values & (cnn["UV"] >= speed_min).values

        metric_result = metric_result[condition_speed]
        nwp = nwp[condition_speed]
        cnn = cnn[condition_speed]
        obs = obs[condition_speed]

        # Unify true north
        nwp[nwp[variable] == 0] = 360
        cnn[cnn[variable] == 0] = 360

        print(f"__Evaluate direction for speeds >= 0.1")

        return obs, nwp, cnn, metric_result

    @staticmethod
    def _select_variables_to_extract(variable):
        if "DIR" in variable or "direction" in variable:
            variables_to_extract = ["UV_DIR_deg", "UV"]
        else:
            variables_to_extract = [variable]
        return variables_to_extract

    def create_latex_table(self, results, use_QC=False, time_series_qc=None):

        variables_to_extract = ["UV", "UV_DIR_deg"]
        results = self.intersection_model_obs_on_results(results, variables=variables_to_extract,
                                                         use_QC=use_QC, time_series_qc=time_series_qc)
        cnn, nwp, obs = [self.create_df_from_dict(results, data_type=data_type, variables=variables_to_extract)
                         for data_type in ["cnn", "nwp", "obs"]]
        metric_result = cnn.copy()
        index_variable = ["Speed"] * 6 + ["Direction"] * 4
        index_metric = ["Bias", "$\rho$", "mean MAE", "q25 MAE", "median MAE", "q75 MAE", "mean MAE",
                        "q25 MAE", "median MAE", "q75 MAE"]

        tuples = list(zip(*[index_variable, index_metric]))

        list_biases_speed = np.zeros(4)
        list_rho_speed = np.zeros(4)
        list_rho_direction = np.zeros(4)
        list_mean_mae_speed = np.zeros(4)
        list_mean_mae_direction = np.zeros(4)
        list_q25_mae_speed = np.zeros(4)
        list_q25_mae_direction = np.zeros(4)
        list_median_mae_speed = np.zeros(4)
        list_median_mae_direction = np.zeros(4)
        list_q75_mae_speed = np.zeros(4)
        list_q75_mae_direction = np.zeros(4)

        for variable, str_variable in zip(variables_to_extract, ["Speed", "Direction"]):
            if str_variable == "Direction":
                obs, nwp, cnn, metric_result = self.prepare_direction_data_for_evaluation(obs, nwp, cnn, metric_result,
                                                                                          variable="UV_DIR_deg")
            if str_variable == "Speed":
                metric = "bias"
                metric_2 = "abs_error"
            else:
                metric = "bias_direction"
                metric_2 = "abs_bias_direction"

            # variables
            metric_result[f"nwp_{variable}"] = nwp[variable].values
            metric_result[f"obs_{variable}"] = obs[variable].values

            # bias
            metric_func = self._select_metric(metric)
            metric_result[f"cnn_{metric}_{variable}"] = metric_func(cnn[variable].values, obs[variable].values)
            metric_result[f"nwp_{metric}_{variable}"] = metric_func(nwp[variable].values, obs[variable].values)

            _, p_value = ttest_ind(metric_result[f"nwp_{metric}_{variable}"].values, metric_result[f"cnn_{metric}_{variable}"].values, nan_policy='omit')
            print("p_value computation DONE")
            if p_value < 0.05:
                print(f"{metric} {variable} mean differences significant")
                print(f"p_value: {p_value}")
            else:
                print(f"{metric} {variable} mean differences NOT significant")
                print(f"p_value: {p_value}")

            # absolute error
            metric_func = self._select_metric(metric_2)
            metric_result[f"cnn_{metric_2}_{variable}"] = metric_func(cnn[variable].values, obs[variable].values)
            metric_result[f"nwp_{metric_2}_{variable}"] = metric_func(nwp[variable].values, obs[variable].values)

            print("\ndebug44")
            print("metric computation 1 DONE")
            print(metric_result[f"cnn_{metric}_{variable}"].head())
            print(metric_result[f"nwp_{metric}_{variable}"].head())

            _, p_value = ttest_ind(metric_result[f"nwp_{metric_2}_{variable}"].values, metric_result[f"cnn_{metric_2}_{variable}"].values, nan_policy='omit')
            if p_value < 0.05:
                print(f"{metric_2} {variable} mean differences significant")
                print(f"p_value: {p_value}")
            else:
                print(f"{metric_2} {variable} mean differences NOT significant")
                print(f"p_value: {p_value}")

            # Statistics AROME
            nwp_corr_coeff = metric_result[[f"nwp_{variable}", f"obs_{variable}"]].corr().iloc[0, 1]
            nwp_mae = metric_result[f"nwp_{metric_2}_{variable}"].mean()
            nwp_q25 = metric_result[f"nwp_{metric_2}_{variable}"].quantile(0.25)
            nwp_median = metric_result[f"nwp_{metric_2}_{variable}"].median()
            nwp_q75 = metric_result[f"nwp_{metric_2}_{variable}"].quantile(0.75)

            # Statistics DEVINE
            cnn_corr_coeff = metric_result[[variable, f"obs_{variable}"]].corr().iloc[0, 1]
            cnn_mae = metric_result[f"cnn_{metric_2}_{variable}"].mean()
            cnn_q25 = metric_result[f"cnn_{metric_2}_{variable}"].quantile(0.25)
            cnn_median = metric_result[f"cnn_{metric_2}_{variable}"].median()
            cnn_q75 = metric_result[f"cnn_{metric_2}_{variable}"].quantile(0.75)

            if str_variable == "Speed":
                nwp_mean_bias = metric_result[f"nwp_{metric}_{variable}"].mean()
                cnn_mean_bias = metric_result[f"cnn_{metric}_{variable}"].mean()

                list_biases_speed[:2] = np.array([nwp_mean_bias, cnn_mean_bias])
                list_rho_speed[:2] = np.array([nwp_corr_coeff, cnn_corr_coeff])
                list_mean_mae_speed[:2] = np.array([nwp_mae, cnn_mae])
                list_q25_mae_speed[:2] = np.array([nwp_q25, cnn_q25])
                list_median_mae_speed[:2] = np.array([nwp_median, cnn_median])
                list_q75_mae_speed[:2] = np.array([nwp_q75, cnn_q75])
            else:
                #list_rho_direction[:2] = np.array([nwp_corr_coeff, cnn_corr_coeff])
                list_mean_mae_direction[:2] = np.array([nwp_mae, cnn_mae])
                list_q25_mae_direction[:2] = np.array([nwp_q25, cnn_q25])
                list_median_mae_direction[:2] = np.array([nwp_median, cnn_median])
                list_q75_mae_direction[:2] = np.array([nwp_q75, cnn_q75])

        filter_direction = metric_result["nwp_abs_bias_direction_UV_DIR_deg"] <= 30
        filter_speed = metric_result["nwp_abs_error_UV"] <= 3
        metric_result_1 = metric_result[filter_direction & filter_speed]
        print(len(metric_result_1))
        for variable, str_variable in zip(variables_to_extract, ["Speed", "Direction"]):

            if str_variable == "Speed":
                metric = "bias"
                metric_2 = "abs_error"
            else:
                metric = "bias_direction"
                metric_2 = "abs_bias_direction"

            _, p_value = ttest_ind(metric_result_1[f"nwp_{metric}_{variable}"].values,
                                   metric_result_1[f"cnn_{metric}_{variable}"].values, nan_policy='omit')
            if p_value < 0.05:
                print("AROME correct")
                print(f"{metric} {variable} mean differences significant")
                print(p_value)
            else:
                print(f"{metric} {variable} mean differences NOT significant")
                print(p_value)

            _, p_value = ttest_ind(metric_result_1[f"nwp_{metric_2}_{variable}"].values,
                                   metric_result_1[f"cnn_{metric_2}_{variable}"].values, nan_policy='omit')
            if p_value < 0.05:
                print("AROME correct")
                print(f"{metric_2} {variable} mean differences significant")
                print(p_value)
            else:
                print(f"{metric_2} {variable} mean differences NOT significant")
                print(p_value)

            # Statistics AROME when AROME direction is correct
            nwp_corr_coeff = metric_result_1[[f"nwp_{variable}", f"obs_{variable}"]].corr().iloc[0, 1]
            nwp_mae = metric_result_1[f"nwp_{metric_2}_{variable}"].mean()
            nwp_q25 = metric_result_1[f"nwp_{metric_2}_{variable}"].quantile(0.25)
            nwp_median = metric_result_1[f"nwp_{metric_2}_{variable}"].median()
            nwp_q75 = metric_result_1[f"nwp_{metric_2}_{variable}"].quantile(0.5)

            # Statistics DEVINE when AROME direction is correct
            cnn_corr_coeff = metric_result_1[[variable, f"obs_{variable}"]].corr().iloc[0, 1]

            cnn_mae = metric_result_1[f"cnn_{metric_2}_{variable}"].mean()
            cnn_q25 = metric_result_1[f"cnn_{metric_2}_{variable}"].quantile(0.25)
            cnn_median = metric_result_1[f"cnn_{metric_2}_{variable}"].median()
            cnn_q75 = metric_result_1[f"cnn_{metric_2}_{variable}"].quantile(0.5)

            if str_variable == "Speed":
                cnn_mean_bias = metric_result_1[f"cnn_{metric}_{variable}"].mean()
                nwp_mean_bias = metric_result_1[f"nwp_{metric}_{variable}"].mean()

                list_biases_speed[2:] = [nwp_mean_bias, cnn_mean_bias]
                list_rho_speed[2:] = [nwp_corr_coeff, cnn_corr_coeff]
                list_mean_mae_speed[2:] = [nwp_mae, cnn_mae]
                list_q25_mae_speed[2:] = [nwp_q25, cnn_q25]
                list_median_mae_speed[2:] = [nwp_median, cnn_median]
                list_q75_mae_speed[2:] = [nwp_q75, cnn_q75]
            else:
                #list_rho_direction[2:] = [cnn_corr_coeff, cnn_corr_coeff]
                list_mean_mae_direction[2:] = [nwp_mae, cnn_mae]
                list_q25_mae_direction[2:] = [nwp_q25, cnn_q25]
                list_median_mae_direction[2:] = [nwp_median, cnn_median]
                list_q75_mae_direction[2:] = [nwp_q75, cnn_q75]

        index = pd.MultiIndex.from_tuples(tuples, names=["Variable", "Metric"])

        all_stat = np.array([np.array(list_biases_speed), np.array(list_rho_speed), np.array(list_mean_mae_speed), np.array(list_q25_mae_speed), np.array(list_median_mae_speed),
                    np.array(list_q75_mae_speed), np.array(list_mean_mae_direction), np.array(list_q25_mae_direction), np.array(list_median_mae_direction),
                    np.array(list_q75_mae_direction)])
        print(np.shape(all_stat))
        df = pd.DataFrame(all_stat.reshape((10, 4)), index=index)  # , columns=["AROME", "DEVINE", "DEVINE when AROME is correct"]
        print(df)
        print(df.to_latex())
        return df

    def compute_stats(self, results, variable="UV", variable_qc="vw10m(m/s)", metric="abs_error", metric_2="bias",
                      use_QC=False, time_series_qc=None):

        variables_to_extract = self._select_variables_to_extract(variable)

        results = self.intersection_model_obs_on_results(results, variables=variables_to_extract,
                                                         use_QC=use_QC, time_series_qc=time_series_qc,
                                                         variable_qc=variable_qc)

        cnn, nwp, obs = [self.create_df_from_dict(results, data_type=data_type, variables=variables_to_extract)
                         for data_type in ["cnn", "nwp", "obs"]]

        cnn["hour"] = cnn.index.hour
        cnn["month"] = cnn.index.month

        metric_result = cnn.copy()
        metric_result = self.update_df_with_topo_carac(metric_result)

        if 'direction' in metric:
            obs, nwp, cnn, metric_result = self.prepare_direction_data_for_evaluation(obs, nwp, cnn, metric_result,
                                                                                      variable=variable)

        # metric
        metric_func = self._select_metric(metric)
        metric_result[metric] = metric_func(cnn[variable].values, obs[variable].values)
        metric_result[f"nwp_{variable}"] = nwp[variable].values
        metric_result[f"nwp_{metric}"] = metric_func(nwp[variable].values, obs[variable].values)
        metric_result[f"nwp_cnn_{metric}"] = metric_func(nwp[variable].values, cnn[variable].values)
        print("the maximum modification in speed or direction introduced by DEVINE, in comparison to AROME, is obtained at the Vallot station and equals t38 \°")
        print(metric_result[["name", f"nwp_cnn_{metric}"]][metric_result[f"nwp_cnn_{metric}"] == metric_result[f"nwp_cnn_{metric}"].max()])
        # metric_2
        metric_func = self._select_metric(metric_2)
        metric_result[metric_2] = metric_func(cnn[variable].values, obs[variable].values)
        metric_result[f"nwp_{metric_2}"] = metric_func(nwp[variable].values, obs[variable].values)
        metric_result[f"nwp_cnn_{metric_2}"] = metric_func(nwp[variable].values, cnn[variable].values)
        print("the maximum modification in speed or direction introduced by DEVINE, in comparison to AROME, is obtained at the Vallot station and equals t38 \°")
        print(metric_result[["name", f"nwp_cnn_{metric_2}"]][metric_result[f"nwp_cnn_{metric}"] == metric_result[f"nwp_cnn_{metric}"].max()])

        improvments_mae = 0
        degradation_mae = 0
        no_improvment_mae = 0
        improvments_rmse = 0
        degradation_rmse = 0
        no_improvment_rmse = 0
        improvments_bias = 0
        degradation_bias = 0
        no_improvment_bias = 0
        above_2000m = 0
        station_above_2000m = 0
        for station in metric_result["name"].unique():
            df_station = metric_result[metric_result["name"] == station]

            if df_station["alti"].iloc[0] >= 1500:
                station_above_2000m += 1

            print(f"\n\nResults at {station}")
            print("MAE NWP")
            mae_nwp = df_station[f"nwp_{metric}"].mean()
            print(mae_nwp)
            print("MAE CNN")
            mae_cnn = df_station[metric].mean()
            print(mae_cnn)
            print("p_value mae")
            _, p_value_mae = ttest_ind(df_station[f"nwp_{metric}"].values, df_station[metric].values, nan_policy='omit')
            print(p_value_mae)
            print("Median AE NWP")
            median_nwp = df_station[f"nwp_{metric}"].median()
            print(median_nwp)
            print("Median AE CNN")
            median_cnn = df_station[metric].median()
            print(median_cnn)
            print("Q25 AE NWP")
            q25_nwp = df_station[f"nwp_{metric}"].quantile(0.25)
            print(q25_nwp)
            print("Q25 AE CNN")
            q25_cnn = df_station[metric].quantile(0.25)
            print(q25_cnn)
            print("Q75 AE NWP")
            q75_nwp = df_station[f"nwp_{metric}"].quantile(0.75)
            print(q75_nwp)
            print("Q75 AE CNN")
            q75_cnn = df_station[metric].quantile(0.75)
            print(q75_cnn)

            print("\nRMSE NWP")
            rmse_nwp = np.sqrt(np.mean(df_station[f"nwp_{metric}"].values ** 2))
            print(rmse_nwp)
            print("RMSE CNN")
            rmse_cnn = np.sqrt(np.mean(df_station[metric].values ** 2))
            print(rmse_cnn)
            print("p_value rmse")
            _, p_value_rmse = ttest_ind(df_station[f"nwp_{metric}"].values ** 2, df_station[metric].values ** 2,
                                        nan_policy='omit')
            print(p_value_rmse)

            print("\nBias NWP")
            bias_nwp = df_station[f"nwp_{metric_2}"].mean()
            print(bias_nwp)
            print("Bias CNN")
            bias_cnn = df_station[metric_2].mean()
            print(bias_cnn)
            print("p_value rmse")
            _, p_value_bias = ttest_ind(df_station[f"nwp_{metric_2}"].values, df_station[metric_2].values,
                                        nan_policy='omit')
            print(p_value_bias)

            if mae_nwp > mae_cnn and p_value_mae < 0.05:
                improvments_mae += 1
            elif mae_nwp < mae_cnn and p_value_mae < 0.05:
                degradation_mae += 1
            else:
                no_improvment_mae += 1

            if rmse_nwp > rmse_cnn and p_value_mae < 0.05:
                improvments_rmse += 1
            elif rmse_nwp < rmse_cnn and p_value_rmse < 0.05:
                degradation_rmse += 1
            else:
                no_improvment_rmse += 1

            if np.abs(bias_nwp) > np.abs(bias_cnn) and p_value_bias < 0.05:
                improvments_bias += 1
                if df_station["alti"].iloc[0] > 1500:
                    above_2000m += 1
            elif np.abs(bias_nwp) < np.abs(bias_cnn) and p_value_bias < 0.05:
                degradation_bias += 1
            else:
                no_improvment_bias += 1

            print("\nNb improvments mae")
            print(improvments_mae)
            print("Nb degradation mae")
            print(degradation_mae)
            print("No improvment mae")
            print(no_improvment_mae)

            print("\nNb improvments RMSE")
            print(improvments_rmse)
            print("Nb degradation RMSE")
            print(degradation_rmse)
            print("No improvment RMSE")
            print(no_improvment_rmse)

            print("\nNb improvments Bias")
            print(improvments_bias)
            print("Nb degradation Bias")
            print(degradation_bias)
            print("No improvment Bias")
            print(no_improvment_bias)
            print("Improvment for stations above 1500m")
            print(above_2000m)
            print("Number of station above 1500m")
            print(station_above_2000m)

        print("\n\nNb stations")
        print(len(metric_result["name"].unique()))

        print("\n\n\n\n______________________")
        print("______________________")
        print("Global results")
        print("______________________")
        print("______________________\n\n\n\n")
        print("MAE NWP")
        print(metric_result[f"nwp_{metric}"].mean())
        print("MAE CNN")
        print(metric_result[metric].mean())
        print("\nRMSE NWP")
        print(np.sqrt(np.mean(metric_result[f"nwp_{metric}"].values ** 2)))
        print("RMSE CNN")
        print(np.sqrt(np.mean(metric_result[metric].values ** 2)))
        print("\nBias NWP")
        print(metric_result[f"nwp_{metric_2}"].mean())
        print("Bias CNN")
        print(metric_result[metric_2].mean())
        print("_______________________________________________________________________________________________________")
        print("_______________________________________________________________________________________________________")

    def plot_metric_all_stations(self, results, variables=["UV"], metrics=["abs_error"], sort_by="alti",
                                 variable_qc="vw10m(m/s)", use_QC=False,
                                 time_series_qc=None, showfliers=False, prm=None):

        gt = GaussianTopo()

        # Create DataFrame
        if variables == ["UV_DIR_deg"]:
            variables_to_extract = ["UV_DIR_deg", "UV"]
        else:
            variables_to_extract = variables

        # Keep only common rows between observation and model
        results = self.intersection_model_obs_on_results(results, variables=variables_to_extract,
                                                         use_QC=use_QC, time_series_qc=time_series_qc,
                                                         variable_qc=variable_qc)

        cnn, nwp, obs = [self.create_df_from_dict(results, data_type=data_type, variables=variables_to_extract)
                         for data_type in ["cnn", "nwp", "obs"]]

        cnn["hour"] = cnn.index.hour
        cnn["month"] = cnn.index.month

        for variable in variables:

            for metric in metrics:

                metric_result = cnn.copy()
                metric_func = self._select_metric(metric)

                if metric == "bias_rel_wind_1":
                    condition_speed = (obs[variable] >= 1).values
                    metric_result = metric_result[condition_speed]
                    nwp = nwp[condition_speed]
                    cnn = cnn[condition_speed]
                    obs = obs[condition_speed]

                if 'direction' in metric:
                    obs, nwp, cnn, metric_result = self.prepare_direction_data_for_evaluation(obs, nwp, cnn,
                                                                                              metric_result,
                                                                                              variable=variable)
                # Result CNN
                metric_result[metric] = metric_func(cnn[variable].values, obs[variable].values)
                # NWP variable
                metric_result[f"nwp_{variable}"] = nwp[variable].values
                # Result NWP
                metric_result[f"nwp_{metric}"] = metric_func(nwp[variable].values, obs[variable].values)
                metric_result = self.update_df_with_topo_carac(metric_result)

                # Boxplot
                for y in ["name", "alti", "hour", "month", "tpi_2000", "tpi_500", "laplacian", "mu", "curvature"]:
                    sort_by_i = sort_by if y == "name" else None
                    self._plot_boxplot(data=metric_result, x=metric, y=y, sort_by=sort_by_i)
                    ax = plt.gca()
                    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
                    save_figure(f"boxplot_{y}", prm)

                # Boxplot with two models
                self._plot_two_boxplot(data=metric_result, metric=metric, figsize=(20, 20))
                ax = plt.gca()
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
                save_figure(f"cnn_nwp_boxplot_station", prm)

                # Boxplot by altitude category
                metric_result = self._group_metric_result_by_alti_category(metric_result)
                metric_result = gt.classify(metric_result, "tpi_500", True)
                metric_result = gt.classify(metric_result, "mu", True)
                metric_result = gt.classify(metric_result, "laplacian", True)
                metric_result = gt.classify_tpi(metric_result)
                order = ["Station altitude <= 1000m", "1000m < Station altitude <= 1700m",
                         "1700m < Station altitude <= 2500m", "2500m < Station altitude"]
                print("Result by group alti")
                if "bias" in metric_result.columns and "nwp_bias" in metric_result.columns:
                    print(metric_result.groupby("group_alti")[["bias", "nwp_bias"]].mean())
                self._plot_two_boxplot(data=metric_result, metric=metric, y="group_alti", order=order,
                                       showfliers=showfliers, figsize=(20, 20))
                plt.tight_layout()
                ax = plt.gca()
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
                save_figure(f"cnn_nwp_boxplot", prm)

                """
                # Figure grouped by class of altitude and class of metric
                # Not used
                for group_alti in order:
                    df_1 = metric_result[metric_result["group_alti"] == group_alti]
                    for parameter in ["tpi_500", "mu", "laplacian"]:
                        self._plot_two_boxplot(data=df_1,
                                               metric=metric, y=f"class_{parameter}", order=order,
                                               showfliers=showfliers, figsize=(20, 20))
                        plt.tight_layout()
                        ax = plt.gca()
                        ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
                        save_figure(f"alti_and_parameter", prm)
                """

                """
                # Boxplot for wind categories
                metric_result = self._group_metric_result_by_wind_category(metric_result, variable)
                self._plot_boxplot(data=metric_result, x=metric, y=f"group_{variable}_nwp")
                ax = plt.gca()
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
                save_figure(f"cnn_nwp_boxplot_wind_category", prm)
                """

                # Two elevation categories
                df_low = metric_result[metric_result["group_alti"].isin(["Station altitude <= 1000m", "1000m < Station altitude <= 1700m"])]
                df_high = metric_result[metric_result["group_alti"].isin(["1700m < Station altitude <= 2500m", "2500m < Station altitude"])]

                for df, altitude in zip([metric_result, df_low, df_high], ["all", "low", "high"]):
                    for y in ["tpi_500"]:
                        print("Results by class TPI 500")
                        print(metric_result.columns)
                        print(metric_result.groupby("class_tpi_500")[["bias", "nwp_bias"]].mean())
                        self._plot_two_boxplot(data=df,
                                               metric=metric,
                                               y=f"class_{y}",
                                               showfliers=showfliers,
                                               order=["$TPI \leq q_{25}$", "$q_{25}<TPI \leq q_{50}$", "$q_{50}<TPI \leq q_{75}$", "$q_{75}<TPI$"])
                        ax = plt.gca()
                        ax.set_yticklabels(ax.get_yticklabels(), fontsize=50)
                        plt.xticks(fontsize=50)
                        plt.tight_layout()
                        save_figure(f"alti_{altitude}_{y}", prm)

    def plot_accelerations(self, results, variable="UV", sort_by="alti", showfliers=False):

        # Create DataFrame
        cnn = self.create_df_from_dict(results, data_type="cnn", variables=[variable])
        nwp = self.create_df_from_dict(results, data_type="nwp", variables=[variable])
        cnn["hour"] = cnn.index.hour
        cnn["month"] = cnn.index.month

        acceleration = cnn.copy()
        acceleration[variable] = cnn[variable] / nwp[variable]
        acceleration = acceleration[nwp[variable] >= 1]

        acceleration = self.update_df_with_topo_carac(acceleration)

        """
        # Boxplot
        for y in ["name", "alti", "hour", "month", "tpi_2000", "tpi_500", "laplacian", "mu", "curvature"]:
            sort_by_i = sort_by if y == "name" else None
            self._plot_boxplot(data=acceleration, x=variable, y=y, sort_by=sort_by_i, showfliers=showfliers)
            ax = plt.gca()
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

        # Boxplot with two models
        self._plot_two_boxplot(data=metric_result, metric=metric)
        ax = plt.gca()
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

        # Boxplot for wind categories
        metric_result = self._group_metric_result_by_wind_category(metric_result, variable)
        self._plot_boxplot(data=metric_result, x=metric, y=f"group_{variable}_nwp")
        ax = plt.gca()
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

        # Metric = f(Acceleration)
        UV_cnn = metric_result[variable].values
        nwp_UV_cnn = metric_result[f"nwp_{variable}"].values
        metric_result["acceleration"] = np.where(nwp_UV_cnn > 0, UV_cnn/nwp_UV_cnn, 1)
        self._plot_classic(data=metric_result, x="acceleration", y=metric)

        # 1-1 plot metric CNN vs metric NWP
        bins = (100, 100)
        cnn_metric = metric_result[metric].values
        nwp_metric = metric_result[f"nwp_{metric}"].values
        self.density_scatter(nwp_metric, cnn_metric, s=5, bins=bins, cmap=cmap, use_power_norm=2)
        max = np.nanmax([nwp_metric, cnn_metric])
        min = np.nanmin([nwp_metric, cnn_metric])
        plt.xlim((min - 2, max + 2))
        plt.ylim((min - 2, max + 2))
        plt.plot(nwp_metric, nwp_metric, color='red')
        plt.axis('square')
        plt.xlabel(f"{metric} NWP")
        plt.ylabel(f"{metric} CNN")
        """

    def plot_metric_distribution(self, results, variables=["Wind_DIR"], metrics=["bias_direction"], use_QC=True,
                                 time_series_qc=None, prm=None):
        # Keep only common rows between observation and model
        results = self.intersection_model_obs_on_results(results, variables=variables,
                                                         use_QC=use_QC, time_series_qc=time_series_qc)

        # Create DataFrame
        cnn, nwp, obs = [self.create_df_from_dict(results, data_type=data_type, variables=variables)
                         for data_type in ["cnn", "nwp", "obs"]]
        cnn["hour"] = cnn.index.hour
        cnn["month"] = cnn.index.month

        for variable in variables:

            for metric in metrics:

                metric_result = cnn.copy()
                metric_func = self._select_metric(metric)

                if metric == "bias_rel_wind_1":
                    metric_result = metric_result[obs[variable] >= 1]
                    nwp = nwp[obs[variable] >= 1]
                    cnn = cnn[obs[variable] >= 1]
                    obs = obs[obs[variable] >= 1]

                # Result CNN
                metric_result[metric] = metric_func(cnn[variable].values, obs[variable].values)
                # NWP variable
                metric_result[f"nwp_{variable}"] = nwp[variable].values
                # Result NWP
                metric_result[f"nwp_{metric}"] = metric_func(nwp[variable].values, obs[variable].values)
                metric_result = self.update_df_with_topo_carac(metric_result)

                all_df0 = pd.DataFrame()
                all_df0[f"{metric}"] = metric_result[metric]
                all_df0["name"] = "CNN"
                all_df1 = pd.DataFrame()
                all_df1[f"{metric}"] = metric_result[f"nwp_{variable}"]
                all_df1["name"] = "NWP"
                all_df = pd.concat([all_df0, all_df1])
                all_df.index = list(range(len(all_df)))

                plt.figure()
                sns.displot(all_df.sample(10_000), x=f"{metric}", hue="name", kind="kde", fill=True, bw_adjust=3, cut=0)
                save_figure(f"distribution_{metric}", prm)

    def plot_accelerations_exposed_distributions(self, results):
        # Create DataFrame
        cnn = self.create_df_from_dict(results, data_type="cnn", variables=['NWP_wind_speed', 'exp_Wind'])
        cnn["hour"] = cnn.index.hour
        cnn["month"] = cnn.index.month

        acceleration = cnn.copy()
        acceleration["acceleration_exposed"] = cnn['exp_Wind'] / cnn['NWP_wind_speed']
        acceleration = acceleration[cnn['NWP_wind_speed'] >= 1]

        acceleration = self.update_df_with_topo_carac(acceleration)
        sns.displot(data=acceleration, x="acceleration_exposed", kind='kde', fill=True, bw_adjust=3, cut=0)

    def plot_distribution_all_stations(self, results, variables=["UV"], models=["cnn", "nwp", "obs"],
                                       use_QC=False, time_series_qc=False):
        """Wind speed distribution (one distribution for all stations)"""
        results = self.intersection_model_obs_on_results(results, variables=variables,
                                                         use_QC=use_QC, time_series_qc=time_series_qc)
        all_df = []
        for variable in variables:
            for model in models:
                data = [[station, data[0]]
                        for station, df_data in results[variable][model].items()
                        for data in df_data.values]
                df = pd.DataFrame(data, columns=["station", variable])
                df["data"] = model
                all_df.append(df)
            all_df = pd.concat(all_df)
            sns.displot(data=all_df, x=variable, hue="data", kind='kde', fill=True, bw_adjust=3, cut=0)

    def plot_1_1_density(self, results, cmap="viridis", variables=["UV"],
                         xlim_min=None, xlim_max=None, ylim_min=None, ylim_max=None,
                         use_QC=False, time_series_qc=None, prm=None):
        """
        1-1 plots

        NWP-CNN, CNN-obs, NWP-obs
        """
        results = self.intersection_model_obs_on_results(results, variables=variables,
                                                         use_QC=use_QC, time_series_qc=time_series_qc)

        for variable in variables:
            all_df = []
            for model in ["cnn", "nwp", "obs"]:
                data = [[station, data[0]]
                        for station, df_data in results[variable][model].items()
                        for data in df_data.values]
                df = pd.DataFrame(data, columns=["station", variable])
                df["data"] = model
                all_df.append(df)
            all_df = pd.concat(all_df)

            cnn_values = all_df[variable][all_df["data"] == "cnn"].values
            nwp_values = all_df[variable][all_df["data"] == "nwp"].values
            obs_values = all_df[variable][all_df["data"] == "obs"].values

            # NWP vs CNN
            bins = (100, 100)
            ax = self.v.density_scatter(nwp_values, cnn_values, s=5, bins=bins, cmap=cmap)
            max = np.nanmax([nwp_values, cnn_values])
            min = np.nanmin([nwp_values, cnn_values])
            plt.plot(nwp_values, nwp_values, color='red')
            if xlim_min is None and xlim_max is None:
                ax.set_xlim((min - 2, max + 2))
            else:
                ax.set_xlim((xlim_min, xlim_max))
            if ylim_min is None and ylim_max is None:
                ax.set_ylim((min - 2, max + 2))
            else:
                ax.set_ylim((ylim_min, ylim_max))
            save_figure(f"density_{variable}_NWP_CNN", prm)

            # CNN vs obs
            ax = self.v.density_scatter(obs_values, cnn_values, s=5, bins=bins, cmap=cmap)
            max = np.nanmax([obs_values, cnn_values])
            min = np.nanmin([obs_values, cnn_values])
            plt.plot(obs_values, obs_values, color='red')
            if xlim_min is None and xlim_max is None:
                ax.set_xlim((min - 2, max + 2))
            else:
                ax.set_xlim((xlim_min, xlim_max))
            if ylim_min is None and ylim_max is None:
                ax.set_ylim((min - 2, max + 2))
            else:
                ax.set_ylim((ylim_min, ylim_max))
            save_figure(f"density_{variable}_obs_CNN", prm)

            # NWP vs obs
            ax = self.v.density_scatter(obs_values, nwp_values, s=5, bins=bins, cmap=cmap)
            max = np.nanmax([obs_values, nwp_values])
            min = np.nanmin([obs_values, nwp_values])
            plt.plot(obs_values, obs_values, color='red')
            if xlim_min is None and xlim_max is None:
                ax.set_xlim((min - 2, max + 2))
            else:
                ax.set_xlim((xlim_min, xlim_max))
            if ylim_min is None and ylim_max is None:
                ax.set_ylim((min - 2, max + 2))
            else:
                ax.set_ylim((ylim_min, ylim_max))
            save_figure(f"density_{variable}_obs_NWP", prm)

    def plot_1_1_density_by_station(self, results, cmap="viridis", variables=["UV"],
                                    xlim_min=None, xlim_max=None, ylim_min=None, ylim_max=None,
                                    use_QC=False, time_series_qc=None, prm=None):
        """
        1-1 plots

        NWP-CNN, CNN-obs, NWP-obs
        """
        results = self.intersection_model_obs_on_results(results, variables=variables,
                                                         use_QC=use_QC, time_series_qc=time_series_qc)

        for variable in variables:
            all_df = []
            for model in ["cnn", "nwp", "obs"]:
                data = [[station, data[0]]
                        for station, df_data in results[variable][model].items()
                        for data in df_data.values]
                df = pd.DataFrame(data, columns=["station", variable])
                df["data"] = model
                all_df.append(df)
            all_df = pd.concat(all_df)

            for station in all_df["station"].unique():
                filter_station = all_df["station"] == station
                cnn_values = all_df[variable][filter_station & (all_df["data"] == "cnn")].values
                nwp_values = all_df[variable][filter_station & (all_df["data"] == "nwp")].values
                obs_values = all_df[variable][filter_station & (all_df["data"] == "obs")].values

                fig = plt.figure(figsize=(45, 15))

                # NWP vs CNN
                plt.subplot(131)
                ax = plt.gca()
                bins = (100, 100)
                ax = self.v.density_scatter(nwp_values, cnn_values, s=10, bins=bins, cmap=cmap, ax=ax, colorbar=False)
                max = np.nanmax([nwp_values, cnn_values])
                min = np.nanmin([nwp_values, cnn_values])
                plt.plot(nwp_values, nwp_values, color='red')
                ax = plt.gca()
                if xlim_min is None and xlim_max is None:
                    ax.set_xlim((min - 2, max + 2))
                else:
                    ax.set_xlim((xlim_min, xlim_max))
                if ylim_min is None and ylim_max is None:
                    ax.set_ylim((min - 2, max + 2))
                else:
                    ax.set_ylim((ylim_min, ylim_max))
                plt.axis("equal")

                # CNN vs obs
                plt.subplot(132)
                ax = plt.gca()
                ax = self.v.density_scatter(obs_values, cnn_values, s=10, bins=bins, cmap=cmap, ax=ax, colorbar=False)
                plt.plot(obs_values, obs_values, color='red')
                ax = plt.gca()
                if xlim_min is None and xlim_max is None:
                    ax.set_xlim((min - 2, max + 2))
                else:
                    ax.set_xlim((xlim_min, xlim_max))
                if ylim_min is None and ylim_max is None:
                    ax.set_ylim((min - 2, max + 2))
                else:
                    ax.set_ylim((ylim_min, ylim_max))
                plt.axis("equal")

                # NWP vs obs
                plt.subplot(133)
                ax = plt.gca()
                ax = self.v.density_scatter(obs_values, nwp_values, s=10, bins=bins, cmap=cmap, ax=ax, colorbar=False)
                plt.plot(obs_values, obs_values, color='red')
                ax = plt.gca()
                if xlim_min is None and xlim_max is None:
                    ax.set_xlim((min - 2, max + 2))
                else:
                    ax.set_xlim((xlim_min, xlim_max))
                if ylim_min is None and ylim_max is None:
                    ax.set_ylim((min - 2, max + 2))
                else:
                    ax.set_ylim((ylim_min, ylim_max))
                plt.axis("equal")

                uuid_str = str(uuid.uuid4())[:4]
                fig.savefig(prm["save_figure_path"] + f"density_{station}_{uuid_str}.png")
                fig.savefig(prm["save_figure_path"] + f"density_{station}_{uuid_str}.svg")

    def plot_quantile_quantile(self, results, variables=["UV"], use_QC=False, time_series_qc=None, metric=None,
                               prm=None):
        """
        1-1 plots

        NWP-CNN, CNN-obs, NWP-obs
        """
        results = self.intersection_model_obs_on_results(results, variables=variables,
                                                         use_QC=use_QC, time_series_qc=time_series_qc)

        for variable in variables:
            all_df = []
            for model in ["cnn", "nwp", "obs"]:
                data = [[station, data[0]]
                        for station, df_data in results[variable][model].items()
                        for data in df_data.values]
                df = pd.DataFrame(data, columns=["station", variable])
                df["data"] = model
                all_df.append(df)
            all_df = pd.concat(all_df)

            cnn_values = all_df[variable][all_df["data"] == "cnn"].values
            nwp_values = all_df[variable][all_df["data"] == "nwp"].values
            obs_values = all_df[variable][all_df["data"] == "obs"].values

            # QQ plot
            # percs = np.linspace(0, 100, 11)
            percs = np.round(np.linspace(0, 100, 10000), 2)
            qn_obs = np.percentile(obs_values, percs)
            qn_nwp = np.percentile(nwp_values, percs)
            qn_cnn = np.percentile(cnn_values, percs)
            x = np.linspace(np.min((qn_obs.min(), qn_obs.min())), np.max((qn_obs.max(), qn_obs.max())))

            plt.figure()
            plt.plot(qn_obs, qn_cnn, linestyle="-", marker="x", markersize=5, color="C0")
            plt.plot(qn_obs, qn_nwp, linestyle="-", marker="+", markersize=5, color="C1")
            plt.plot(x, x, ls="--", color='red')
            plt.legend(("DEVINE", "AROME"))
            plt.xlabel("Observed wind speed [m/s]")
            plt.ylabel("Modeled wind speed [m/s]")
            plt.xlim(-1, 45)
            plt.ylim(-1, 45)
            plt.axis("square")
            plt.tight_layout()
            save_figure(f"QQ_plot", prm)

            plt.figure(figsize=(20, 20))
            plt.plot(qn_obs, qn_cnn, linestyle="", marker="x", color="C0")
            plt.plot(qn_obs, qn_nwp, linestyle="", marker="+", color="C1")
            plt.plot(x, x, ls="--", color='red')
            plt.legend(("DEVINE", "AROME"))
            plt.xlabel("Observed wind speed [m/s]")
            plt.ylabel("Modeled wind speed [m/s]")
            plt.xlim(-1, 15)
            plt.ylim(-1, 15)
            plt.tight_layout()
            save_figure(f"zoom_QQ_plot", prm)

    def plot_heatmap(self, results, variables=["UV"], metrics=["abs_error"], topo_carac="alti", fontsize=15,
                     sort_by="alti", plot_classic=False, cmap="viridis",
                     use_QC=False, time_series_qc=None, min_speed=1, vmax=0.2, vmax_diff=0.03, prm=None):

        # Keep only common rows between observation and model
        results = self.intersection_model_obs_on_results(results,
                                                         use_QC=use_QC, time_series_qc=time_series_qc)

        cnn = self.create_df_from_dict(results, data_type="cnn", variables=variables)
        nwp = self.create_df_from_dict(results, data_type="nwp", variables=variables)
        obs = self.create_df_from_dict(results, data_type="obs", variables=variables)

        cnn["hour"] = cnn.index.hour
        cnn["month"] = cnn.index.month

        for variable in variables:

            for metric in metrics:

                metric_result = cnn.copy()

                if metric == "abs_error":
                    metric_func = self.absolute_error
                if metric == "bias":
                    metric_func = self.bias
                if metric == "abs_error_rel":
                    metric_func = self.absolute_error_relative
                if metric == "bias_rel":
                    metric_func = self.bias_rel
                if metric == "bias_rel_wind_1":
                    metric_func = self.bias_rel_wind_1
                    metric_result = metric_result[obs[variable] >= min_speed]
                    nwp = nwp[obs[variable] >= min_speed]
                    cnn = cnn[obs[variable] >= min_speed]
                    obs = obs[obs[variable] >= min_speed]
                if metric == "abs_error_rel_wind_1":
                    metric_func = self.abs_error_rel_wind_1
                    metric_result = metric_result[obs[variable] >= min_speed]
                    nwp = nwp[obs[variable] >= min_speed]
                    cnn = cnn[obs[variable] >= min_speed]
                    obs = obs[obs[variable] >= min_speed]

                metric_result[metric] = metric_func(cnn[variable].values, obs[variable].values, min_speed=min_speed)
                metric_result[f"nwp_{variable}"] = nwp[variable].values

                metric_result[f"nwp_{metric}"] = metric_func(nwp[variable].values, obs[variable].values,
                                                             min_speed=min_speed)
                metric_result = self.update_df_with_topo_carac(metric_result)

                metric_result[f"diff_cnn_nwp_{metric}"] = np.abs(metric_result[metric]) - np.abs(
                    metric_result[f"nwp_{metric}"])

                # Heatmap month vs hour
                piv_table_cnn = metric_result.pivot_table(index="month", columns="hour", values=metric, aggfunc='mean')
                plt.figure()
                ax = plt.gca()
                sns.heatmap(piv_table_cnn, ax=ax, cmap='coolwarm',
                            norm=MidpointNormalize(midpoint=0, vmin=-vmax, vmax=vmax))
                plt.tight_layout()
                save_figure(f"degree", prm)

                # Heatmap month vs hour for NWP
                piv_table_nwp = metric_result.pivot_table(index="month", columns="hour",
                                                          values=f"nwp_{metric}", aggfunc='mean')
                plt.figure()
                ax = plt.gca()
                sns.heatmap(piv_table_nwp, ax=ax, cmap='coolwarm',
                            norm=MidpointNormalize(midpoint=0, vmin=-vmax, vmax=vmax))
                plt.tight_layout()
                save_figure(f"degree_heatmap_NWP", prm)

                # Heatmap month vs hour for difference CNN NWP
                piv_table_diff = np.abs(piv_table_cnn) - np.abs(piv_table_nwp)
                plt.figure()
                ax = plt.gca()
                sns.heatmap(piv_table_diff, ax=ax, cmap='BrBG_r',
                            norm=MidpointNormalize(midpoint=0, vmin=-vmax_diff, vmax=vmax_diff))
                plt.tight_layout()
                save_figure(f"degree_heatmap_diff_CNN_NWP", prm)

                """
                # Heatmap laplacian vs mu
                piv_table = metric_result.pivot_table(index="laplacian", columns="mu", values=metric,
                                                      aggfunc='mean')
                plt.figure()
                ax = plt.gca()
                sns.heatmap(piv_table, cmap="viridis", ax=ax)
                plt.title(f"{metric}: laplacian vs mu")

                # Heatmap laplacian vs mu for NWP
                piv_table = metric_result.pivot_table(index="laplacian", columns="mu", values=f"nwp_{metric}",
                                                      aggfunc='mean')
                plt.figure()
                ax = plt.gca()
                sns.heatmap(piv_table, cmap="viridis", ax=ax)
                plt.title(f"{metric}: laplacian vs mu for NWP")

                # Heatmap laplacian vs mu for difference CNN NWP
                piv_table = metric_result.pivot_table(index="laplacian", columns="mu", values=f"diff_cnn_nwp_{metric}", aggfunc='mean')
                plt.figure()
                ax = plt.gca()
                sns.heatmap(piv_table, cmap="viridis", ax=ax)
                plt.title(f"{metric}: laplacian vs mu for difference CNN NWP")

                # Heatmap tpi vs altitude
                piv_table = metric_result.pivot_table(index="tpi_500", columns="alti", values=metric,
                                                      aggfunc='mean')
                plt.figure()
                ax = plt.gca()
                sns.heatmap(piv_table, cmap="viridis", ax=ax)
                plt.title(f"{metric}: tpi vs altitude")

                # Heatmap tpi vs altitude for NWP
                piv_table = metric_result.pivot_table(index="tpi_500", columns="alti", values=f"nwp_{metric}",
                                                      aggfunc='mean')
                plt.figure()
                ax = plt.gca()
                sns.heatmap(piv_table, cmap="viridis", ax=ax)
                plt.title(f"{metric}: tpi vs altitude for NWP")

                # Heatmap tpi vs altitude for difference CNN NWP
                piv_table = metric_result.pivot_table(index="tpi_500", columns="alti", values=f"diff_cnn_nwp_{metric}", aggfunc='mean')
                plt.figure()
                ax = plt.gca()
                sns.heatmap(piv_table, cmap="viridis", ax=ax)
                plt.title(f"{metric}: tpi vs altitude for difference CNN NWP")
                """

    def plot_heatmap_by_wind_category(self, results, variables=["UV"], metrics=["abs_error"], min_speed=1, max_speed=3,
                                      use_QC=False, time_series_qc=None, vmax=0.2, vmax_diff=0.03, prm=None):

        # Keep only common rows between observation and model
        results = self.intersection_model_obs_on_results(results,
                                                         use_QC=use_QC, time_series_qc=time_series_qc)

        cnn = self.create_df_from_dict(results, data_type="cnn", variables=variables)
        nwp = self.create_df_from_dict(results, data_type="nwp", variables=variables)
        obs = self.create_df_from_dict(results, data_type="obs", variables=variables)

        cnn["hour"] = cnn.index.hour
        cnn["month"] = cnn.index.month

        for variable in variables:

            for metric in metrics:

                metric_result = cnn.copy()
                condition_min = (obs[variable] >= min_speed).values
                condition_max = (obs[variable] <= max_speed).values
                metric_result = metric_result[condition_min & condition_max]
                nwp = nwp[condition_min & condition_max]
                cnn = cnn[condition_min & condition_max]
                obs = obs[condition_min & condition_max]
                if metric == "abs_error":
                    metric_func = self.absolute_error
                if metric == "bias":
                    metric_func = self.bias
                if metric == "abs_error_rel":
                    metric_func = self.absolute_error_relative
                if metric == "bias_rel":
                    metric_func = self.bias_rel
                if metric == "bias_rel_wind_1":
                    metric_func = self.bias_rel_wind_1
                if metric == "abs_error_rel_wind_1":
                    metric_func = self.abs_error_rel_wind_1

                metric_result[metric] = metric_func(cnn[variable].values, obs[variable].values, min_speed=min_speed)
                metric_result[f"nwp_{variable}"] = nwp[variable].values

                metric_result[f"nwp_{metric}"] = metric_func(nwp[variable].values, obs[variable].values,
                                                             min_speed=min_speed)

                # Heatmap month vs hour
                piv_table_cnn = metric_result.pivot_table(index="month", columns="hour", values=metric, aggfunc='mean')
                piv_table_nwp = metric_result.pivot_table(index="month", columns="hour", values=f"nwp_{metric}",
                                                          aggfunc='mean')
                piv_table_diff = np.abs(piv_table_cnn) - np.abs(piv_table_nwp)
                plt.figure()
                ax = plt.gca()
                sns.heatmap(piv_table_diff, ax=ax, cmap='BrBG_r',
                            norm=MidpointNormalize(midpoint=0, vmin=-vmax_diff, vmax=vmax_diff))
                plt.tight_layout()
                save_figure(f"min_speed={min_speed}_max_speed={max_speed}_heatmap_diff_CNN_NWP_", prm)

    @staticmethod
    def _plot_classic(data=None, x=None, y=None, fontsize=15):
        plt.figure()
        sns.scatterplot(data=data, x=x, y=y)
        plt.ylabel(y, fontsize=fontsize)
        plt.xlabel(x, fontsize=fontsize)

    @staticmethod
    def _plot_boxplot(data=None, sort_by="alti", y="name", x="absolute error", showfliers=False):
        plt.figure()
        list_ordered = list(data.sort_values(by=[sort_by])[y].unique()) if sort_by is not None else None
        sns.boxplot(data=data, y=y, x=x, orient="h", showfliers=showfliers, order=list_ordered)
        sns.despine(trim=True, left=True)

    @staticmethod
    def _plot_two_boxplot(data=None, metric=None, y="name", fontsize=15, figsize=(20, 20), order=None,
                          showfliers=False):
        plt.figure(figsize=figsize)
        ax = plt.gca()
        data = data.melt(id_vars=[y], var_name='dataset', value_name='values')
        data = data[data["dataset"].isin([metric, f"nwp_{metric}"])]
        sns.boxplot(data=data, y=y, x="values", hue="dataset", orient="h",
                    showfliers=showfliers, palette=["C0", "C1"], ax=ax, order=order)
        ax.legend().remove()
        ax.set(ylabel="")
        ax.set(xlabel="")
        ax.xaxis.grid(True)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", size=20)
        sns.despine(ax=ax, trim=True, left=True)

    def results_to_three_arrays(self, results, variable="UV", use_QC=False, time_series_qc=None):
        """
        Return nwp, cnn and obs arrays from resuts

        Parameters
        ----------
        results : dict
        variable : str
        use_QC : boolean
        time_series_qc : pd.DataFrame

        Returns
        -------
        cnn_values : ndarray
        nwp_values : ndarray
        obs_values : ndarray
        """
        results = self.intersection_model_obs_on_results(results, variables=[variable],
                                                         use_QC=use_QC, time_series_qc=time_series_qc)

        all_df = []
        for model in ["cnn", "nwp", "obs"]:
            data = [[station, data[0]]
                    for station, df_data in results[variable][model].items()
                    for data in df_data.values]
            df = pd.DataFrame(data, columns=["station", variable])
            df["data"] = model
            all_df.append(df)
        all_df = pd.concat(all_df)

        cnn_values = all_df[variable][all_df["data"] == "cnn"].values
        nwp_values = all_df[variable][all_df["data"] == "nwp"].values
        obs_values = all_df[variable][all_df["data"] == "obs"].values

        return cnn_values, nwp_values, obs_values
