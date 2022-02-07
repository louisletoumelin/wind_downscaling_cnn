import uuid

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ModuleNotFoundError:
    pass

from downscale.visu.MidpointNormalize import MidpointNormalize
from downscale.eval.evaluation import Evaluation
from downscale.utils.utils_func import save_figure


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
            for station in results[variable]["cnn"].keys():
                nwp, cnn, obs = [self._drop_nans_and_duplicates_at_station_results(results[variable][data_key][station])
                                 for data_key in ["nwp", "cnn", "obs"]]

                # Keep index intersection
                index_intersection = obs.index.intersection(nwp.index).intersection(cnn.index)
                if use_QC:
                    qc_filter_station = time_series_qc["name"] == station
                    qc = self._drop_nans_and_duplicates_at_station_results(time_series_qc[variable_qc][qc_filter_station])
                    index_intersection = index_intersection.intersection(qc.index)

                obs, nwp, cnn, qc = [df[df.index.isin(index_intersection)] for df in [obs, nwp, cnn, qc]]

                results[variable]["nwp"][station] = nwp
                results[variable]["cnn"][station] = cnn
                if use_QC:
                    results[variable]["obs"][station] = qc
                else:
                    results[variable]["obs"][station] = obs

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
        for variable in variables:
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

            if len(result_df) != 0:
                try:
                    assert len(result_df_var_i) == len(result_df)
                except AssertionError:
                    print(len(result_df_var_i))
                    print(len(result_df))
                    print(result_df)
                    print(result_df_var_i)
                    raise

            result_df[variable] = result_df_var_i[variable]

            if "name" not in result_df.columns:
                result_df["name"] = result_df_var_i["name"]

        #result_df = pd.concat(result_df, axis=1)

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

    def plot_metric_all_stations(self, results, variables=["UV"], metrics=["abs_error"], sort_by="alti", use_QC=False,
                                 time_series_qc=None, showfliers=False, prm=None):

        # Keep only common rows between observation and model
        results = self.intersection_model_obs_on_results(results, variables=variables,
                                                         use_QC=use_QC, time_series_qc=time_series_qc)

        # Create DataFrame
        if variables == ["UV_DIR_deg"]:
            variables_to_extract = ["UV_DIR_deg", "UV"]
        else:
            variables_to_extract = variables

        cnn, nwp, obs = [self.create_df_from_dict(results, data_type=data_type, variables=variables_to_extract)
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

                if metric == "bias_direction":
                    condition_direction = (obs["UV"] > 0) & (nwp["UV"] > 0) & (cnn["UV"] > 0)
                    metric_result = metric_result[condition_direction]
                    nwp = nwp[condition_direction]
                    cnn = cnn[condition_direction]
                    obs = obs[condition_direction]

                    nwp[nwp[variable] == 0] = 360
                    cnn[cnn[variable] == 0] = 360

                # Result CNN
                metric_result[metric] = metric_func(cnn[variable].values, obs[variable].values)
                # NWP variable
                metric_result[f"nwp_{variable}"] = nwp[variable].values
                # Result NWP
                metric_result[f"nwp_{metric}"] = metric_func(nwp[variable].values, obs[variable].values)
                metric_result = self.update_df_with_topo_carac(metric_result)

                """
                # Classic
                if plot_classic:
                    self._plot_classic(data=metric_result, x=topo_carac, y=metric, fontsize=fontsize)
                
                """
                # Boxplot
                for y in ["name", "alti", "hour", "month", "tpi_2000", "tpi_500", "laplacian", "mu", "curvature"]:
                    sort_by_i = sort_by if y == "name" else None
                    self._plot_boxplot(data=metric_result, x=metric, y=y, sort_by=sort_by_i)
                    ax = plt.gca()
                    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
                    save_figure(f"boxplot_{y}", prm)

                # Boxplot with two models
                print("metric NWP")
                print(metric_result[f"nwp_{metric}"].describe())
                print("metric CNN")
                print(metric_result[metric].describe())

                print("abs metric NWP")
                print(metric_result[f"nwp_{metric}"].apply(np.abs).describe())
                print("metric CNN")
                print(metric_result[metric].apply(np.abs).describe())

                self._plot_two_boxplot(data=metric_result, metric=metric, figsize=(20, 20))
                ax = plt.gca()
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
                save_figure(f"cnn_nwp_boxplot_station", prm)

                # Boxplot by altitude category
                metric_result = self._group_metric_result_by_alti_category(metric_result)
                order = ["Station altitude <= 1000m", "1000m < Station altitude <= 1700m",
                         "1700m < Station altitude <= 2500m", "2500m < Station altitude"]
                self._plot_two_boxplot(data=metric_result, metric=metric, y="group_alti", order=order,
                                       showfliers=showfliers, figsize=(20, 20))
                plt.tight_layout()
                ax = plt.gca()
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
                save_figure(f"cnn_nwp_boxplot", prm)

                # Boxplot for wind categories
                metric_result = self._group_metric_result_by_wind_category(metric_result, variable)
                self._plot_boxplot(data=metric_result, x=metric, y=f"group_{variable}_nwp")
                ax = plt.gca()
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
                save_figure(f"cnn_nwp_boxplot_wind_category", prm)


                """
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

    def plot_quantile_quantile(self, results,  variables=["UV"], use_QC=False, time_series_qc=None, metric=None,
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
                condition_min = (obs[variable] >= min_speed)
                condition_max = (obs[variable] <= max_speed)
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
                piv_table_nwp = metric_result.pivot_table(index="month", columns="hour", values=f"nwp_{metric}", aggfunc='mean')
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
    def _plot_two_boxplot(data=None, metric=None, y="name", fontsize=15, figsize=(20, 20), order=None, showfliers=False):
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
