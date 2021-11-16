import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ModuleNotFoundError:
    pass

from ..visu.MidpointNormalize import MidpointNormalize
from .evaluation import Evaluation


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
                                          use_QC=False, time_series_qc=None, variable_qc="vw10m(m/s)"):
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

                obs, nwp, cnn = [df[df.index.isin(index_intersection)] for df in [obs, nwp, cnn]]

                results[variable]["nwp"][station] = nwp
                results[variable]["cnn"][station] = cnn
                results[variable]["obs"][station] = obs

                assert len(results[variable]["nwp"][station]) == len(results[variable]["obs"][station])
                assert len(results[variable]["cnn"][station]) == len(results[variable]["obs"][station])

        return results

    @staticmethod
    def create_df_from_dict(results, data_type="cnn", variables=["UV"]):
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

        result_df = []
        for variable in variables:
            result_df_var_i = []
            list_stations = []
            for station in results[variable][data_type].keys():
                result_df_var_i.append(results[variable][data_type][station])
                list_stations.extend([station] * len(results[variable][data_type][station]))

            result_df_var_i = pd.concat(result_df_var_i)

            if isinstance(result_df_var_i, pd.core.series.Series):
                result_df_var_i = result_df_var_i.to_frame()

            result_df_var_i.columns = [variable]
            result_df.append(result_df_var_i)

        result_df = pd.concat(result_df, axis=1)
        result_df["name"] = list_stations

        return result_df

    def update_df_with_topo_carac(self, df, variables=["laplacian", "alti", "mu", "tpi_2000", "tpi_500", "curvature"]):
        """update result DataFrame with topographical information"""
        if not self.is_updated_with_topo_characteristics:
            self.update_station_with_topo_characteristics()

        stations = self.observation.stations

        for variable in variables:

            df[variable] = np.nan

            for station in stations["name"].values:
                filter_station_df = df["name"] == station
                filter_station = stations["name"] == station
                df[variable][filter_station_df] = stations[variable][filter_station].values[0]

        return df

    @staticmethod
    def _group_metric_result_by_alti_category(metric_result, z0=1030, z1=1675, z2=2500):
        metric_result["group_alti"] = np.nan

        filter_low = metric_result["alti"] < z0
        metric_result["group_alti"][filter_low] = f"Station altitude <= {z0}m"

        filter_low = z0 < metric_result["alti"]
        filter_high = metric_result["alti"] <= z1
        metric_result["group_alti"][filter_low & filter_high] = f"{z0}m < Station altitude <= {z1}m"

        filter_low = z1 < metric_result["alti"]
        filter_high = metric_result["alti"] <= z2
        metric_result["group_alti"][filter_low & filter_high] = f"{z1}m < Station altitude <= {z2}m"

        filter_low = metric_result["alti"] > z2
        metric_result["group_alti"][filter_low] = f"{z2}m < Station altitude"

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
            f"{w1} m/s <= NWP wind speed < {w2} m/s: {len(metric_result[f'group_{variable}_nwp'][filter_low & filter_high])} ")

        filter_low = metric_result[f"nwp_{variable}"] >= w2
        filter_high = metric_result[f"nwp_{variable}"] < w3
        metric_result[f"group_{variable}_nwp"][filter_low & filter_high] = f"{w2} m/s <= NWP wind speed < {w3} m/s"
        print(f"{w2} m/s <= NWP wind speed < {w3} m/s: {len(metric_result[f'group_{variable}_nwp'][filter_low & filter_high])} ")

        filter_low = metric_result[f"nwp_{variable}"] >= w3
        metric_result[f"group_{variable}_nwp"][filter_low] = f"{w3} m/s <= NWP wind speed"
        print(f"{w3} m/s <= NWP wind speed: {len(metric_result[f'group_{variable}_nwp'][filter_low])} ")

        return metric_result

    def plot_metric_all_stations(self, results, plot_classic=False,
                                 variables=["UV"], metrics=["abs_error"], topo_carac="alti", fontsize=15,
                                 sort_by="alti", cmap="viridis", use_QC=False, time_series_qc=None, showfliers=False):

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

                """
                # Classic
                if plot_classic:
                    self._plot_classic(data=metric_result, x=topo_carac, y=metric, fontsize=fontsize)

                # Boxplot
                for y in ["name", "alti", "hour", "month", "tpi_2000", "tpi_500", "laplacian", "mu", "curvature"]:
                    sort_by_i = sort_by if y == "name" else None
                    self._plot_boxplot(data=metric_result, x=metric, y=y, sort_by=sort_by_i)
                    ax = plt.gca()
                    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

                # Boxplot with two models
                self._plot_two_boxplot(data=metric_result, metric=metric)
                ax = plt.gca()
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
                """

                # Boxplot with two models
                metric_result = self._group_metric_result_by_alti_category(metric_result)
                order = ["Station altitude <= 1030m", "1030m < Station altitude <= 1675m",
                         "1675m < Station altitude <= 2500m", "2500m < Station altitude"]

                self._plot_two_boxplot(data=metric_result, metric=metric, y="group_alti", order=order,
                                       showfliers=showfliers)
                ax = plt.gca()
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

                """
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
                         use_QC=False, time_series_qc=None):
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

            bins = (100, 100)
            ax = self.density_scatter(nwp_values, cnn_values, s=5, bins=bins, cmap=cmap)
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

            ax = self.density_scatter(obs_values, cnn_values, s=5, bins=bins, cmap=cmap)
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

            ax = self.density_scatter(obs_values, nwp_values, s=5, bins=bins, cmap=cmap)
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

    def plot_heatmap(self, results, variables=["UV"], metrics=["abs_error"], topo_carac="alti", fontsize=15,
                     sort_by="alti", plot_classic=False, cmap="viridis",
                     use_QC=False, time_series_qc=None, min_speed=1):

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
                piv_table = metric_result.pivot_table(index="month", columns="hour", values=metric, aggfunc='mean')
                plt.figure()
                ax = plt.gca()
                sns.heatmap(piv_table, ax=ax, cmap='coolwarm',
                            norm=MidpointNormalize(midpoint=0, vmin=-0.20, vmax=0.20))
                plt.title(f"{metric}: month vs hour")

                # Heatmap month vs hour for NWP
                piv_table = metric_result.pivot_table(index="month", columns="hour", values=f"nwp_{metric}",
                                                      aggfunc='mean')
                plt.figure()
                ax = plt.gca()
                sns.heatmap(piv_table, ax=ax, cmap='coolwarm',
                            norm=MidpointNormalize(midpoint=0, vmin=-0.20, vmax=0.20))
                plt.title(f"{metric}: month vs hour for NWP")

                # Heatmap month vs hour for difference CNN NWP
                piv_table = metric_result.pivot_table(index="month", columns="hour", values=f"diff_cnn_nwp_{metric}",
                                                      aggfunc='mean')
                plt.figure()
                ax = plt.gca()
                sns.heatmap(piv_table, ax=ax, cmap='coolwarm',
                            norm=MidpointNormalize(midpoint=0, vmin=-0.03, vmax=0.03))
                plt.title(f"{metric}: month vs hour for difference CNN NWP")

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
    def _plot_two_boxplot(data=None, metric=None, y="name", fontsize=15, order=None, showfliers=False):
        plt.figure()
        ax = plt.gca()
        data = data.melt(id_vars=[y], var_name='dataset', value_name='values')
        data = data[data["dataset"].isin([metric, f"nwp_{metric}"])]
        sns.boxplot(data=data, y=y, x="values", hue="dataset", orient="h", showfliers=showfliers,
                    palette=["C0", "C1"], ax=ax, order=order)

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