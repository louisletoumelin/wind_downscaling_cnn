import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time as t

try:
    import seaborn as sns
except ModuleNotFoundError:
    pass

from downscale.Analysis.Metrics import Metrics


class Evaluation(Metrics):

    def __init__(self, v, array_xr=None, verbose=True):
        t0 = t()
        super().__init__()
        self.v = v
        if array_xr is not None:
            self.array_xr = array_xr.rename({"UV_DIR_deg": "Wind_DIR"})
        self._rename_columns_obs_time_series() if self.v.p.observation is not None else None

        print(f"\nEvaluation created in {np.round(t() - t0, 2)} seconds\n") if verbose else None

    def _rename_columns_obs_time_series(self):
        """
        Rename time_series  of observation columns:

        "winddir(deg)" => "Wind_DIR"
        "vw10m(m/s)" => "UV"
        """
        list_variables = self.v.p.observation.time_series.columns
        if 'UV' not in list_variables:
            self.v.p.observation.time_series['UV'] = self.v.p.observation.time_series["vw10m(m/s)"]
        if 'Wind_DIR' not in list_variables:
            self.v.p.observation.time_series['Wind_DIR'] = self.v.p.observation.time_series["winddir(deg)"]

    def create_dataframe_from_nwp_pixel(self, station_name='Col du Lac Blanc', interp_str=""):
        """
        Select a station and extracts correspoding NWP simulations

        Input:
        station name (default 'Col du Lac Blanc')

        Output:
        Dataframe containing NWP values at the station

        """

        wind_dir, wind_speed, time_index = self.v.p._extract_variable_from_nwp_at_station(station_name,
                                                                                          variable_to_extract=[
                                                                                              "wind_direction",
                                                                                              "wind_speed", "time"],
                                                                                          interp_str=interp_str,
                                                                                          verbose=False)

        nwp_time_serie = pd.DataFrame(np.transpose([wind_dir, wind_speed]),
                                      columns=['Wind_DIR', 'UV'], index=time_index)

        return nwp_time_serie

    def create_dataframe_from_predictions(self, station_name='Col du Lac Blanc', array_xr=None,
                                          extract_around="station", interp_str=""):
        """
        Creates a dataframe at a specified location using predictions stored in array_xr (xarray data)

        Input:
        station_name (Default: 'Col du Lac Blanc')
        array_xr: predictions obtained with CNN
        """

        if extract_around == "nwp_neighbor_interp":
            nwp_name = self.v.p.nwp.name
            mnt_name = self.v.p.mnt.name
            idx_str_neighbors = f"index_{nwp_name}_NN_0{interp_str}_ref_{mnt_name}"
            idx_str_station = f"index_{mnt_name}_NN_0_cKDTree_ref_{mnt_name}"

            stations = self.v.p.observation.stations
            filter_station = stations["name"] == station_name
            stations = stations[filter_station]

            delta_idx = stations[idx_str_neighbors].apply(np.array) - stations[idx_str_station].apply(np.array)
            delta_idx_x = delta_idx.apply(lambda x: x[0]).values[0]
            delta_idx_y = delta_idx.apply(lambda x: x[1]).values[0]

        delta_idx_x = delta_idx_x if extract_around != "station" else 0
        delta_idx_y = delta_idx_y if extract_around != "station" else 0

        idx_x = np.intp(34 - delta_idx_x)
        idx_y = np.intp(39 - delta_idx_y)

        dataframe = array_xr[
            ['U', 'V', 'W', 'UV', 'UVW', 'UV_DIR_deg', 'alpha_deg', 'NWP_wind_speed',
             'NWP_wind_DIR', "exp_Wind", "acceleration_CNN", "Z0", 'ZS_mnt']].sel(station=station_name).isel(x=idx_x, y=idx_y).to_dataframe()

        return dataframe

    @staticmethod
    def _select_dataframe_time_window(dataframe, day=None, month=None, year=2019):
        """
        Select a specified date on a dataframe ex: dataframe.index.year == year

        Input:
        dataframe, year (default=2019), month (optional), day (optional)

        Output:
        dataframe for the specified date

        Other: Internal method
        """

        # Time conditions
        time_condition = (dataframe.index.year == year)

        if month is not None:
            time_condition = time_condition & (dataframe.index.month == month)
        if day is not None:
            time_condition = time_condition & (dataframe.index.day == day)

        dataframe = dataframe[time_condition]

        return dataframe

    @staticmethod
    def _select_dataframe_time_window_begin_end(dataframe, begin=None, end=None):

        # Time conditions
        time_condition_begin = begin <= dataframe.index
        time_condition_end = dataframe.index <= end

        dataframe = dataframe[time_condition_begin & time_condition_end]

        return dataframe

    def _select_dataframe(self, array_xr, station_name='Col du Lac Blanc', begin=None, end=None, day=None, month=None,
                          year=2019, variable="UV", rolling_mean=None, rolling_window="24H",
                          extract_around="station", interp_str=""):
        """
        Create 3 dataframes corresponding to NWP, CNN and observations at the specified location, time and rolling_mean

        Input:
        array_xr (predictions), station_name, variable (default "UV"),
        year (default 2019), month, day

        Output:
        nwp_time_serie, cnn_predictions, obs_time_serie (dataframe)
        """

        # Create DataFrames
        nwp_time_serie = self.create_dataframe_from_nwp_pixel(station_name, interp_str=interp_str)
        cnn_predictions = self.create_dataframe_from_predictions(station_name=station_name, array_xr=array_xr,
                                                                 extract_around=extract_around, interp_str=interp_str)

        # self._rename_columns_obs_time_series()
        obs_time_serie = self.v.p.observation.time_series
        obs_time_serie = obs_time_serie[obs_time_serie["name"] == station_name]

        # Time conditions
        if begin is not None and end is not None:
            try:
                nwp_time_serie[variable]
                nwp_time_serie = self._select_dataframe_time_window_begin_end(nwp_time_serie[variable],
                                                                              begin=begin, end=end)
            except KeyError:
                nwp_time_serie = pd.DataFrame()
            cnn_predictions = self._select_dataframe_time_window_begin_end(cnn_predictions[variable],
                                                                           begin=begin, end=end)
            try:
                obs_time_serie[variable]
                obs_time_serie = self._select_dataframe_time_window_begin_end(obs_time_serie[variable],
                                                                          begin=begin, end=end)
            except KeyError:
                obs_time_serie = pd.DataFrame()
        else:
            try:
                nwp_time_serie[variable]
                nwp_time_serie = self._select_dataframe_time_window(nwp_time_serie[variable],
                                                                    day=day, month=month, year=year)
            except KeyError:
                nwp_time_serie = pd.DataFrame()
            cnn_predictions = self._select_dataframe_time_window(cnn_predictions[variable],
                                                                 day=day, month=month, year=year)
            try:
                obs_time_serie[variable]
                obs_time_serie = self._select_dataframe_time_window(obs_time_serie[variable],
                                                                    day=day, month=month, year=year)
            except KeyError:
                obs_time_serie = pd.DataFrame()

        # Rolling mean
        if rolling_mean is not None:
            nwp_time_serie = nwp_time_serie.rolling(rolling_window).mean()
            cnn_predictions = cnn_predictions.rolling(rolling_window).mean()
            obs_time_serie = obs_time_serie.rolling(rolling_window).mean()

        return nwp_time_serie, cnn_predictions, obs_time_serie


class EvaluationFromArrayXr(Evaluation):

    def __init__(self, v, array_xr=None, verbose=True):
        super().__init__(v, array_xr=array_xr, verbose=verbose)

    def plot_time_serie_from_array_xr(self, array_xr, station_name='Col du Lac Blanc', day=None, month=None, year=2019,
                                      variable="UV", rolling_mean=None, rolling_window="24H",
                                      color_nwp='C0', color_obs='black', color_cnn='C1',
                                      figsize=(20, 20), new_figure=True,
                                      markersize=2, marker_nwp='x', marker_cnn='x', marker_obs='x',
                                      linestyle_nwp='-', linestyle_cnn='-', linestyle_obs='-',
                                      display_NWP=True, display_CNN=True, display_obs=True):

        # Create a new figure
        if new_figure:
            plt.figure(figsize=figsize)
        else:
            plt.gcf()

        # Select Dataframes

        nwp_time_serie, cnn_predictions, obs_time_serie = self._select_dataframe(array_xr,
                                                                                 station_name=station_name,
                                                                                 day=day, month=month, year=year,
                                                                                 variable=variable,
                                                                                 rolling_mean=rolling_mean,
                                                                                 rolling_window=rolling_window)
        # Plot
        plt.figure(figsize=figsize)
        ax = plt.gca()

        # NWP
        if display_NWP:
            x_nwp = nwp_time_serie.index
            y_nwp = nwp_time_serie.values
            ax.plot(x_nwp, y_nwp, color=color_nwp, marker=marker_nwp, markersize=markersize, linestyle=linestyle_nwp,
                    label='NWP')

        # CNN
        if display_CNN:
            x_cnn = cnn_predictions.index
            y_cnn = cnn_predictions.values
            ax.plot(x_cnn, y_cnn, color=color_cnn, marker=marker_cnn, markersize=markersize, linestyle=linestyle_cnn,
                    label='CNN')

            if display_NWP: assert len(x_nwp) == len(x_cnn)

        # Try to plot observations
        if display_obs:
            try:
                x_obs = obs_time_serie.index
                y_obs = obs_time_serie.values
                ax.plot(x_obs, y_obs, color=color_obs, marker=marker_obs, markersize=markersize,
                        linestyle=linestyle_obs, label='Obs')
            except:
                pass
        plt.legend()
        plt.title(station_name)

    def plot_metric_time_serie_from_array_xr(self, array_xr, station_name='Col du Lac Blanc', metric='bias',
                                             day=None, month=None, year=2019, new_figure=True,
                                             variable="UV", rolling_mean=None, rolling_window="24H",
                                             figsize=(20, 20), color_nwp='C0', color_cnn='C1',
                                             markersize=2, marker_nwp='x', marker_cnn='x',
                                             linestyle_nwp=' ', linestyle_cnn=' '):
        # Create a new figure
        if new_figure:
            plt.figure(figsize=figsize)
        else:
            plt.gcf()

        # Select Dataframes
        nwp_time_serie, cnn_predictions, obs_time_serie = self._select_dataframe(array_xr,
                                                                                 station_name=station_name,
                                                                                 day=day, month=month, year=year,
                                                                                 variable=variable,
                                                                                 rolling_mean=rolling_mean,
                                                                                 rolling_window=rolling_window)
        if metric == 'bias':
            nwp_obs = nwp_time_serie - obs_time_serie
            cnn_obs = cnn_predictions - obs_time_serie

            time_nwp = nwp_obs.index
            time_cnn = cnn_obs.index

            ax = plt.gca()
            ax.plot(time_nwp, nwp_obs, marker=marker_nwp, markersize=markersize, linestyle=linestyle_nwp,
                    color=color_nwp)
            ax.plot(time_cnn, cnn_obs, marker=marker_cnn, markersize=markersize, linestyle=linestyle_cnn,
                    color=color_cnn)

            plt.title(station_name + '\n' + "Bias" + '\n' + variable)


class EvaluationFromDict(Evaluation):

    def __init__(self, v, verbose=True):
        super().__init__(v, array_xr=None, verbose=verbose)

    @staticmethod
    def intersection_model_obs_on_results(results, variables=["UV"]):
        """
        Keep only data were we have simultaneously model and observed data

        Input = dictionary
        Output = dictionary
        """

        for variable in variables:
            for station in results[variable]["cnn"].keys():

                # Drop Nans
                nwp = results[variable]["nwp"][station].dropna()
                cnn = results[variable]["cnn"][station].dropna()
                obs = results[variable]["obs"][station].dropna()

                # Drop duplicates
                obs = obs[~obs.index.duplicated(keep='first')]
                cnn = cnn[~cnn.index.duplicated(keep='first')]
                nwp = nwp[~nwp.index.duplicated(keep='first')]

                # Convert dictionary values to DataFrame (if necessary)
                obs = obs.to_frame() if isinstance(obs, pd.core.series.Series) else obs
                nwp = nwp.to_frame() if isinstance(nwp, pd.core.series.Series) else nwp
                cnn = cnn.to_frame() if isinstance(cnn, pd.core.series.Series) else cnn

                # Keep index intersection
                index_intersection = obs.index.intersection(nwp.index).intersection(cnn.index)
                obs = obs[obs.index.isin(index_intersection)]
                nwp = nwp[nwp.index.isin(index_intersection)]
                cnn = cnn[cnn.index.isin(index_intersection)]

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
            result_df_var_i

            if isinstance(result_df_var_i, pd.core.series.Series):
                result_df_var_i = result_df_var_i.to_frame()

            result_df_var_i.columns = [variable]
            result_df.append(result_df_var_i)

        result_df = pd.concat(result_df, axis=1)
        result_df["name"] = list_stations

        return result_df

    def update_df(self, df, variables=["laplacian", "alti", "mu", "tpi_2000", "tpi_500", "curvature"]):
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
    def _group_metric_result_by_wind_category(metric_result, variable):
        metric_result[f"group_{variable}_nwp"] = np.nan

        filter_low = metric_result[f"nwp_{variable}"] < 5
        metric_result[f"group_{variable}_nwp"][filter_low] = "NWP wind speed < 5 m/s"

        filter_low = metric_result[f"nwp_{variable}"] >= 5
        filter_high = metric_result[f"nwp_{variable}"] < 10
        metric_result[f"group_{variable}_nwp"][filter_low & filter_high] = "5 m/s <= NWP wind speed < 10 m/s"

        filter_low = metric_result[f"nwp_{variable}"] >= 10
        filter_high = metric_result[f"nwp_{variable}"] < 15
        metric_result[f"group_{variable}_nwp"][filter_low & filter_high] = "10 m/s <= NWP wind speed < 15 m/s"

        filter_low = metric_result[f"nwp_{variable}"] >= 15
        filter_high = metric_result[f"nwp_{variable}"] < 20
        metric_result[f"group_{variable}_nwp"][filter_low & filter_high] = "15 m/s <= NWP wind speed < 20 m/s"

        filter_high = metric_result[f"nwp_{variable}"] >= 20
        metric_result[f"group_{variable}_nwp"][filter_high] = "20 m/s <= NWP wind speed"
        return metric_result

    def plot_metric_all_stations(self, results,
                                 variables=["UV"], metrics=["abs_error"], topo_carac="alti", fontsize=15,
                                 sort_by="alti", plot_classic=False, cmap="viridis"):

        # Keep only common rows between observation and model
        results = self.intersection_model_obs_on_results(results, variables=variables)

        # Create DataFrame
        cnn = self.create_df_from_dict(results, data_type="cnn", variables=variables)
        nwp = self.create_df_from_dict(results, data_type="nwp", variables=variables)
        obs = self.create_df_from_dict(results, data_type="obs", variables=variables)
        cnn["hour"] = cnn.index.hour
        cnn["month"] = cnn.index.month

        for variable in variables:

            for metric in metrics:
                if metric == "abs_error":
                    metric_func = self.absolute_error
                if metric == "bias":
                    metric_func = self.bias
                if metric == "abs_error_rel":
                    metric_func = self.absolute_error_relative
                if metric == "bias_rel":
                    metric_func = self.bias_rel

                metric_result = cnn.copy()
                metric_result[metric] = metric_func(cnn[variable].values, obs[variable].values)
                metric_result[f"nwp_{variable}"] = nwp[variable].values

                metric_result[f"nwp_{metric}"] = metric_func(nwp[variable].values, obs[variable].values)
                metric_result = self.update_df(metric_result)

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
                self.v.density_scatter(nwp_metric, cnn_metric, s=5, bins=bins, cmap=cmap, use_power_norm=2)
                max = np.nanmax([nwp_metric, cnn_metric])
                min = np.nanmin([nwp_metric, cnn_metric])
                plt.xlim((min - 2, max + 2))
                plt.ylim((min - 2, max + 2))
                plt.plot(nwp_metric, nwp_metric, color='red')
                plt.axis('square')
                plt.xlabel(f"{metric} NWP")
                plt.ylabel(f"{metric} CNN")

    def plot_distribution_all_stations(self, results, variables=["UV"]):
        """Wind speed distribution (one distribution for all stations)"""
        results = self.intersection_model_obs_on_results(results, variables=variables)
        all_df = []
        for variable in variables:
            for model in ["cnn", "nwp", "obs"]:
                data = [[station, data[0]]
                        for station, df_data in results[variable][model].items()
                        for data in df_data.values]
                df = pd.DataFrame(data, columns=["station", variable])
                df["data"] = model
                all_df.append(df)
            all_df = pd.concat(all_df)

            sns.displot(data=all_df, x=variable, hue="data", kind='kde', fill=True, bw_adjust=3, cut=0)

    def plot_1_1_density(self, results, cmap="viridis", variables=["UV"]):
        """
        1-1 plots

        NWP-CNN, CNN-obs, NWP-obs
        """
        results = self.intersection_model_obs_on_results(results, variables=variables)

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
            self.v.density_scatter(nwp_values, cnn_values, s=5, bins=bins, cmap=cmap)
            max = np.nanmax([nwp_values, cnn_values])
            min = np.nanmin([nwp_values, cnn_values])
            plt.xlim((min - 2, max + 2))
            plt.ylim((min - 2, max + 2))
            plt.plot(nwp_values, nwp_values, color='red')
            plt.axis('square')

            self.v.density_scatter(obs_values, cnn_values, s=5, bins=bins, cmap=cmap)
            max = np.nanmax([obs_values, cnn_values])
            min = np.nanmin([obs_values, cnn_values])
            plt.xlim((min - 2, max + 2))
            plt.ylim((min - 2, max + 2))
            plt.plot(obs_values, obs_values, color='red')
            plt.axis('square')

            self.v.density_scatter(obs_values, nwp_values, s=5, bins=bins, cmap=cmap)
            max = np.nanmax([obs_values, nwp_values])
            min = np.nanmin([obs_values, nwp_values])
            plt.xlim((min - 2, max + 2))
            plt.ylim((min - 2, max + 2))
            plt.plot(obs_values, obs_values, color='red')
            plt.axis('square')

    def plot_heatmap(self, results, variables=["UV"], metrics=["abs_error"], topo_carac="alti", fontsize=15,
                      sort_by="alti", plot_classic=False, cmap="viridis"):

        # Keep only common rows between observation and model
        results = self.intersection_model_obs_on_results(results)

        cnn = self.create_df_from_dict(results, data_type="cnn", variables=variables)
        nwp = self.create_df_from_dict(results, data_type="nwp", variables=variables)
        obs = self.create_df_from_dict(results, data_type="obs", variables=variables)

        cnn["hour"] = cnn.index.hour
        cnn["month"] = cnn.index.month

        for variable in variables:

            for metric in metrics:
                if metric == "abs_error":
                    metric_func = self.absolute_error
                if metric == "bias":
                    metric_func = self.bias
                if metric == "abs_error_rel":
                    metric_func = self.absolute_error_relative
                if metric == "bias_rel":
                    metric_func = self.bias_rel

                metric_result = cnn.copy()
                metric_result[metric] = metric_func(cnn[variable].values, obs[variable].values)
                metric_result[f"nwp_{variable}"] = nwp[variable].values

                metric_result[f"nwp_{metric}"] = metric_func(nwp[variable].values, obs[variable].values)
                metric_result = self.update_df(metric_result)

                metric_result[f"diff_cnn_nwp_{metric}"] = metric_result[metric] - metric_result[f"nwp_{metric}"]

                # Heatmap month vs hour
                piv_table = metric_result.pivot_table(index="month", columns="hour", values=metric, aggfunc='mean')
                plt.figure()
                ax = plt.gca()
                sns.heatmap(piv_table, cmap="viridis", ax=ax)
                plt.title(f"{metric}: month vs hour")

                # Heatmap month vs hour for NWP
                piv_table = metric_result.pivot_table(index="month", columns="hour", values=f"nwp_{metric}", aggfunc='mean')
                plt.figure()
                ax = plt.gca()
                sns.heatmap(piv_table, cmap="viridis", ax=ax)
                plt.title(f"{metric}: month vs hour for NWP")

                # Heatmap month vs hour for difference CNN NWP
                piv_table = metric_result.pivot_table(index="month", columns="hour", values=f"diff_cnn_nwp_{metric}", aggfunc='mean')
                plt.figure()
                ax = plt.gca()
                sns.heatmap(piv_table, cmap="viridis", ax=ax)
                plt.title(f"{metric}: month vs hour for difference CNN NWP")

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

    @staticmethod
    def _plot_classic(data=None, x=None, y=None, fontsize=15):
        plt.figure()
        sns.scatterplot(data=data, x=x, y=y)
        plt.ylabel(y, fontsize=fontsize)
        plt.xlabel(x, fontsize=fontsize)

    @staticmethod
    def _plot_boxplot(data=None, sort_by="alti", y="name", x="absolute error"):
        plt.figure()
        list_ordered = list(data.sort_values(by=[sort_by])[y].unique()) if sort_by is not None else None
        sns.boxplot(data=data, y=y, x=x, orient="h", showfliers=False, order=list_ordered)
        sns.despine(trim=True, left=True)

    @staticmethod
    def _plot_two_boxplot(data=None, metric=None):
        plt.figure()
        ax = plt.gca()
        data = data.melt(id_vars=["name"], var_name='dataset', value_name='values')
        data = data[data["dataset"].isin([metric, f"nwp_{metric}"])]

        sns.boxplot(data=data, y="name", x="values", hue="dataset", orient="h", showfliers=False,
                    palette=["Red", "Blue"], ax=ax)
        ax = plt.gca()
        ax.set(ylabel="")
        sns.despine(trim=True, left=True)

    def create_three_df_results(self, results, variable="UV"):

        results = self.intersection_model_obs_on_results(results, variables=[variable])

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

"""
RMSE_nwp = []
RMSE_cnn = []
pearson_correlation_nwp = []
pearson_correlation_cnn = []
bias_nwp = []
bias_cnn = []
cos_deviation_nwp = []
cos_deviation_cnn = []

def cos_deviation(pred, true):
    pred = np.pi * pred / 180
    true = np.pi * true / 180
    return(np.mean((1/2)*(1-np.cos(pred, true))))

for station in BDclim.stations["name"]:
    nwp, cnn, obs = e._select_dataframe(array_xr, station_name=station, day=None, month=9,year=2018, variable='UV', rolling_mean=None, rolling_window=None)
    print(station)
    try:
        print("RMSE NWP: ", e.RMSE(nwp, obs))
        RMSE_nwp.append(e.RMSE(nwp, obs))
        print("RMSE CNN: ", e.RMSE(cnn, obs))
        RMSE_cnn.append(e.RMSE(cnn, obs))
        print("corr coeff NWP: ", e.pearson_correlation(nwp, obs))
        pearson_correlation_nwp.append(e.pearson_correlation(nwp, obs))
        print("corr coeff  CNN: ", e.pearson_correlation(cnn, obs))
        pearson_correlation_cnn.append(e.pearson_correlation(cnn, obs))
        print("bias NWP: ", e.mean_bias(nwp, obs))
        bias_nwp.append(e.mean_bias(nwp, obs))
        print("bias CNN: ", e.mean_bias(cnn, obs))
        bias_cnn.append(e.mean_bias(cnn, obs))
        print("Cosinus deviation NWP:", cos_deviation(nwp, obs))
        cos_deviation_nwp.append(cos_deviation(nwp, obs))
        print("Cosinus deviation NWP:", cos_deviation(cnn, obs))
        cos_deviation_cnn.append(cos_deviation(cnn, obs))
    except:
        pass

# Septembre

print(np.nanmean(RMSE_nwp))
print(np.nanmean(RMSE_cnn))
print(np.nanmean(pearson_correlation_nwp))
print(np.nanmean(pearson_correlation_cnn))
print(np.nanmean(bias_nwp))
print(np.nanmean(bias_cnn))
print(np.nanmean(cos_deviation_nwp))
print(np.nanmean(cos_deviation_cnn))
1.3695469247606928
1.6713452531909345
0.5702183566090461
0.5708304297320466
-0.13584922638229838
0.5035137805435913
0.26571486711478093
0.3009208003428228











results = e.intersection_model_obs_on_results(results)
cnn = e.create_df_from_dict(results, data_type="cnn", variable=variable)
nwp = e.create_df_from_dict(results, data_type="nwp", variable=variable)
obs = e.create_df_from_dict(results, data_type="obs", variable=variable)
metric_result = cnn.copy()
metric_func = e.absolute_error
metric = "abs_error"
metric_result[metric] = metric_func(cnn[variable].values, obs[variable].values)
metric_result[f"nwp_{metric}"] = metric_func(nwp[variable].values, obs[variable].values)
metric_func = e.bias
metric = "bias"
metric_result[metric] = metric_func(cnn[variable].values, obs[variable].values)
metric_result[f"nwp_{metric}"] = metric_func(nwp[variable].values, obs[variable].values)
metric_by_name = metric_result.groupby("name").mean()[['abs_error', 'bias']]
df_merge = pd.concat([cnn, obs], axis=1)
df_merge.columns = ["UV", "name", "UV_obs", "name_obs"]
df_merge = df_merge.drop(columns="name_obs")
correlation = df_merge.groupby("name").apply(lambda x: x.corr().iloc[0,1])
metric_by_name["corr_coeff"] = correlation
metric_by_name = metric_by_name["abs_error"]
fig = plt.figure(figsize=(50, 500))
ax = plt.gca()
sns.heatmap(np.transpose(metric_by_name.to_frame()), annot=True, cmap="viridis_r", annot_kws={"size":8}, square=True, fmt="0.1f", linewidths=.5, vmax=5, cbar_kws = dict(shrink=0.25), ax=ax)
xlabel = ax.get_xticklabels()
ax.set_xticklabels(xlabel, fontsize=10)
ylabel = ax.get_yticklabels()
ax.set_yticklabels(ylabel, rotation=0, fontsize=15)
ax.set_xlabel(None)
fig.axes[1].set_visible(False)
plt.tight_layout()

"""
