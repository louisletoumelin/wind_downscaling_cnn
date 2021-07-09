import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time as t

from downscale.Analysis.Metrics import Metrics


class Evaluation(Metrics):

    def __init__(self, v, array_xr=None, verbose=True):
        t0 = t()
        super().__init__()
        self.v = v
        if array_xr is not None:
            self.array_xr = array_xr.rename({"UV_DIR_deg": "Wind_DIR"})
        self._rename_columns_obs_time_series()

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

        delta_idx_x = delta_idx_x if extract_around is not None else 0
        delta_idx_y = delta_idx_y if extract_around is not None else 0

        idx_x = np.intp(34 - delta_idx_x)
        idx_y = np.intp(39 - delta_idx_y)

        dataframe = array_xr[
            ['U', 'V', 'W', 'UV', 'UVW', 'UV_DIR_deg', 'alpha_deg', 'NWP_wind_speed',
             'NWP_wind_DIR', 'ZS_mnt']].sel(station=station_name).isel(x=idx_x, y=idx_y).to_dataframe()

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
            nwp_time_serie = self._select_dataframe_time_window_begin_end(nwp_time_serie[variable],
                                                                          begin=begin, end=end)
            cnn_predictions = self._select_dataframe_time_window_begin_end(cnn_predictions[variable],
                                                                           begin=begin, end=end)
            obs_time_serie = self._select_dataframe_time_window_begin_end(obs_time_serie[variable],
                                                                          begin=begin, end=end)
        else:
            nwp_time_serie = self._select_dataframe_time_window(nwp_time_serie[variable],
                                                                day=day, month=month, year=year)
            cnn_predictions = self._select_dataframe_time_window(cnn_predictions[variable],
                                                                 day=day, month=month, year=year)
            obs_time_serie = self._select_dataframe_time_window(obs_time_serie[variable],
                                                                day=day, month=month, year=year)

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
    def intersection_model_obs_on_results(results):

        for station in results["cnn"].keys():
            nwp = results["nwp"][station].dropna()
            cnn = results["cnn"][station].dropna()
            obs = results["obs"][station].dropna()

            obs = obs[~obs.index.duplicated(keep='first')]
            cnn = cnn[~cnn.index.duplicated(keep='first')]
            nwp = nwp[~nwp.index.duplicated(keep='first')]

            obs = obs.to_frame().merge(nwp.to_frame(), left_index=True, right_index=True, how='left').dropna()
            nwp = nwp.to_frame().merge(obs, left_index=True, right_index=True, how='left').dropna()
            cnn = cnn.to_frame().merge(obs, left_index=True, right_index=True, how='left').dropna()
            obs = obs.merge(nwp, left_index=True, right_index=True, how='left').dropna()

            results["nwp"][station] = nwp
            results["cnn"][station] = cnn
            results["obs"][station] = obs

            assert len(results["nwp"][station]) == len(results["obs"][station])
            assert len(results["cnn"][station]) == len(results["obs"][station])

        return results

    @staticmethod
    def create_df_from_dict(results, data_type="cnn", variable="UV"):
        """
        result = create_df_with_all_stations_from_dict(results, data_type="cnn", variable="UV")
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
        list_stations = []

        for station in results[data_type].keys():
            result_df.append(results[data_type][station])
            list_stations.extend([station] * len(results[data_type][station]))

        result_df = pd.concat(result_df)
        result_df = result_df[variable]

        if isinstance(result_df, pd.core.series.Series):
            result_df = result_df.to_frame()

        result_df.columns = [variable]
        result_df["name"] = list_stations

        return result_df

    def update_df(self, df, variables=["alti", "mu", "tpi_2000", "tpi_500", "curvature"]):

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

    def plot_metric_all_stations(self, results,
                                 variables=["UV"], metrics=["abs_error"], topo_carac="alti", fontsize=15):

        # Keep only common rows between observation and model
        results = self.intersection_model_obs_on_results(results)

        for variable in variables:

            cnn = self.create_df_from_dict(results, data_type="cnn", variable=variable)
            nwp = self.create_df_from_dict(results, data_type="nwp", variable=variable)
            obs = self.create_df_from_dict(results, data_type="obs", variable=variable)

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

                metric_result = self.update_df(metric_result)

                plt.figure()
                sns.scatterplot(data=metric_result, x=topo_carac, y=metric)
                ax = plt.gca()
                plt.xlabel(metrics, fontsize=fontsize)
                plt.xlabel(topo_carac, fontsize=fontsize)

    @staticmethod
    def plot_bias_long_period_from_dict(results, stations=['Col du Lac Blanc'], groupby='month', rolling_time=None):

        if groupby is not None:
            rolling_time = None

        for station in stations:
            plt.figure()
            nwp = results["nwp"][station]
            cnn = results["cnn"][station]
            obs = results["obs"][station]

            nwp_obs = nwp - obs
            cnn_obs = cnn - obs

            if groupby == 'month':
                nwp_obs.groupby(nwp_obs.index.month).mean().plot(linestyle='--', marker='x')
                cnn_obs.groupby(cnn_obs.index.month).mean().plot(linestyle='--', marker='x')
            elif groupby == 'year':
                nwp_obs.groupby(nwp_obs.index.month).mean().plot(linestyle='--', marker='x')
                cnn_obs.groupby(cnn_obs.index.month).mean().plot(linestyle='--', marker='x')
            elif groupby is None:
                nwp_obs.plot(linestyle='--', marker='x')
                cnn_obs.plot(linestyle='--', marker='x')
            elif rolling_time is not None:
                nwp_obs.rolling(rolling_time).mean().plot(linestyle='--', marker='x')
                cnn_obs.rolling(rolling_time).mean().plot(linestyle='--', marker='x')

            plt.legend(("bias NWP", "bias CNN"))
            plt.ylabel("Wind speed [m/s]")
            plt.title(station)


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



"""
