import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from time import time as t

from Visualization import Visualization


class Evaluation:

    def __init__(self, v, array_xr=None):
        t0 = t()
        self.v = v
        if array_xr is not None:
            self.array_xr = array_xr.rename({"UV_DIR_deg": "Wind_DIR"})
        self._rename_colums_obs_time_series()
        t1 = t()
        print(f"\nEvaluation created in {np.round(t1-t0, 2)} seconds\n")

    def create_dataframe_from_nwp_pixel(self, station_name='Col du Lac Blanc'):
        """
        Select a station and extracts correspoding NWP simulations

        Input:
        station name (default 'Col du Lac Blanc')

        Output:
        Dataframe containing NWP values at the station

        """
        wind_dir, wind_speed, time_index, _, _, _ = self.v.p._select_nwp_time_serie_at_pixel(station_name, time=False, all=True)
        nwp_time_serie = pd.DataFrame(np.transpose([wind_dir, wind_speed]), columns=['Wind_DIR', 'UV'], index=time_index)
        return (nwp_time_serie)

    def create_dataframe_from_predictions(self, station_name='Col du Lac Blanc', array_xr=None):
        """
        Creates za dataframe at a specified location using predictions stored in array_xr (xarray data)

        Input:
        station_name (Default: 'Col du Lac Blanc')
        array_xr: predictions obtained with CNN
        """
        dataframe = array_xr[
            ['U', 'V', 'W', 'UV', 'UVW', 'UV_DIR_deg', 'alpha_deg', 'NWP_wind_speed',
             'NWP_wind_DIR', 'ZS_mnt']].sel(station=station_name).isel(x=34, y=39).to_dataframe()
        return (dataframe)

    def _rename_colums_obs_time_series(self):
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

        return(dataframe)

    def _select_dataframe(self, array_xr, station_name='Col du Lac Blanc', day=None, month=None, year=2019,
                          variable="UV", rolling_mean=None, rolling_window="24H"):
        """
        Create 3 dataframes corresponding to NWP, CNN and observations at the specified location, time and rolling_mean

        Input:
        array_xr (predictions), station_name, variable (default "UV"),
        year (default 2019), month, day

        Output:
        nwp_time_serie, cnn_predictions, obs_time_serie (dataframe)
        """
        # Create DataFrames
        nwp_time_serie = self.create_dataframe_from_nwp_pixel(station_name)
        cnn_predictions = self.create_dataframe_from_predictions(station_name=station_name, array_xr=array_xr)

        # self._rename_colums_obs_time_series()
        obs_time_serie = self.v.p.observation.time_series
        obs_time_serie = obs_time_serie[obs_time_serie["name"] == station_name]

        # Time conditions
        nwp_time_serie = self._select_dataframe_time_window(nwp_time_serie[variable], day=day, month=month, year=year)
        cnn_predictions = self._select_dataframe_time_window(cnn_predictions[variable], day=day, month=month, year=year)
        obs_time_serie = self._select_dataframe_time_window(obs_time_serie[variable], day=day, month=month, year=year)

        # Rolling mean
        if rolling_mean is not None:
            nwp_time_serie = nwp_time_serie.rolling(rolling_window).mean()
            cnn_predictions = cnn_predictions.rolling(rolling_window).mean()
            obs_time_serie = obs_time_serie.rolling(rolling_window).mean()

        return (nwp_time_serie, cnn_predictions, obs_time_serie)

    def plot_time_serie(self, array_xr, station_name='Col du Lac Blanc', day=None, month=None, year=2019,
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
            ax.plot(x_nwp, y_nwp, color=color_nwp, marker=marker_nwp, markersize=markersize, linestyle=linestyle_nwp, label='NWP')

        # CNN
        if display_CNN:
            x_cnn = cnn_predictions.index
            y_cnn = cnn_predictions.values
            ax.plot(x_cnn, y_cnn, color=color_cnn, marker=marker_cnn, markersize=markersize, linestyle=linestyle_cnn, label='CNN')

            if display_NWP: assert len(x_nwp) == len(x_cnn)

        # Try to plot observations
        if display_obs:
            try:
                x_obs = obs_time_serie.index
                y_obs = obs_time_serie.values
                ax.plot(x_obs, y_obs, color=color_obs, marker=marker_obs, markersize=markersize, linestyle=linestyle_obs, label='Obs')
            except:
                pass
        plt.legend()
        plt.title(station_name)

    def plot_metric_time_serie(self, array_xr, station_name='Col du Lac Blanc', metric='bias',
                               day=None, month=None, year=2019,
                               new_figure=True,
                               variable="UV", rolling_mean=None, rolling_window="24H",
                               figsize=(20, 20),
                               color_nwp='C0', color_cnn='C1',
                               markersize=2, marker_nwp='x', marker_cnn='x', linestyle_nwp=' ', linestyle_cnn=' '):
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
            ax.plot(time_nwp, nwp_obs, marker=marker_nwp, markersize=markersize, linestyle=linestyle_nwp, color=color_nwp)
            ax.plot(time_cnn, cnn_obs, marker=marker_cnn, markersize=markersize, linestyle=linestyle_cnn, color=color_cnn)

            plt.title(station_name + '\n' + "Bias" + '\n' + variable)

    @staticmethod
    def plot_bias_long_period(results, stations=['Col du Lac Blanc'], groupby='month', rolling_time=None):

        if groupby is not None:
            rolling_time = None

        for station in stations:
            plt.figure()
            nwp = pd.concat(results["nwp"][station])
            cnn = pd.concat(results["cnn"][station])
            obs = pd.concat(results["obs"][station])

            nwp_obs = nwp - obs
            cnn_obs = cnn - obs

            if groupby == 'month':
                nwp_obs.groupby(nwp_obs.index.month).mean().plot(linestyle='--', marker='x')
                cnn_obs.groupby(cnn_obs.index.month).mean().plot(linestyle='--', marker='x')
            elif groupby == 'year':
                nwp_obs.groupby(nwp_obs.index.month).mean().plot(linestyle='--', marker='x')
                cnn_obs.groupby(cnn_obs.index.month).mean().plot(linestyle='--', marker='x')
            elif groupby == None:
                nwp_obs.plot(linestyle='--', marker='x')
                cnn_obs.plot(linestyle='--', marker='x')
            elif rolling_time is not None:
                nwp_obs.rolling(rolling_time).mean().plot(linestyle='--', marker='x')
                cnn_obs.rolling(rolling_time).mean().plot(linestyle='--', marker='x')

            plt.legend(("bias NWP", "bias CNN"))
            plt.ylabel("Wind speed [m/s]")
            plt.title(station)

    @staticmethod
    def pearson_correlation(y_true, y_pred):
        # return(tf.linalg.trace(tfp.stats.correlation(y_pred, y_true))/3)
        return (pd.concat([pd.DataFrame(y_true), pd.DataFrame(y_pred)], axis=1).corr().iloc[0, 1])

    @staticmethod
    def RMSE(pred, true):
        if type(pred) == type(true) == pd.core.frame.DataFrame:
            diff = (pred - true) ** 2
            return (diff.mean() ** 0.5)
        else:
            pred = np.array(pred)
            true = np.array(true)
            return (np.sqrt(np.nanmean((pred - true) ** 2)))

    @staticmethod
    def mean_bias(pred, true):
        if type(pred) == type(true) == pd.core.frame.DataFrame:
            diff = (pred - true)
            return (diff.mean())
        else:
            pred = np.array(pred)
            true = np.array(true)
        return (np.nanmean(pred - true))

    @staticmethod
    def bias(pred, true):
        if type(pred) == type(true) == pd.core.frame.DataFrame:
            diff = (pred - true)
            return (diff)
        else:
            pred = np.array(pred)
            true = np.array(true)
        return (pred - true)

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