import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from downscale.eval.evaluation import Evaluation


class EvaluationFromArrayXr(Evaluation):

    def __init__(self, v, array_xr=None, prm={"verbose": True}):
        super().__init__(v, prm=prm)

        if array_xr is not None:
            array_xr["Wind_DIR"] = array_xr["UV_DIR_deg"]
            self.array_xr = array_xr
            # self.array_xr = array_xr.rename({"UV_DIR_deg": "Wind_DIR"})

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

    def _select_dataframe(self, array_xr, station_name='Col du Lac Blanc', begin=None, end=None, day=None, month=None,
                          year=2019, variable="UV", rolling_mean=None, rolling_window="24H",
                          extract_around="station", interp_str="", interp_str_nwp=""):
        """
        Create 3 dataframes corresponding to NWP, CNN and observations at the specified location, time and rolling_mean
        Input:
        array_xr (predictions), station_name, variable (default "UV"),
        year (default 2019), month, day
        Output:
        nwp_time_serie, cnn_predictions, obs_time_serie (dataframe)
        """

        # Create DataFrames
        nwp_time_serie = self.create_dataframe_from_nwp_pixel(station_name, interp_str=interp_str_nwp)
        cnn_predictions = self.create_dataframe_from_predictions(station_name=station_name, array_xr=array_xr,
                                                                 extract_around=extract_around, interp_str=interp_str)

        # self._rename_columns_obs_time_series()
        obs_time_serie = self.v.p.observation.time_series
        obs_time_serie = obs_time_serie[obs_time_serie["name"] == station_name]

        # Time conditions
        variable_nwp = variable if variable != "UV_DIR_deg" else "Wind_DIR"
        variable_obs = variable if variable != "UV_DIR_deg" else "Wind_DIR"

        if begin is not None and end is not None:
            try:
                nwp_time_serie = self._select_dataframe_time_window_begin_end(nwp_time_serie[variable_nwp],
                                                                              begin=begin, end=end)
            except KeyError:
                nwp_time_serie = pd.DataFrame()
            cnn_predictions = self._select_dataframe_time_window_begin_end(cnn_predictions[variable],
                                                                           begin=begin, end=end)
            try:
                obs_time_serie = self._select_dataframe_time_window_begin_end(obs_time_serie[variable_obs],
                                                                          begin=begin, end=end)
            except KeyError:
                obs_time_serie = pd.DataFrame()
        else:
            try:
                nwp_time_serie = self._select_dataframe_time_window(nwp_time_serie[variable_nwp],
                                                                    day=day, month=month, year=year)
            except KeyError:
                nwp_time_serie = pd.DataFrame()
            cnn_predictions = self._select_dataframe_time_window(cnn_predictions[variable],
                                                                 day=day, month=month, year=year)
            try:
                obs_time_serie = self._select_dataframe_time_window(obs_time_serie[variable_obs],
                                                                    day=day, month=month, year=year)
            except KeyError:
                obs_time_serie = pd.DataFrame()

        # Rolling mean
        if rolling_mean is not None:
            nwp_time_serie = nwp_time_serie.rolling(rolling_window).mean()
            cnn_predictions = cnn_predictions.rolling(rolling_window).mean()
            obs_time_serie = obs_time_serie.rolling(rolling_window).mean()

        return nwp_time_serie, cnn_predictions, obs_time_serie

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