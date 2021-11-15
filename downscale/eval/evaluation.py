import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ModuleNotFoundError:
    pass

from downscale.eval.metrics import Metrics
from downscale.utils.context_managers import print_all_context


class Evaluation(Metrics):

    def __init__(self, v, prm={"verbose": True}):
        with print_all_context("Evaluation", level=0, unit="second", verbose=prm.get("verbose")):

            super().__init__()

            self.v = v
            self._rename_columns_obs_time_series() if self.v.p.observation is not None else None

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
