from time import time as t
import uuid

import pandas as pd
import matplotlib
#matplotlib.use('Agg')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    pass

t_init = t()

# Create prm
from PRM_predict import create_prm

prm = create_prm(month_prediction=True)

from downscale.data_source.observation import Observation
from downscale.utils.utils_func import save_figure

# BDclim
BDclim = Observation(prm["BDclim_stations_path"], prm["BDclim_data_path"], prm=prm)
BDclim.time_series = pd.read_pickle(prm["QC_pkl"])
print(prm["QC_pkl"])
time_series_qc = BDclim.time_series
BDclim.replace_obs_by_QC_obs(prm=prm, drop_not_valid=False)
time_series_valid = BDclim.time_series
assert 'qc' in time_series_valid.columns

sns.set_style("darkgrid")
matplotlib.use('Agg')
"""
prm["hour_begin"] = 1
prm["day_begin"] = 2
prm["month_begin"] = 8
prm["year_begin"] = 2017

# 31 May 2020 1h
prm["hour_end"] = 1
prm["day_end"] = 31
prm["month_end"] = 5
prm["year_end"] = 2020

prm["begin"] = str(prm["year_begin"]) + "-" + str(prm["month_begin"]) + "-" + str(prm["day_begin"])
prm["begin_after"] = str(prm["year_begin"]) + "-" + str(prm["month_begin"]) + "-" + str(prm["day_begin"] + 1)
prm["end"] = str(prm["year_end"]) + "-" + str(prm["month_end"]) + "-" + str(prm["day_end"])

filter_time_valid = (time_series_valid.index >= prm["begin"]) & (time_series_valid.index <= prm["end"])
time_series_valid = time_series_valid[filter_time_valid]
"""

time_series_valid["Suspicious data"] = "Valid observation"

filter_unphysical_speed = (time_series_valid['qc_1_speed'] != 1)
time_series_valid["Suspicious data"][filter_unphysical_speed] = ["Speed: unphysical"]

filter_unphysical_direction = (time_series_valid['qc_1_direction'] != 1)
time_series_valid["Suspicious data"][filter_unphysical_direction] = ["Direction: unphysical"]

filter_excessive_MISS_speed = (time_series_valid['qc_2_speed'] != 1)
time_series_valid["Suspicious data"][filter_excessive_MISS_speed] = ["Speed: excessive miss"]

filter_excessive_MISS_direction = (time_series_valid['qc_2_direction'] != 1)
time_series_valid["Suspicious data"][filter_excessive_MISS_direction] = ["Direction: excessive miss"]

filter_constant_sequence_speed = (time_series_valid['qc_3_speed'] != 1)
time_series_valid["Suspicious data"][filter_constant_sequence_speed] = ["Speed: constant seq."]

filter_constant_sequence_direction = (time_series_valid['qc_3_direction'] != 1) & (
            time_series_valid['qc_3_direction_pref'] != 1)
time_series_valid["Suspicious data"][filter_constant_sequence_direction] = ["Direction: constant seq."]

filter_high_variability_speed = (time_series_valid['qc_5_speed'] != 1)
time_series_valid["Suspicious data"][filter_high_variability_speed] = ["Speed: high vari."]

filter_bias_speed = (time_series_valid['qc_6_speed'] != 1)
time_series_valid["Suspicious data"][filter_bias_speed] = ["Speed: bias"]

filter_isolated_recors_speed = (time_series_valid['qc_7_isolated_records_speed'] != 1)
time_series_valid["Suspicious data"][filter_isolated_recors_speed] = ["Speed: isolated record"]

filter_isolated_recors_direction = (time_series_valid['qc_7_isolated_records_direction'] != 1)
time_series_valid["Suspicious data"][filter_isolated_recors_direction] = ["Direction: isolated record"]

palette = dict(zip(time_series_valid["Suspicious data"].unique(),
                   sns.color_palette(n_colors=len(time_series_valid["Suspicious data"].unique()))))
print(palette)
palette["Valid observation"] = "grey"
markers = dict(zip(time_series_valid["Suspicious data"].unique(), ['d'] + ['o'] * 9))
sizes = dict(zip(time_series_valid["Suspicious data"].unique(), [5] + [30] * 9))
print(markers)
print(sizes)

# time_series_valid = time_series_valid[filter_time_valid] #time_series_valid["name"].unique()
for idx, station in enumerate(["Argentiere", "Saint-Sorlin", "Vallot"]):
    filter_station_valid = time_series_valid["name"] == station

    list_hue = list(time_series_valid["Suspicious data"][filter_station_valid].unique())
    list_hue.remove('Valid observation')
    unique_hue = ['Valid observation']
    unique_hue.extend(list_hue)
    """
    sizes=[5]
    other_sizes = [30] * (len(unique_hue) - 1)
    sizes.extend(other_sizes)
    markers=['d']
    other_markers = ['o'] * (len(unique_hue) - 1)
    markers.extend(other_markers)
    print(sizes)
    print(markers)
    """

    plt.figure(figsize=(20, 20))
    plt.subplot(211)
    sns.scatterplot(x=time_series_valid[filter_station_valid].index,
                    y='vw10m(m/s)',
                    data=time_series_valid[filter_station_valid],
                    hue="Suspicious data",
                    style="Suspicious data",
                    size="Suspicious data",
                    hue_order=unique_hue,
                    palette=palette,
                    sizes=sizes,
                    markers=markers,
                    edgecolor=None)
    """
    time_series_valid['wind_corrected'][filter_station_valid & filter_time_valid].plot(marker='x', label="wind speed [m/s]")
    #time_series_valid['wind_corrected'][filter_station_valid & filter_time_valid & (time_series_valid['qc']>0)].plot(marker='x', linestyle='', label="Suspicious data")
    time_series_valid['wind_corrected'][filter_station_valid & filter_unphysical_speed].plot(marker='x', label="Speed: unphysical", color=cmaplist[0])
    time_series_valid['wind_corrected'][filter_station_valid & filter_unphysical_direction].plot(marker='x', label="Direction: unphysical", color=cmaplist[1])
    time_series_valid['wind_corrected'][filter_station_valid & filter_excessive_MISS_speed].plot(marker='x', label="Speed: excessive miss", color='C3')
    time_series_valid['wind_corrected'][filter_station_valid & filter_excessive_MISS_direction].plot(marker='x', label="Direction: excessive miss ", color='C4')
    time_series_valid['wind_corrected'][filter_station_valid & filter_constant_sequence_speed].plot(marker='x', label="Speed: constant seq.", color='C5')
    time_series_valid['wind_corrected'][filter_station_valid & filter_constant_sequence_direction].plot(marker='x', label="Direction: constant seq", color='C1')
    time_series_valid['wind_corrected'][filter_station_valid & filter_high_variability_speed].plot(marker='x', label="Speed: high vari.", color='C1')
    time_series_valid['wind_corrected'][filter_station_valid & filter_bias_speed].plot(marker='x', label="Speed: bias", color='C1')
    time_series_valid['wind_corrected'][filter_station_valid & filter_isolated_recors_speed].plot(marker='x', label="Speed: isolated record", color='C1')
    time_series_valid['wind_corrected'][filter_station_valid & filter_isolated_recors_direction].plot(marker='x', label="Direction: Isolated record", color='C1')
    """
    plt.subplot(212)
    sns.scatterplot(x=time_series_valid[filter_station_valid].index,
                    y='winddir(deg)',
                    data=time_series_valid[filter_station_valid],
                    hue="Suspicious data",
                    style="Suspicious data",
                    size="Suspicious data",
                    hue_order=unique_hue,
                    palette=palette,
                    sizes=sizes,
                    markers=markers,
                    edgecolor=None)
    """
    time_series_valid['winddir(deg)'][filter_station_valid & filter_time_valid].plot(marker='x', linestyle='', label="wind direction [°]", color='C1')
    #time_series_valid['winddir(deg)'][filter_station_valid & filter_time_valid & (time_series_valid['qc']>0)].plot(marker='x', linestyle='', label="Suspicious data")
    time_series_valid['winddir(deg)'][filter_station_valid & filter_unphysical_speed].plot(marker='x', label="Speed: unphysical", color='C1')
    time_series_valid['winddir(deg)'][filter_station_valid & filter_unphysical_direction].plot(marker='x', label="Direction: unphysical", color='C1')
    time_series_valid['winddir(deg)'][filter_station_valid & filter_excessive_MISS_speed].plot(marker='x', label="Speed: excessive miss", color='C1')
    time_series_valid['winddir(deg)'][filter_station_valid & filter_excessive_MISS_direction].plot(marker='x', label="Direction: excessive miss ", color='C1')
    time_series_valid['winddir(deg)'][filter_station_valid & filter_constant_sequence_speed].plot(marker='x', label="Speed: constant seq.", color='C1')
    time_series_valid['winddir(deg)'][filter_station_valid & filter_constant_sequence_direction].plot(marker='x', label="Direction: constant seq", color='C1')
    time_series_valid['winddir(deg)'][filter_station_valid & filter_high_variability_speed].plot(marker='x', label="Speed: high vari.", color='C1')
    time_series_valid['winddir(deg)'][filter_station_valid & filter_bias_speed].plot(marker='x', label="Speed: bias", color='C1')
    time_series_valid['winddir(deg)'][filter_station_valid & filter_isolated_recors_speed].plot(marker='x', label="Speed: isolated record", color='C1')
    time_series_valid['winddir(deg)'][filter_station_valid & filter_isolated_recors_direction].plot(marker='x', label="Direction: Isolated record", color='C1')
    """
    plt.title(station)
    plt.tight_layout()

    save_figure(f"qc_{station}", prm, svg=False)
