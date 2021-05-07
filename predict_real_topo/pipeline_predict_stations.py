from time import time as t

t_init = t()

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from line_profiler import LineProfiler


def round(t1, t2):  return (np.round(t2 - t1, 2))


from Processing import Processing
from Visualization import Visualization
from MNT import MNT
from NWP import NWP
from Observation import Observation
from Data_2D import Data_2D
from MidpointNormalize import MidpointNormalize
from Evaluation import Evaluation
from PRM_predict import create_prm, update_selected_path
from Utils import connect_GPU_to_horovod, select_range, check_save_and_load

"""
Stations
"""
"""
['BARCELONNETTE', 'DIGNE LES BAINS', 'RESTEFOND-NIVOSE',
       'LA MURE-ARGENS', 'ARVIEUX', 'PARPAILLON-NIVOSE', 'EMBRUN',
       'LA FAURIE', 'GAP', 'LA MEIJE-NIVOSE', 'COL AGNEL-NIVOSE',
       'GALIBIER-NIVOSE', 'ORCIERES-NIVOSE', 'RISTOLAS',
       'ST JEAN-ST-NICOLAS', 'TALLARD', "VILLAR D'ARENE",
       'VILLAR ST PANCRACE', 'ASCROS', 'PEIRA CAVA', 'PEONE',
       'MILLEFONTS-NIVOSE', 'CHAPELLE-EN-VER', 'LUS L CROIX HTE',
       'ST ROMAN-DIOIS', 'AIGLETON-NIVOSE', 'CREYS-MALVILLE',
       'LE GUA-NIVOSE', "ALPE-D'HUEZ", 'LA MURE- RADOME',
       'LES ECRINS-NIVOSE', 'GRENOBLE-ST GEOIRS', 'ST HILAIRE-NIVOSE',
       'ST-PIERRE-LES EGAUX', 'GRENOBLE - LVD', 'VILLARD-DE-LANS',
       'CHAMROUSSE', 'ALBERTVILLE JO', 'BONNEVAL-NIVOSE', 'MONT DU CHAT',
       'BELLECOTE-NIVOSE', 'GRANDE PAREI NIVOSE', 'FECLAZ_SAPC',
       'COL-DES-SAISIES', 'ALLANT-NIVOSE', 'LA MASSE',
       'ST MICHEL MAUR_SAPC', 'TIGNES_SAPC', 'LE CHEVRIL-NIVOSE',
       'LES ROCHILLES-NIVOSE', 'LE TOUR', 'AGUIL. DU MIDI',
       'AIGUILLES ROUGES-NIVOSE', 'LE GRAND-BORNAND', 'MEYTHET',
       'LE PLENAY', 'SEYNOD-AREA', 'Col du Lac Blanc', 'Col du Lautaret', 'Vallot', 'Saint-Sorlin', 'Argentiere']
"""

"""
To be modified
"""

GPU = False
Z0 = False
load_z0 = True
save_z0 = False
peak_valley = True
launch_predictions = False
select_date_time_serie = True
verbose = True
stations_to_predict = ['Col du Lac Blanc']

# Date to predict
day_begin = 1
month_begin = 1
year_begin = 2008

day_end = 30
month_end = 6
year_end = 2021

begin = str(year_begin) + "-" + str(month_begin) + "-" + str(day_begin)
end = str(year_end) + "-" + str(month_end) + "-" + str(day_end)

"""
Utils
"""


# Safety
load_z0, save_z0 = check_save_and_load(load_z0, save_z0)

# Initialize horovod and GPU
if GPU: connect_GPU_to_horovod()

# Create prm
prm = create_prm(GPU=GPU, Z0=Z0, end=end, month_prediction=True)

"""
MNT, NWP and observations
"""
# IGN
IGN = MNT(prm["topo_path"],
          name="IGN")

# AROME
AROME = NWP(prm["selected_path"],
            name="AROME",
            begin=begin,
            end=end,
            save_path=prm["save_path"],
            path_Z0_2018=prm["path_Z0_2018"],
            path_Z0_2019=prm["path_Z0_2019"],
            verbose=verbose,
            load_z0=load_z0,
            save=save_z0)

# BDclim
BDclim = Observation(prm["BDclim_stations_path"],
                     prm["BDclim_data_path"],
                     begin=begin,
                     end=end,
                     select_date_time_serie=select_date_time_serie,
                     path_vallot=prm["path_vallot"],
                     path_saint_sorlin=prm["path_saint_sorlin"],
                     path_argentiere=prm["path_argentiere"],
                     path_Dome_Lac_Blanc=prm["path_Dome_Lac_Blanc"],
                     path_Col_du_Lac_Blanc=prm["path_Col_du_Lac_Blanc"],
                     path_Muzelle_Lac_Blanc=prm["path_Muzelle_Lac_Blanc"])
#BDclim.qc()
# Compute nearest neighbor sif CPU, load them if GPU
if not (GPU):
    number_of_neighbors = 4
    BDclim.update_stations_with_KNN_from_NWP(number_of_neighbors, AROME)
    BDclim.update_stations_with_KNN_from_MNT_using_cKDTree(IGN)

"""
Processing, visualization and evaluation
"""

# Processing
p = Processing(obs=BDclim,
               mnt=IGN,
               nwp=AROME,
               model_path=prm['model_path'],
               GPU=GPU,
               data_path=prm['data_path'])

t1 = t()
if launch_predictions:

    if stations_to_predict == 'all':
        stations_to_predict = BDclim.stations["name"].values

    array_xr = p.predict_UV_with_CNN(stations_to_predict,
                                     verbose=True,
                                     Z0_cond=Z0,
                                     peak_valley=peak_valley,
                                     ideal_case=False)

t2 = t()
print(f'\nPredictions in {round(t1, t2)} seconds')

"""
lp = LineProfiler()
lp_wrapper = lp(p.predict_map_indexes)
lp_wrapper(year_0=2019, month_0=6, day_0=20, hour_0=15, year_1=2019, month_1=6, day_1=20, hour_1=15, dx=20_000, dy=25_000)
lp.print_stats()
"""

# Visualization
v = Visualization(p)
#v.qc_plot_last_flagged(stations=['Vallot', 'Argentiere'])
# v.plot_predictions_2D(array_xr, ['Col du Lac Blanc'])
# v.plot_predictions_3D(array_xr, ['Col du Lac Blanc'])
# v.plot_comparison_topography_MNT_NWP(station_name='Col du Lac Blanc', new_figure=False)

# Evaluation
if launch_predictions: e = Evaluation(v, array_xr)
# e.plot_time_serie(array_xr, 'Col du Lac Blanc', year=year_begin)

t_end = t()
print(f"\n All prediction in  {round(t_init, t_end) / 60} minutes")
"""
import pandas as pd
import matplotlib.pyplot as plt
for station in ['Vallot']:
    time_series = BDclim.time_series
    time_serie_station = time_series[time_series["name"] == station]
    wind_speed='vw10m(m/s)'
    wind_direction='winddir(deg)'
    # Select wind speed
    wind = time_serie_station[wind_speed]
    # Daily wind
    wind = wind.resample('1D').mean()
    result = wind.copy(deep=True)*0
    # No outliers
    rol_mean = wind.rolling('15D').mean()
    rol_std = wind.rolling('15D').std()
    no_outliers = wind.copy(deep=True)
    no_outliers[(no_outliers > rol_mean + 2*rol_std) | (no_outliers < rol_mean - 2*rol_std)] = np.nan
    # Groupby days
    seasonal = no_outliers.groupby([(no_outliers.index.month),(no_outliers.index.day)]).mean()
    seasonal.index = pd.date_range(start='2000-01-01', freq='D', periods=366)
    # Rolling mean
    seasonal_rolling = seasonal.rolling('15D').mean()
    # Interpolate missing values
    seasonal_rolling = seasonal_rolling.interpolate()
    # Divide two datasets by seasonal
    for month in range(1, 13):
        for day in range(1, 32):
            filter_wind = (wind.index.month == month) & (wind.index.day == day)
            filter_no_outlier = (no_outliers.index.month == month) & (no_outliers.index.day == day)
            filter_seasonal = (seasonal_rolling.index.month == month) &  (seasonal_rolling.index.day == day)
            try:
                wind[filter_wind] = wind[filter_wind] / seasonal_rolling[filter_seasonal].values[0]
            except IndexError:
                wind[filter_wind] = wind / 1
            try:
                no_outliers[filter_wind] = no_outliers[filter_no_outlier] / seasonal_rolling[filter_seasonal].values[0]
            except IndexError:
                no_outliers[filter_wind] = no_outliers / 1
    # Rolling
    wind_rolling = wind.rolling('15D').mean()
    no_outliers_rolling = no_outliers.rolling('15D').mean()
    # Wind speed    
    P95 = no_outliers.rolling('15D').quantile(0.95)
    P25 = no_outliers.rolling('15D').quantile(0.25)
    P75 = no_outliers.rolling('15D').quantile(0.75)
    criteria_mean = (wind_rolling > (P95+3.7*(P75-P25))) | (wind_rolling<0.5)
    # Standard deviation
    standard_deviation = np.abs(wind-wind.mean())
    standard_deviation_rolling = standard_deviation.rolling('15D').mean()
    standard_deviation_no_outliers = np.abs(no_outliers - no_outliers.mean())
    P95 = standard_deviation_no_outliers.rolling('15D').quantile(0.95)
    P25 = standard_deviation_no_outliers.rolling('15D').quantile(0.25)
    P75 = standard_deviation_no_outliers.rolling('15D').quantile(0.75)
    criteria_std = (standard_deviation_rolling > (P95+7.5*(P75-P25))) | (standard_deviation_rolling<(0.044))
    # Coefficient of variation
    coeff_variation = standard_deviation / wind_rolling.mean()
    coeff_variation_rolling = coeff_variation.rolling('15D').mean()
    coeff_variation_no_outliers = standard_deviation_no_outliers / no_outliers.mean()
    P95 = coeff_variation_no_outliers.rolling('15D').quantile(0.95)
    P25 = coeff_variation_no_outliers.rolling('15D').quantile(0.25)
    P75 = coeff_variation_no_outliers.rolling('15D').quantile(0.75)
    criteria_coeff_var = (coeff_variation_rolling > (P95+7.5*(P75-P25))) | (coeff_variation_rolling<0.22/1.5)
    result[criteria_mean | criteria_std | criteria_coeff_var] = 1
    plt.figure()
    ax = plt.gca()
    time_serie_station[wind_speed].plot()
    time_serie_station[wind_speed].rolling('30D').mean().plot()
    plt.figure()
    wind.plot()
    #wind[result == 1].plot(marker='x', linestyle='')
    
    
    result[result == 0] = np.nan
    labels = result.diff().ne(0).cumsum()
    result_3 = (labels.map(labels.value_counts()) >= 12).astype(int)
    try:
        wind[(wind.index.isin(result_3.index)) & (result_3==1)].plot(marker='x', linestyle='')
        result_4 = result_3.resample('1H').nearest()
        time_serie_station[wind_speed][result_4==1].plot(ax=ax, marker='x', linestyle='')
    except:
        pass
"""