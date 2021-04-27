import numpy as np
import pandas as pd
# import tensorflow_probability as tfp
from tensorflow.keras import backend as K

'''
Keras Metrics
'''


# Custom Metrics : NRMSE
def nrmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred))) / (K.max(y_pred) - K.min(y_pred))


# Custom Metrics : RMSE
def root_mse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


# Custom Metrics : coeff de variation
def coeff_variation(y_true, y_pred):
    return K.std(y_pred) / K.mean(y_pred)


# Custom Metrics : coeff de variation
def coeff_variation(y_true, y_pred):
    return (K.std(y_pred) / K.mean(y_pred))


'''
Pandas, numpy and other Metrics
'''


# Custom Metrics : Pearson correlation coeff
def pearson_correlation(y_true, y_pred):
    # return(tf.linalg.trace(tfp.stats.correlation(y_pred, y_true))/3)
    return (pd.concat([pd.DataFrame(y_true), pd.DataFrame(y_pred)], axis=1).corr().iloc[0, 1])


def spatial_RMSE(pred, true):
    return (np.mean((pred - true) ** 2, axis=(1, 2)) ** 0.5)


def mean_bias(pred, true):
    return(np.mean(pred, axis=(1, 2)) - np.mean(true, axis=(1, 2)))


'''
Wind operations
'''


def horizontal_wind_speed(wind_speed_map):
    dim = len(np.shape(wind_speed_map))
    if dim == 4:
        U = wind_speed_map[:, :, :, 0]
        V = wind_speed_map[:, :, :, 1]
    if dim == 3:
        U = wind_speed_map[:, :, 0]
        V = wind_speed_map[:, :, 1]
    return ((U ** 2 + V ** 2) ** (0.5))


def vertical_wind_speed(wind_speed_map):
    dim = len(np.shape(wind_speed_map))
    if dim == 4:
        U = wind_speed_map[:, :, :, 0]
        V = wind_speed_map[:, :, :, 1]
        W = wind_speed_map[:, :, :, 2]
    if dim == 3:
        U = wind_speed_map[:, :, 0]
        V = wind_speed_map[:, :, 1]
        W = wind_speed_map[:, :, 2]
    return ((U ** 2 + V ** 2 + W ** 2) ** (0.5))


'''
Modify Prm
'''


def create_prm_metrics(prm):
    """
    Select the desired metric

    Input: str, ex: 'mae' or 'root_mse'
    Output: str or function, ex: 'mae' or  root_mse
    """
    prm['metrics'] = []
    for metric in prm['list_metrics']:

        if metric == 'mae':
            prm['metrics'].append(metric)
        if metric == 'root_mse':
            prm['metrics'].append(root_mse)

    return (prm)


def create_dependencies(prm):
    """Create dependencies in prm to call load_model"""
    prm['dependencies'] = {}
    if 'root_mse' in prm['list_metrics']:
        prm['dependencies']['root_mse'] = root_mse
    if 'nrmse' in prm['list_metrics']:
        prm['dependencies']['root_mse'] = nrmse
    if 'pearson_correlation' in prm['list_metrics']:
        prm['dependencies']['root_mse'] = pearson_correlation
    return (prm)