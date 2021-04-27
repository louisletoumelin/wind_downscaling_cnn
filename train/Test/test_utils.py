import numpy as np
import pandas as pd


from Type_of_training import learning_utils
from Metrics import metrics


def load_test_data(df_all, prm):
    """Load test data as numpy arrays"""

    # Select test data
    df_test = df_all[df_all['group'] == 'test']

    # Loading data

    TOPO_TEST = learning_utils.filenames_to_array(df_test, prm['output_dir'], 'topo_name')
    TOPO_TEST = TOPO_TEST.reshape((len(TOPO_TEST), 79, 69, 1))
    WIND_TEST = learning_utils.filenames_to_array(df_test, prm['output_dir'], 'wind_name')
    WIND_TEST = WIND_TEST.reshape((len(WIND_TEST), 79, 69, 3))
    return (TOPO_TEST, WIND_TEST)


def correlation_coefficient_test(true_UV, prediction_UV):
    # Correlation coefficient
    true_list = true_UV.reshape((len(true_UV), 79 * 69))
    pred_list = prediction_UV.reshape((len(prediction_UV), 79 * 69))
    list_corr_coef = [pd.concat([pd.DataFrame(true), pd.DataFrame(pred)], axis=1).corr().iloc[0, 1] for true, pred in
                      zip(true_list, pred_list)]

    # Mean and std
    corr_coeff_UV_mean = np.mean(list_corr_coef)
    corr_coeff_UV_std = np.std(list_corr_coef)
    corr_coeff_UV_min = np.min(list_corr_coef)
    corr_coeff_UV_max = np.max(list_corr_coef)

    return (list_corr_coef, corr_coeff_UV_mean, corr_coeff_UV_std, corr_coeff_UV_min, corr_coeff_UV_max)


def RMSE_test(true_UV, prediction_UV):
    list_RMSE = metrics.spatial_RMSE(prediction_UV, true_UV)
    return (list_RMSE, list_RMSE.mean(), list_RMSE.std(), list_RMSE.min(), list_RMSE.max())


def mean_bias_test(prediction_UV, true_UV):
    list_bias = metrics.mean_bias(prediction_UV, true_UV)
    return (list_bias, list_bias.mean(), list_bias.std(), list_bias.min(), list_bias.max())


def calculate_all_test_statistics(WIND_PRED, WIND_TEST, list_or_values):
    """ Caluclate statistics (list of all stats or mean and std of stats)"""

    # Calculate UV and UVW
    prediction_UV = metrics.horizontal_wind_speed(WIND_PRED)
    prediction_UVW = metrics.vertical_wind_speed(WIND_PRED)
    true_UV = metrics.horizontal_wind_speed(WIND_TEST)
    true_UVW = metrics.vertical_wind_speed(WIND_TEST)

    # RMSE
    list_RMSE_UV, RMSE_UV_mean, RMSE_UV_std, RMSE_UV_min, RMSE_UV_max = RMSE_test(true_UV, prediction_UV)
    list_RMSE_UVW, RMSE_UVW_mean, RMSE_UVW_std, RMSE_UVW_min, RMSE_UVW_max = RMSE_test(true_UVW, prediction_UVW)
    list_RMSE_U, RMSE_U_mean, RMSE_U_std, RMSE_U_min, RMSE_U_max = RMSE_test(WIND_TEST[:, :, :, 0],
                                                                             WIND_PRED[:, :, :, 0])
    list_RMSE_V, RMSE_V_mean, RMSE_V_std, RMSE_V_min, RMSE_V_max = RMSE_test(WIND_TEST[:, :, :, 1],
                                                                             WIND_PRED[:, :, :, 1])
    list_RMSE_W, RMSE_W_mean, RMSE_W_std, RMSE_W_min, RMSE_W_max = RMSE_test(WIND_TEST[:, :, :, 2],
                                                                             WIND_PRED[:, :, :, 2])

    # Correlation coefficient
    list_corr_coeff_UV, corr_coeff_UV_mean, corr_coeff_UV_std, corr_coeff_UV_min, corr_coeff_UV_max = \
        correlation_coefficient_test(true_UV, prediction_UV)
    list_corr_coeff_UVW, corr_coeff_UVW_mean, corr_coeff_UV_std, corr_coeff_UVW_min, corr_coeff_UV_max = \
        correlation_coefficient_test(true_UVW, prediction_UVW)
    list_corr_coeff_U, corr_coeff_U_mean, corr_coeff_U_std, corr_coeff_U_min, corr_coeff_U_max = \
        correlation_coefficient_test(WIND_TEST[:, :, :, 0],WIND_PRED[:, :, :, 0])
    list_corr_coeff_V, corr_coeff_V_mean, corr_coeff_V_std, corr_coeff_V_min, corr_coeff_V_max = \
        correlation_coefficient_test(WIND_TEST[:, :, :, 1],WIND_PRED[:, :, :, 1])
    list_corr_coeff_W, corr_coeff_W_mean, corr_coeff_W_std, corr_coeff_W_min, corr_coeff_W_max = \
        correlation_coefficient_test(WIND_TEST[:, :, :, 2],WIND_PRED[:, :, :, 2])

    # Mean bias
    list_bias_UV, bias_UV_mean, bias_UV_std, bias_UV_min, bias_UV_max = mean_bias_test(prediction_UV, true_UV)
    list_bias_UVW, bias_UVW_mean, bias_UVW_std, bias_UVW_min, bias_UVW_max = mean_bias_test(prediction_UVW, true_UVW)
    list_bias_U, bias_U_mean, bias_U_std, bias_U_min, bias_U_max = mean_bias_test(WIND_PRED[:, :, :, 0],
                                                                                  WIND_TEST[:, :, :, 0])
    list_bias_V, bias_V_mean, bias_V_std, bias_V_min, bias_V_max = mean_bias_test(WIND_PRED[:, :, :, 1],
                                                                                  WIND_TEST[:, :, :, 1])
    list_bias_W, bias_W_mean, bias_W_std, bias_W_min, bias_W_max = mean_bias_test(WIND_PRED[:, :, :, 2],
                                                                                  WIND_TEST[:, :, :, 2])

    # Results
    list_to_add = [list_RMSE_UV, list_RMSE_UVW, list_RMSE_U, list_RMSE_V, list_RMSE_W,
                   list_corr_coeff_UV, list_corr_coeff_UVW, list_corr_coeff_U,
                   list_corr_coeff_V, list_corr_coeff_W,
                   list_bias_UV, list_bias_UVW, list_bias_U, list_bias_V, list_bias_W]

    values = [RMSE_UV_mean, RMSE_UV_std, RMSE_UV_min, RMSE_UV_max,
              RMSE_UVW_mean, RMSE_UVW_std, RMSE_UVW_min, RMSE_UVW_max,
              RMSE_U_mean, RMSE_U_std, RMSE_U_min, RMSE_U_max,
              RMSE_V_mean, RMSE_V_std, RMSE_V_min, RMSE_V_max,
              RMSE_W_mean, RMSE_W_std, RMSE_W_min, RMSE_W_max,
              corr_coeff_UV_mean, corr_coeff_UV_std, corr_coeff_UV_min, corr_coeff_UV_max,
              corr_coeff_UVW_mean, corr_coeff_UV_std, corr_coeff_UVW_min, corr_coeff_UV_max,
              corr_coeff_U_mean, corr_coeff_U_std, corr_coeff_U_min, corr_coeff_U_max,
              corr_coeff_V_mean, corr_coeff_V_std, corr_coeff_V_min, corr_coeff_V_max,
              corr_coeff_W_mean, corr_coeff_W_std, corr_coeff_W_min, corr_coeff_W_max,
              bias_UV_mean, bias_UV_std, bias_UV_min, bias_UV_max,
              bias_UVW_mean, bias_UVW_std, bias_UVW_min, bias_UVW_max,
              bias_U_mean, bias_U_std, bias_U_min, bias_U_max,
              bias_V_mean, bias_V_std, bias_V_min, bias_V_max,
              bias_W_mean, bias_W_std, bias_W_min, bias_W_max]

    if list_or_values == 'list':
        return (list_to_add)
    if list_or_values == 'values':
        return (np.array(values))


"""
def update_df_all(df_all, list_to_add):
    var_name = ['RMSE_UV', 'RMSE_UVW', 'RMSE_U', 'RMSE_V', 'RMSE_W',
                'cc_UV', 'cc_UVW', 'cc_U', 'cc_V', 'cc_W',
                'bias_UV', 'bias_UVW', 'bias_U', 'bias_V', 'bias_W']

    for variable_name, values in zip(var_name, list_to_add):
        df_all[variable_name] = np.nan
        df_all[variable_name][df_all['group'] == 'test'] = values

    return (df_all)
"""


def update_training_prm_records(list_to_add, prm):
    path = prm['output_dir'] + 'training_results/'
    training_prm_record = pd.read_csv(path + 'training_prm_record.csv')
    var_name = ['RMSE_UV_mean', 'RMSE_UV_std', 'RMSE_UV_min', 'RMSE_UV_max',
                'RMSE_UVW_mean', 'RMSE_UVW_std', 'RMSE_UVW_min', 'RMSE_UVW_max',
                'RMSE_U_mean', 'RMSE_U_std', 'RMSE_U_min', 'RMSE_U_max',
                'RMSE_V_mean', 'RMSE_V_std', 'RMSE_V_min', 'RMSE_V_max',
                'RMSE_W_mean', 'RMSE_W_std', 'RMSE_W_min', 'RMSE_W_max',
                'corr_coeff_UV_mean', 'corr_coeff_UV_std', 'corr_coeff_UV_min', 'corr_coeff_UV_max',
                'corr_coeff_UVW_mean', 'corr_coeff_UV_std', 'corr_coeff_UVW_min', 'corr_coeff_UV_max',
                'corr_coeff_U_mean', 'corr_coeff_U_std', 'corr_coeff_U_min', 'corr_coeff_U_max',
                'corr_coeff_V_mean', 'corr_coeff_V_std', 'corr_coeff_V_min', 'corr_coeff_V_max',
                'corr_coeff_W_mean', 'corr_coeff_W_std', 'corr_coeff_W_min', 'corr_coeff_W_max',
                'bias_UV_mean', 'bias_UV_std', 'bias_UV_min', 'bias_UV_max',
                'bias_UVW_mean', 'bias_UVW_std', 'bias_UVW_min', 'bias_UVW_max',
                'bias_U_mean', 'bias_U_std', 'bias_U_min', 'bias_U_max',
                'bias_V_mean', 'bias_V_std', 'bias_V_min', 'bias_V_max',
                'bias_W_mean', 'bias_W_std', 'bias_W_min', 'bias_W_max']

    print(len(list_to_add))
    assert len(var_name) == len(list_to_add)

    for variable, value in zip(var_name, list_to_add):
        training_prm_record[variable][training_prm_record['info'] == prm['info']] = value
    training_prm_record.to_csv(path + 'training_prm_record.csv')
