import numpy as np
import pandas as pd
import os
import shutil
import sys
from tensorflow.keras.models import load_model

from . import test_utils

sys.path.append('//home/mrmn/letoumelinl/train/Metrics/')
from metrics import *

"""
Functions
"""


def get_path_by_type(index, prm):
    if prm['type_of_training'] == 'fold':
        path_to_data = prm['output_dir'] + f'fold/fold{index}/'
        df_all_name = f'df_all_{index}.csv'
        path_to_experience = prm['output_dir'] + 'training_results/' + prm['info']
        path_to_sub_experience = prm['output_dir'] + 'training_results/' + prm['info'] + f'/fold{index}/'
        model_weights = f'fold_{index}.h5'

    if prm['type_of_training'] == 'class':
        path_to_data = prm['output_dir'] + f'class_nb_/class_nb{index}/'
        df_all_name = f'df_all_class_excluded_{index}.csv'
        path_to_experience = prm['output_dir'] + 'training_results/' + prm['info']
        path_to_sub_experience = prm['output_dir'] + 'training_results/' + prm['info'] + f'/class_nb{index}/'
        model_weights = f'class_nb_{index}.h5'

    if prm['type_of_training'] in ['degree', 'xi']:
        type = prm['type_of_training']
        path_to_data = prm['output_dir'] + f'{type}_/{type}{index}/'
        df_all_name = f'df_all_{type}_excluded_{index}.csv'
        path_to_experience = prm['output_dir'] + 'training_results/' + prm['info']
        path_to_sub_experience = prm['output_dir'] + 'training_results/' + prm['info'] + f'/{type}{index}/'
        model_weights = f'{type}_{index}.h5'
    return (path_to_data, df_all_name, path_to_experience, path_to_sub_experience, model_weights)


def load_df_all(index, prm):
    """Load e.g. df_all_0.csv"""
    path, file_name, _, _, _ = get_path_by_type(index, prm)
    df_all = pd.read_csv(path + file_name)
    return (df_all)


def load_CNN(index, prm):
    _, _, _, path, model_weights = get_path_by_type(index, prm)
    model = load_model(path + model_weights, custom_objects=prm['dependencies'])
    return (model)


def normalize_test_data(index, TOPO_TEST, prm):
    _, _, path, _, _ = get_path_by_type(index, prm)
    dict_norm = pd.read_csv(path + '/dict_norm.csv')
    train_mean, train_std = dict_norm[str(index)]
    TOPO_TEST_norm = (TOPO_TEST - train_mean) / train_std
    return (TOPO_TEST_norm)


def save_test_predictions(index, WIND_PRED, prm):
    _, _, _, path, _ = get_path_by_type(index, prm)
    np.save(path + f'WIND_PRED_{index}.npy', WIND_PRED)


def save_updated_df_all(index, df_all, prm):
    _, _, _, path, _ = get_path_by_type(index, prm)
    df_all.to_csv(path + f'df_all_{index}.csv')


def update_with_stats_and_save_df_all(index, prm, df_all, list_to_add):
    var_name = ['RMSE_UV', 'RMSE_UVW', 'RMSE_U', 'RMSE_V', 'RMSE_W',
                'cc_UV', 'cc_UVW', 'cc_U', 'cc_V', 'cc_W',
                'bias_UV', 'bias_UVW', 'bias_U', 'bias_V', 'bias_W']

    for variable_name, values in zip(var_name, list_to_add):
        df_all[variable_name] = np.nan
        df_all[variable_name][df_all['group'] == 'test'] = values

    _, df_all_name, _, path_to_sub_experience, _ = get_path_by_type(index, prm)

    df_all.to_csv(path_to_sub_experience+df_all_name)


def delete_variables_assets_folders(index, prm):
    _, _, _, path, _ = get_path_by_type(index, prm)
    path_to_assets = path + 'assets/'
    path_to_variables = path + 'variables/'
    if os.path.exists(path_to_assets):
        shutil.rmtree(path_to_assets)
    if os.path.exists(path_to_variables):
        shutil.rmtree(path_to_variables)


def predict_CNN(index, prm):
    # Load df_all by fold/class_nb/degree/xi
    df_all = load_df_all(index, prm)

    # Load data
    TOPO_TEST, WIND_TEST = test_utils.load_test_data(df_all, prm)

    # Load model
    model = load_CNN(index, prm)

    # Normalize data
    TOPO_TEST_norm = normalize_test_data(index, TOPO_TEST, prm)

    # Predict
    WIND_PRED = model.predict(TOPO_TEST_norm)

    # Save predictions
    save_test_predictions(index, WIND_PRED, prm)

    # Add statistics to df_all by fold
    list_to_add = test_utils.calculate_all_test_statistics(WIND_PRED, WIND_TEST, 'list')

    update_with_stats_and_save_df_all(index, prm, df_all, list_to_add)

    values_to_add = test_utils.calculate_all_test_statistics(WIND_PRED, WIND_TEST, 'values')

    return (values_to_add)


def block_iteration(prm):
    if prm['type_of_training'] == 'fold':
        statistics = np.zeros(60)
        for fold in range(10):
            values_to_add = predict_CNN(fold, prm)
            statistics += values_to_add
            delete_variables_assets_folders(fold, prm)
        statistics = statistics / (fold + 1)

    if prm['type_of_training'] == 'class':
        statistics = np.zeros(60)
        for class_nb in range(2):
            values_to_add = predict_CNN(class_nb, prm)
            statistics += values_to_add
            delete_variables_assets_folders(class_nb, prm)
        statistics = statistics / (class_nb + 1)

    if prm['type_of_training'] == 'degree':
        statistics = np.zeros(60)
        for index, degree in enumerate([5, 10, 13, 16, 20]):
            values_to_add = predict_CNN(degree, prm)
            statistics += values_to_add
            delete_variables_assets_folders(degree, prm)
        statistics = statistics / (index + 1)

    if prm['type_of_training'] == 'xi':
        statistics = np.zeros(60)
        for index, xi in enumerate([1000, 200, 300, 400, 500, 600, 700, 800, 900]):
            values_to_add = predict_CNN(xi, prm)
            statistics += values_to_add
            delete_variables_assets_folders(xi, prm)
        statistics = statistics / (index + 1)
    return (statistics)


"""
Pipeline
"""


def predict_test(prm):

    # Iterate over folds/class_nb/degree/xi
    statistics = block_iteration(prm)

    # update the training_prm_record.csv
    test_utils.update_training_prm_records(statistics, prm)