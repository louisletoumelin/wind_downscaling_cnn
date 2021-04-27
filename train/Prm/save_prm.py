import numpy as np
import pandas as pd
import os


def prm_to_df(prm):
    """Convert prm to a pandas DataFrame"""
    values = list(prm.values())
    columns = list(prm.keys())
    df_prm = pd.DataFrame(columns=columns)
    for value, column in zip(values, columns):
        df_prm[column] = [value]
    return (df_prm)


def save_on_disk(prm):
    """Saving prm on a specific .csv file"""
    df_prm = prm_to_df(prm)
    info = '/training_results/' + prm['info']
    df_prm.to_csv(prm['output_dir'] + info + '/prm.csv')


def add_columns_to_df(df, columns):
    """Create new columns in a dataframe"""
    for column in columns:
        df[column] = np.nan

    return (df)


def update_training_record(prm):
    """Update a csv file containing all previous info about training"""
    # New prm
    new_prm = prm_to_df(prm)

    # Add new columns
    columns = ['RMSE_UV_mean', 'RMSE_UV_std', 'RMSE_UV_min', 'RMSE_UV_max',
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

    new_prm = add_columns_to_df(new_prm, columns)

    # Path
    out_dir = prm['output_dir'] + 'training_results/'
    path_file = out_dir + 'training_prm_record.csv'

    if os.path.isfile(path_file):

        # Load all_prm
        all_prm = pd.read_csv(path_file)

        # Append new prm to all_prm
        all_prm = all_prm.append(new_prm)

    else:
        all_prm = new_prm

    # Save all_prm
    all_prm.to_csv(path_file)
    print('\nprm saved in training_prm_record.csv\n')


def create_name_simu_and_info(index, prm):
    """Create name_simu and info key"""
    prm['name_simu'] = prm['name_simu'] + "_" + str(index)
    prm['info'] = 'date_' + prm['date'] + '_name_simu_' + prm['name_simu'] + '_model_' + prm['model']
    return (prm)
