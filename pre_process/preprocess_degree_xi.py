import pandas as pd
import os

from utils_preprocess import train_valid

def create_specific_folder_by_degree_xi(degree_or_xi, output_dir, degree_or_xi_str):
    """Create subfolder for each degree or each xi"""
    if degree_or_xi_str == 'degree':
        parent_folder = 'degree_/'
    if degree_or_xi_str == 'xi':
        parent_folder = 'xi_/'
    newpath = output_dir + parent_folder + degree_or_xi_str + str(degree_or_xi)
    if not os.path.exists(newpath):
        os.makedirs(newpath)


def train_test_degree_xi(degree_or_xi_to_exclude, df_all, degree_or_xi_str):
    """Split train and test"""
    df_all['group'] = 0
    df_all['group'][df_all[degree_or_xi_str] == degree_or_xi_to_exclude] = 'test'
    df_all['group'][df_all[degree_or_xi_str] != degree_or_xi_to_exclude] = 'train'
    return (df_all)


def train_validation_test_split_degree_xi(df_all, input_dir):
    for degree_to_exclude in [5, 10, 13, 16, 20]:
        df_all = train_test_degree_xi(degree_to_exclude, df_all, 'degree')
        df_all = train_valid(df_all)

        # Create subfolder
        create_specific_folder_by_degree_xi(degree_to_exclude, input_dir, 'degree')

        # Save file
        output = input_dir + '/degree_' + f"/degree{degree_to_exclude}"
        df_all.to_csv(output + f'/df_all_degree_excluded_{degree_to_exclude}.csv')

    for xi_to_exclude in [1000, 200, 300, 400, 500, 600, 700, 800, 900]:
        df_all = train_test_degree_xi(xi_to_exclude, df_all, 'xi')
        df_all = train_valid(df_all)

        # Create subfolder
        create_specific_folder_by_degree_xi(xi_to_exclude, input_dir, 'xi')

        # Save file
        output = input_dir + 'xi_' + f"/xi{xi_to_exclude}"
        df_all.to_csv(output + f'/df_all_xi_excluded_{xi_to_exclude}.csv')


def preprocess_degree_xi(input_dir):
    """Preprocessing function for xi and degree"""
    # Read file
    df_all = pd.read_csv(input_dir + 'fold/' + 'df_all.csv')

    # Check no group already exists
    try: df_all = df_all.drop('group', axis=1)
    except: pass

    # Split data and save file
    train_validation_test_split_degree_xi(df_all, input_dir)