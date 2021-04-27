from sklearn.model_selection import train_test_split
import numpy as np
import os


"""
All functions
"""


def select_data_folds(df_all):
    """Train/valid/test selection"""
    training_data, validation_data, test_data = [], [], []
    for degree_xi in df_all['degree_xi'].unique():
        # Select data
        simu_names = df_all['topo_name'][df_all['degree_xi'] == degree_xi].values

        # Shuffle data
        np.random.shuffle(simu_names)

        # Partition data
        names_train_valid, name_test = train_test_split(simu_names, test_size=0.1, shuffle=True)
        names_train, names_valid = train_test_split(names_train_valid, test_size=0.1, shuffle=True)

        # Store data
        training_data.extend(list(names_train))
        validation_data.extend(list(names_valid))
        test_data.extend(list(name_test))

    return (training_data, validation_data, test_data)


def update_df_all(df_all, training_data, validation_data, test_data):
    """Add a column "group" to df_all indicating train/validation or test"""
    df_all['group'] = 0
    for simu_name in training_data:
        df_all['group'][df_all['topo_name'] == simu_name] = 'train'
    for simu_name in validation_data:
        df_all['group'][df_all['topo_name'] == simu_name] = 'validation'
    for simu_name in test_data:
        df_all['group'][df_all['topo_name'] == simu_name] = 'test'

    return (df_all)


def create_specific_folder_by_fold(fold, output_dir):
    """Create subfolder for each fold"""
    newpath = output_dir + '/fold' + f"/fold{fold}"
    if not os.path.exists(newpath):
        os.makedirs(newpath)


"""
Preprocessing
"""


def preprocess_folds(df_all, output_dir):
    """Preprocessing function for folds"""

    for fold in range(10):

        # Split data
        df_all_copy = df_all.copy()
        training_data, validation_data, test_data = select_data_folds(df_all_copy)

        # Update dataframe with train/test information
        df_all_copy = update_df_all(df_all_copy, training_data, validation_data, test_data)

        # Create a new folder
        create_specific_folder_by_fold(fold, output_dir)

        # Save df_all
        if fold == 0:
            df_all.to_csv(output_dir + 'fold/' + "df_all.csv")

        # Save dataframe
        df_all_copy.to_csv(output_dir + 'fold/' + f"fold{fold}/df_all_{fold}.csv")