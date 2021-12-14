import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os

from . import learning_utils

'''
Functions
'''


def save_folds_weights_and_history_for_fold(model, history, fold, prm):
    """Save model weights and history for each fold"""
    out_dir = prm['output_dir'] + 'training_results/' + prm['info'] + f'/fold{fold}'
    model.save(out_dir + '/fold_{}.h5'.format(fold))
    tf.keras.models.save_model(model, out_dir + '/fold{}'.format(fold))
    np.save(out_dir + '/fold_{}_history.npy'.format(fold), history.history)


def create_subfolder(fold, prm):
    """Create a folder for each fold"""
    newpath = prm['output_dir'] + 'training_results/' + prm['info'] + f'/fold{fold}'
    if not os.path.exists(newpath):
        os.makedirs(newpath)


def load_data_by_fold(fold, prm):
    # Data filepath
    filepath = prm['output_dir'] + f"fold/fold{fold}/"

    print("\nLOADING DATA")

    # Loading data
    df_all = pd.read_csv(filepath + f'df_all_{fold}.csv')

    # Training data
    df_train = df_all[df_all['group'] == 'train']
    TOPO_TRAIN = learning_utils.filenames_to_array(df_train, prm['output_dir'], 'topo_name')
    TOPO_TRAIN = TOPO_TRAIN.reshape(len(TOPO_TRAIN), 79 * 69)
    WIND_TRAIN = learning_utils.filenames_to_array(df_train, prm['output_dir'], 'wind_name')
    WIND_TRAIN = WIND_TRAIN.reshape(len(WIND_TRAIN), 79 * 69 * 3)

    # Validation data
    df_valid = df_all[df_all['group'] == 'validation']
    TOPO_VALID = learning_utils.filenames_to_array(df_valid, prm['output_dir'], 'topo_name')
    TOPO_VALID = TOPO_VALID.reshape(len(TOPO_VALID), 79 * 69)
    WIND_VALID = learning_utils.filenames_to_array(df_valid, prm['output_dir'], 'wind_name')
    WIND_VALID = WIND_VALID.reshape(len(WIND_VALID), 79 * 69 * 3)

    if prm["additional_flat_topo"]:

        # Parameters
        input_wind_speed = 3

        # Training data
        additional_topo_training = np.ones((15, 79, 69))
        additional_wind_training = np.ones((15, 79, 69, 3))
        additional_wind_training[:, :, :, :2] = additional_wind_training[:, :, :, :2] * input_wind_speed
        additional_wind_training[:, :, :, 2] = additional_wind_training[:, :, :, 2] * 0

        for index, elevation in enumerate(range(0, 3750, 250)):
            additional_topo_training[index, :, :] = elevation * additional_topo_training[index, :, :]

        additional_topo_training.reshape((15, 79 * 69))
        additional_wind_training = additional_wind_training.reshape((15, 79 * 69 * 3))
        TOPO_TRAIN = np.concatenate((TOPO_TRAIN, additional_topo_training), axis=0)
        WIND_TRAIN = np.concatenate((WIND_TRAIN, additional_wind_training), axis=0)

        # Training data
        additional_topo_valid = np.ones((4, 79, 69))
        additional_wind_valid = np.ones((4, 79, 69, 3))
        additional_wind_valid[:, :, :, :2] = additional_wind_valid[:, :, :, :2] * input_wind_speed
        additional_wind_valid[:, :, :, 2] = additional_wind_valid[:, :, :, 2] * 0

        for index, elevation in enumerate(range(0, 3750, 250)):
            additional_topo_valid[index, :, :] = elevation * additional_topo_valid[index, :, :]

        additional_topo_valid = additional_topo_valid.reshape((4, 79 * 69))
        additional_wind_valid = additional_wind_valid.reshape((4, 79 * 69 * 3))
        TOPO_TRAIN = np.concatenate((TOPO_TRAIN, additional_topo_valid), axis=0)
        WIND_TRAIN = np.concatenate((WIND_TRAIN, additional_wind_valid), axis=0)



        print("Flat topo added to training data")

    print('DONE\n')
    return (TOPO_TRAIN, WIND_TRAIN, TOPO_VALID, WIND_VALID)


def callbacks_for_folds(fold, prm):
    filepath = prm['output_dir'] + 'training_results/' + prm['info'] + f'/fold{fold}'
    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=prm['ROP_factor'],
                                  patience=prm['ROP_patience'],
                                  min_lr=prm['ROP_min_lr'],
                                  verbose=1)
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callback_list = [checkpoint, reduce_lr]
    return (callback_list)


'''
Training function
'''


@learning_utils.timer_step
def train_model(prm):
    """Train a CNN with parameters specified in prm"""
    model = learning_utils.init_compile(prm)
    dict_norm = {}
    for fold in range(10):
        print('Fold' + str(fold))

        # Compile
        model = learning_utils.general_compile(prm, model)

        # Load data
        TOPO_TRAIN, WIND_TRAIN, TOPO_VALID, WIND_VALID = load_data_by_fold(fold, prm)

        # Displaying dimensions
        learning_utils.print_data_dimension(TOPO_TRAIN, WIND_TRAIN, TOPO_VALID, WIND_VALID)

        # Reshaping for tensorflow
        x_train, y_train, x_val, y_val = learning_utils.reshape_data(TOPO_TRAIN, WIND_TRAIN,
                                                                     TOPO_VALID, WIND_VALID,
                                                                     prm=prm)

        # Display new dimensions
        learning_utils.print_reshaped_data_dimension(x_train, y_train, x_val, y_val)

        # Feature normalization
        x_train, x_val, train_mean, train_std = learning_utils.normalize_training_features(x_train, x_val)

        # Store mean and std use in normalization
        dict_norm[str(fold)] = {'train_mean': train_mean, 'train_std': train_std}

        # Create subfolder
        create_subfolder(fold, prm)

        # Callbacks
        callback_list = callbacks_for_folds(fold, prm)

        # Training
        history = model.fit(x_train,
                            y_train,
                            batch_size=prm['batch_size'],
                            epochs=prm['epochs'],
                            verbose=1,
                            validation_data=(x_val, y_val),
                            callbacks=callback_list)

        # Saving model weights and history
        save_folds_weights_and_history_for_fold(model, history, fold, prm)

    learning_utils.save_dict_norm(dict_norm, prm)

    return (model, history)
