import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os

from . import learning_utils

'''
Functions
'''


def save_folds_weights_and_history_for_all(model, history, prm):
    """Save model weights and history for all"""
    out_dir = prm['output_dir'] + 'training_results/' + prm['info']
    model.save(out_dir + '/model_weights.h5')
    np.save(out_dir + '/history.npy', history.history)


def load_data_all(prm):
    """Load all data for training"""
    # Data filepath
    filepath = prm['output_dir'] + f"fold/fold{0}/"

    print("\nLOADING DATA")

    # Loading data
    df_all = pd.read_csv(filepath + f'df_all_{0}.csv')

    # Training data
    df_train = df_all
    TOPO_TRAIN = learning_utils.filenames_to_array(df_train, prm['output_dir'], 'topo_name')
    TOPO_TRAIN = TOPO_TRAIN.reshape(len(TOPO_TRAIN), 79 * 69)
    WIND_TRAIN = learning_utils.filenames_to_array(df_train, prm['output_dir'], 'wind_name')
    WIND_TRAIN = WIND_TRAIN.reshape(len(WIND_TRAIN), 79 * 69 * 3)

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

        additional_topo_training = additional_topo_training.reshape((15, 79*69))
        additional_wind_training = additional_wind_training.reshape((15, 79 * 69 * 3))
        TOPO_TRAIN = np.concatenate((TOPO_TRAIN, additional_topo_training), axis=0)
        WIND_TRAIN = np.concatenate((WIND_TRAIN, additional_wind_training), axis=0)

        print("Flat topo added to training data")

    print('DONE\n')
    return (TOPO_TRAIN, WIND_TRAIN)


def callbacks_for_all(prm):
    """Save callbacks"""
    filepath = prm['output_dir'] + 'training_results/' + prm['info']
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

    # Load data
    TOPO_TRAIN, WIND_TRAIN = load_data_all(prm)

    # Displaying dimensions
    learning_utils.print_data_dimension(TOPO_TRAIN, WIND_TRAIN, None, None)

    # Reshaping for tensorflow
    x_train, y_train, x_val, y_val = learning_utils.reshape_data(TOPO_TRAIN, WIND_TRAIN,
                                                                 None, None,
                                                                 prm=prm)

    # Display new dimensions
    learning_utils.print_reshaped_data_dimension(x_train, y_train, None, None)

    # Feature normalization
    x_train, _, train_mean, train_std = learning_utils.normalize_training_features(x_train, None)

    # Store mean and std use in normalization
    dict_norm["0"] = {'train_mean': train_mean, 'train_std': train_std}

    # Callbacks
    callback_list = callbacks_for_all(prm)

    # Training
    history = model.fit(x_train,
                        y_train,
                        batch_size=prm['batch_size'],
                        epochs=prm['epochs'],
                        verbose=1,
                        callbacks=callback_list)

    # Saving model weights and history
    save_folds_weights_and_history_for_all(model, history, prm)

    learning_utils.save_dict_norm(dict_norm, prm)

    return (model, history)
