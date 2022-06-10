import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os
import gc
import random
from pprint import pprint

from . import learning_utils
import sys
sys.path.append('..')
print(sys)
import Models
from Metrics import metrics
from Prm import choose_initializer
from Prm import choose_optimizer

'''
Functions
'''


def save_folds_weights_and_history_for_fold(model, history, fold, prm):
    """Save model weights and history for each fold"""
    out_dir = prm['output_dir'] + 'training_results/' + prm['info'] + f'/fold{fold}'
    tf.keras.models.save_model(model, out_dir)
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
    reduce_lr = ReduceLROnPlateau(monitor='val_mae',
                                  factor=prm['ROP_factor'],
                                  patience=prm['ROP_patience'],
                                  min_lr=prm['ROP_min_lr'],
                                  verbose=1)
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_mae',
                                   patience=prm["early_stopping_patience"],
                                   min_delta=prm["early_stopping_min_delta"],
                                   mode="min",
                                   restore_best_weights=True)
    callback_list = [early_stopping, checkpoint, reduce_lr]

    return callback_list


def reset_seeds():
    np.random.seed(1)
    random.seed(2)
    if tf.__version__[0] == '2':
        tf.random.set_seed(3)
    else:
        tf.set_random_seed(3)
    print("Random seed reset")


'''
Training function
'''


@learning_utils.timer_step
def train_model(prm):
    """Train a CNN with parameters specified in prm"""
    #model = learning_utils.init_compile(prm)
    #initial_weights = model.get_weights()
    dict_norm = {}
    dict_epochs = {}
    for fold in range(10):

        print("\n\n_______________")
        print('_____'+'Fold' + str(fold)+'_____')
        print("_______________\n\n")

        # Compile
        #model = learning_utils.reset_weights_and_compile(prm, model, initial_weights)
        reset_seeds()
        model = Models.choose_model.create_prm_model(prm)['model_func']
        model.compile(loss=prm['loss'],
                      optimizer=prm['optimizer_func'],
                      metrics=prm['metrics'])

        # Load data
        TOPO_TRAIN, WIND_TRAIN, TOPO_VALID, WIND_VALID = load_data_by_fold(fold, prm)

        # Displaying dimensions
        learning_utils.print_data_dimension(TOPO_TRAIN, WIND_TRAIN, TOPO_VALID, WIND_VALID)

        # Reshaping for tensorflow
        x_train, y_train, x_val, y_val = learning_utils.reshape_data(TOPO_TRAIN, WIND_TRAIN,
                                                                     TOPO_VALID, WIND_VALID, prm=prm)

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
        assert model.trainable
        history = model.fit(x_train,
                            y_train,
                            batch_size=prm['batch_size'],
                            epochs=prm['epochs'],
                            verbose=1,
                            validation_data=(x_val, y_val),
                            callbacks=callback_list)

        # Saving model weights and history
        save_folds_weights_and_history_for_fold(model, history, fold, prm)

        del model
        del prm["model_func"]
        del prm["dependencies"]
        del prm["metrics"]
        del prm['optimizer_func']
        del prm['initializer_func']
        del callback_list
        del TOPO_TRAIN
        del WIND_TRAIN
        del TOPO_VALID
        del WIND_VALID
        del x_train
        del y_train
        del x_val
        del y_val
        gc.collect()
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        prm = metrics.create_prm_metrics(prm)
        prm = metrics.create_dependencies(prm)
        prm = choose_optimizer.create_prm_optimizer(prm)
        prm = choose_initializer.create_prm_initializer(prm)

        pprint(history.history)
        hist = history.history['val_mae']
        n_epochs_best = np.argmin(hist) + 1
        dict_epochs[str(fold)] = {'nb_epochs': n_epochs_best}

    learning_utils.save_dict_norm(dict_norm, prm)
    learning_utils.save_dict_epochs(dict_epochs, prm)

    return None, history
