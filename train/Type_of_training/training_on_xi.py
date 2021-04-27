import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os

from . import learning_utils

'''
Functions
'''


def save_weights_and_history_for_xi(model, history, xi, prm):
    """Save model weights and history fo each xi"""
    out_dir = prm['output_dir'] + 'training_results/' + prm['info'] + f'/xi{xi}'
    model.save(out_dir + '/xi_{}.h5'.format(xi))
    np.save(out_dir + '/xi_{}_history.npy'.format(xi), history.history)


def create_subfolder(xi, prm):
    """Create a folder for each xi"""
    newpath = prm['output_dir'] + 'training_results/' + prm['info'] + f'/xi{xi}'
    if not os.path.exists(newpath):
        os.makedirs(newpath)


def load_data_by_xi(xi, prm):
    # Data filepath
    filepath = prm['output_dir'] + f"xi_/xi{xi}/"

    print("\nLOADING DATA")

    # Loading data
    df_all = pd.read_csv(filepath + f'df_all_xi_excluded_{xi}.csv')

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

    print('DONE\n')
    return (TOPO_TRAIN, WIND_TRAIN, TOPO_VALID, WIND_VALID)


def callbacks_for_xi(xi, prm):
    filepath = prm['output_dir'] + 'training_results/' + prm['info'] + f'/xi{xi}'
    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=prm['ROP_factor'],
                                  patience=prm['ROP_patience'],
                                  min_lr=prm['ROP_min_lr'],
                                  verbose=1)
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callback_list = [checkpoint, reduce_lr]
    return(callback_list)


'''
Training function
'''


@learning_utils.timer_step
def train_model(prm):
    """Train a CNN with parameters specified in prm"""
    model = learning_utils.init_compile(prm)
    dict_norm = {}

    for xi in [1000, 200, 300, 400, 500, 600, 700, 800, 900]:
        print('\nXi' + str(xi))

        # Compile
        model = learning_utils.general_compile(prm, model)

        # Load data
        TOPO_TRAIN, WIND_TRAIN, TOPO_VALID, WIND_VALID = load_data_by_xi(xi, prm)

        # Displaying dimensions
        learning_utils.print_data_dimension(TOPO_TRAIN, WIND_TRAIN, TOPO_VALID, WIND_VALID)


        # Reshaping for tensorflow
        x_train, y_train, x_val, y_val = learning_utils.reshape_data(TOPO_TRAIN, WIND_TRAIN,
                                                                     TOPO_VALID, WIND_VALID,
                                                                     prm)

        # Display new dimensions
        learning_utils.print_reshaped_data_dimension(x_train, y_train, x_val, y_val)

        # Feature normalization
        x_train, x_val, train_mean, train_std = learning_utils.normalize_training_features(x_train, x_val)

        # Store mean and std use in normalization
        dict_norm[str(xi)] = {'train_mean': train_mean, 'train_std': train_std}

        # Create subfolder
        create_subfolder(xi, prm)

        # Callbacks
        callback_list = callbacks_for_xi(xi, prm)

        # Training
        history = model.fit(x_train,
                            y_train,
                            batch_size=prm['batch_size'],
                            epochs=prm['epochs'],
                            verbose=1,
                            validation_data=(x_val, y_val),
                            callbacks=callback_list)

        # Saving model weights and history
        save_weights_and_history_for_xi(model, history, xi, prm)

    learning_utils.save_dict_norm(dict_norm, prm)

    return(model, history)
