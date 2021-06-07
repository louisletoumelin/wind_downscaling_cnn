import numpy as np
import pandas as pd
import time


def init_compile(prm):
    """ Make initial compile to init model and create random weights"""
    model = prm['model_func']
    model.compile(loss=prm['loss'],
                  optimizer=prm['optimizer_func'],
                  metrics=prm['metrics'])
    out_dir = prm['output_dir'] + 'training_results/' + prm['info']
    model.save_weights(out_dir + '/weights.h5')
    return (model)


def save_dict_norm(dict_norm, prm):
    """Save normalization features for each fold"""
    out_dir = prm['output_dir'] + 'training_results/' + prm['info']
    pd.DataFrame.from_dict(dict_norm).to_csv(out_dir + '/dict_norm.csv')


def general_compile(prm, model):
    """Compile model"""
    out_dir = prm['output_dir'] + 'training_results/' + prm['info']
    model.load_weights(out_dir + '/weights.h5')
    model.compile(loss=prm['loss'],
                  optimizer=prm['optimizer_func'],
                  metrics=prm['metrics'])
    return (model)


def print_data_dimension(TOPO_TRAIN, WIND_TRAIN, TOPO_VALID=None, WIND_VALID=None):
    """Display dimension of training data"""
    print("\n\nBefore reshaping\nTraining shape: ")
    print(TOPO_TRAIN.shape, WIND_TRAIN.shape)

    if (TOPO_VALID is not None) and (WIND_VALID is not None):
        print("Validation shape: ")
        print(TOPO_VALID.shape, WIND_VALID.shape)

    print(' \n\n')


def reshape_data(TOPO_TRAIN, WIND_TRAIN, TOPO_VALID=None, WIND_VALID=None, prm=None):
    """Reshape data for tensorflow"""
    x_train = TOPO_TRAIN.reshape((TOPO_TRAIN.shape[0], *prm['input_shape']))
    y_train = WIND_TRAIN.reshape((WIND_TRAIN.shape[0], *prm['output_shape']))

    if (TOPO_VALID is not None) and (WIND_VALID is not None):
        x_val = TOPO_VALID.reshape((TOPO_VALID.shape[0], *prm['input_shape']))
        y_val = WIND_VALID.reshape((WIND_VALID.shape[0], *prm['output_shape']))
    else:
        x_val, y_val = None, None
    return (x_train, y_train, x_val, y_val)


def print_reshaped_data_dimension(x_train, y_train, x_val=None, y_val=None):
    """Print dimensions after reshaping data"""
    print("\n\nAfter reshaping:\nTraining shape: ")
    print(x_train.shape, y_train.shape)

    if (x_val is not None) and (y_val is not None):
        print("Validation shape: ")
        print(np.shape(x_val), np.shape(y_val))

    print(' \n\n')


def normalize_training_features(x_train, x_val=None):
    """Normalize training and validation features"""
    train_mean, train_std = np.mean(x_train, axis=(1,2)), np.std(x_train)
    train_mean = train_mean.reshape((train_mean.shape[0], 1, 1, 1))
    print(f"Shape x_train before training: {x_train.shape}")
    x_train = (x_train - train_mean) / train_std
    print(f"Shape x_train after training: {x_train.shape}")
    if x_val is not None:
        train_mean = np.mean(x_val, axis=(1, 2))
        train_mean = train_mean.reshape((train_mean.shape[0], 1, 1, 1))
        x_val = (x_val - train_mean) / train_std
    return (x_train, x_val, np.mean(train_mean), train_std)


def dict_to_array(dictionary):
    """Convert a dict to an array of values"""
    list_to_return = []
    for key in dictionary.keys():
        list_to_return += dictionary[key]
    array_to_return = np.array(list_to_return)
    return(array_to_return)


def read_values(filename):
    """
    Reading ARPS (from Nora) .txt files

    Input:
           path file

    Output:
            1. Data description as described in the files (entête)
            2. Data (2D matrix)
    """
    # opening the file, type TextIOWrapper
    with open(filename, 'r') as fichier:
        entete = []

        # Reading info
        for _ in range(6):
            entete.append(fichier.readline())

        donnees = []
        # Data
        for string in entete:
            donnees = np.loadtxt(entete, dtype=int, usecols=[1])

        ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value = donnees

        # Data begins at line 7
        data = np.zeros((nrows, ncols), dtype=float)
        for i in range(nrows):
            ligne = fichier.readline()
            ligne = ligne.split()
            data[i] = [float(ligne[j]) for j in range(ncols)]

        return donnees, data


def filenames_to_array(df_all, input_dir, name):
    """ Reading data from filename stored in dataframe"""
    all_data = []
    for simu in range(len(df_all.values)):

        degree_folder = str(df_all['degree'].values[simu]) + 'degree/'

        # Topographies
        if name == 'topo_name':
            path = input_dir + 'dem/' + degree_folder
            donnees, data = read_values(path + df_all[name].values[simu])

        # Wind maps
        if name == 'wind_name':
            u_v_w = []

            # Wind components
            for index, wind_component in enumerate(['ucompwind/', 'vcompwind/', 'wcompwind/']):
                path = input_dir + 'Wind/U_V_W/' + wind_component + degree_folder
                uvw_simu = df_all[name].values[simu].split('(')[1].split(')')[0].split(', ')
                donnees, data = read_values(path + uvw_simu[index].split("'")[1])
                u_v_w.append(data.reshape(79 * 69))

            u_v_w = np.array(list(zip(*u_v_w)))
            assert np.shape(u_v_w) == (79 * 69, 3)
            data = u_v_w

        # Store data
        all_data.append(data)
    return (np.array(all_data))


def print_subexecution(step, level_of_subexecution):
    if level_of_subexecution == 1:
        print(' =====> '+step)
    if level_of_subexecution == 2:
        print('     =====> '+step)
    if level_of_subexecution >= 3:
        print('         =====> '+step)


def timer_step(func):
    """Timer decorator"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        rv = func(*args, **kwargs)
        finish = time.perf_counter()
        print(f'\nFinished in {round((finish - start) / 60, 2)} minute(s)')
        return(rv)

    return(wrapper)
