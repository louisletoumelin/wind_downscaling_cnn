from sklearn.model_selection import train_test_split
import numpy as np

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


def train_valid(df_all):
    """Split train and validation"""
    train, validation = train_test_split(df_all['topo_name'][df_all['group'] == 'train'], test_size=0.1, shuffle = True)
    for simu in train.values:
        df_all['group'][df_all['topo_name'] == simu] = 'train'
    for simu in validation.values:
        df_all['group'][df_all['topo_name'] == simu] = 'validation'
    return(df_all)


def print_execution(step):
    print('\n\n_______________\n\n')
    print(f'\n\n {step} created\n\n')
    print('\n\n_______________\n\n')


def print_subexecution(step, level_of_subexecution):
    if level_of_subexecution == 1:
        print(' =====> '+step)
    if level_of_subexecution == 2:
        print('     =====> '+step)
    if level_of_subexecution >= 3:
        print('         =====> '+step)
