"""
Created on Tue Jul 28 12:19:11 2020

@author: Louis Le Toumelin
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Flatten, BatchNormalization, LeakyReLU, LSTM, UpSampling2D, Conv2DTranspose, ZeroPadding2D, Cropping2D, UpSampling1D
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K

date='14_12'

GPU = True
if GPU:
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))
  
    

# Training data dimensions
n_rows = 79
n_col = 69

# Data format (channels_last)
input_shape = (n_rows, n_col, 1)
output_shape = (n_rows, n_col, 3)

input_dir = "//home/mrmn/letoumelinl/train"
output_dir = "//scratch/mrmn/letoumelinl/ARPS/"

# Nombre de filtres par couche de convolution
n_conv_features = 32
# Fonction de perte mse = mean squared error
loss="mse"


def root_mse(y_true, y_pred):
    return(K.sqrt(K.mean(K.square(y_true - y_pred))))



def build_model(input_shape):
    model = Sequential()

    model.add(ZeroPadding2D(padding=((0, 1), (0, 1)), input_shape=input_shape))

    # CONVOLUTION
    model.add(Conv2D(n_conv_features, (5, 5), activation='relu', padding="same"))
    model.add(Conv2D(n_conv_features, (5, 5), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(2*n_conv_features, (5, 5), activation='relu', padding="same"))
    model.add(Conv2D(2*n_conv_features, (5, 5), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(4*n_conv_features, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(4*n_conv_features, (3, 3), activation='relu', padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(8*n_conv_features, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(8*n_conv_features, (3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # DECONVOLUTION
    model.add(Conv2DTranspose(8*n_conv_features, (3, 3), activation='relu', padding="same"))
    model.add(Conv2DTranspose(8*n_conv_features, (3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(ZeroPadding2D(padding=((0, 0), (0, 1))))

    model.add(Conv2DTranspose(4*n_conv_features, (3, 3), activation='relu', padding="same"))
    model.add(Conv2DTranspose(4*n_conv_features, (3, 3), activation='relu', padding="same"))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(ZeroPadding2D(padding=((0, 0), (0, 1))))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2DTranspose(2*n_conv_features, (5, 5), activation='relu', padding="same"))
    model.add(Conv2DTranspose(2*n_conv_features, (5, 5), activation='relu', padding="same"))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2DTranspose(n_conv_features, (5, 5), activation='relu', padding="same"))
    model.add(Conv2DTranspose(n_conv_features, (5, 5), activation='relu', padding="same"))
    model.add(Dropout(0.25))

    # Matrice produite
    model.add(Conv2DTranspose(3, (5, 5), activation='linear', padding="same"))
    model.add(Cropping2D(cropping=((0, 1), (0, 1))))

    # Résumé du réseau
    model.summary()
    
    
    return(model)



def train_model(model, root_mse, input_dir, txt):
    
    model.compile(loss=loss, optimizer=RMSprop(lr=0.001, decay=0.0001), metrics=['mae', root_mse])
    

    model.save_weights('weights.h5')

    for fold in range(8):
        
        print('Fold' + str(fold))

        model.load_weights('weights.h5')

        model.compile(loss=loss, optimizer=RMSprop(lr=0.001, decay=0.0001), metrics=['mae', root_mse])

        # Chemin d'accès aux données (préalablement traitées pour numpy)
        filepath = output_dir + "fold{}/".format(fold)
        
        # Chargement des données
        print("LOADING DATA")

        if txt:
            TOPO_TRAIN = np.loadtxt(filepath + "train/topo.txt", dtype=np.float32)
            WIND_TRAIN = np.loadtxt(filepath + "train/wind.txt", dtype=np.float32)
            TOPO_VALID = np.loadtxt(filepath + "validation/topo.txt", dtype=np.float32)
            WIND_VALID = np.loadtxt(filepath + "validation/wind.txt", dtype=np.float32)
        else:
            TOPO_TRAIN = np.load(filepath + "train/topo.npy")
            WIND_TRAIN = np.load(filepath + "train/wind.npy")
            TOPO_VALID = np.load(filepath + "validation/topo.npy")
            WIND_VALID = np.load(filepath + "validation/wind.npy")


        #Affichage des dimensions
        print("Before reshaping")
        print("Training shape: ")
        print(TOPO_TRAIN.shape)
        print(WIND_TRAIN.shape)
        print("Validation shape: ")
        print(TOPO_VALID.shape)
        print(WIND_VALID.shape)

        # Redimensionnement des données brutes (x*x) pour format keras
        x_train = TOPO_TRAIN.reshape((TOPO_TRAIN.shape[0], * input_shape))
        y_train = WIND_TRAIN.reshape((WIND_TRAIN.shape[0], * output_shape))
        x_val = TOPO_VALID.reshape((TOPO_VALID.shape[0], * input_shape))
        y_val = WIND_VALID.reshape((WIND_VALID.shape[0], * output_shape))
        
        print("\n\nAfter reshaping:\n")
        print("Training shape: ")
        print(x_train.shape)
        print(y_train.shape)
        
        print("Validation shape: ")
        print(np.shape(x_val))
        print(np.shape(y_val))

        # Normalisation des features
        train_mean, train_std = np.mean(x_train), np.std(x_train)
        x_train = (x_train - train_mean)/train_std
        x_val = (x_val - train_mean)/train_std

        # Définition des callbacks utilisés
        filepath="checkpoint.hdf5"
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-10, verbose=1)
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
        callback_list = [checkpoint, reduce_lr]

        # ENTRAINEMENT DU RESEAU
        history = model.fit(x_train, y_train, batch_size=32, epochs=150, verbose=0, validation_data=(x_val, y_val), callbacks=callback_list)
        
        # Sauvegarde du modele
        model.save(output_dir+'model_'+date+'_fold_{}.h5'.format(fold))
        np.save(output_dir+'model_'+date+'_fold_{}_history.npy'.format(fold), history.history)
    return(model, history)
    

if GPU:
    start = time.perf_counter()
    model = build_model(input_shape)
    trained_model, history = train_model(model, root_mse, input_dir, False)
    finish = time.perf_counter()
    print(f'\nFinished in {round((finish-start)/60, 2)} minute(s)')
