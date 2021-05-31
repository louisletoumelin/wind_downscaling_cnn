import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import datetime
import time
import os
import concurrent.futures

from Utils import change_dtype_if_required

try:
    from numba import jit, prange, float64, float32, int32, int64

    _numba = True
except ModuleNotFoundError:
    _numba = False

try:
    import numexpr as ne

    _numexpr = True
except ModuleNotFoundError:
    _numexpr = False

try:
    import dask

    _dask = True
except ModuleNotFoundError:
    _dask = False


class Topo_utils:
    _numexpr = _numexpr
    _numba = _numba
    _dask = _dask

    def __init__(self):
        pass

    def normalize_topo(self, topo_HD, mean, std, dtype=np.float32, librairie='num', verbose=True):
        """
        Normalize a topography with mean and std.

        Parameters
        ----------
        topo_HD : array
        mean : array
        std : array
        dtype: numpy dtype, optional
        tensorflow: boolean, optional
            Use tensorflow array (Default: False)

        Returns
        -------
        Standardized topography : array
        """
        if verbose: print(f"__Normalize done with mean {len(mean)} means and std")

        if librairie == 'tensorflow':
            topo_HD = tf.constant(topo_HD, dtype=tf.float32)
            mean = tf.constant(mean, dtype=tf.float32)
            std = tf.constant(std, dtype=tf.float32)
            result = tf.subtract(topo_HD, mean)
            result = tf.divide(result, result)
            return (result)

        if librairie == 'num':
            topo_HD = np.array(topo_HD, dtype=dtype)
            if self._numexpr:
                return (ne.evaluate("(topo_HD - mean) / std"))
            else:
                return ((topo_HD - mean) / std)

    def mean_peak_valley(self, topo, verbose=True):
        """
        2 * std(topography)

        Mean peak valley height

        Parameters
        ----------
        topo : ndarray
            topography

        Returns
        -------
        peak_valley_height : ndarray
            Mean peak valley height
        """
        peak_valley_height = 2 * np.nanstd(topo)
        if verbose: print("__Mean peak valley computed")
        return(peak_valley_height.astype(np.float32))