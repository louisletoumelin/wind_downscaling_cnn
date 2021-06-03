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
    from numba import jit, guvectorize, vectorize, prange, float64, float32, int32, int64
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

    def laplacian_idx(self, mnt, idx_x, idx_y, dx, verbose=True, librairie='numpy', helbig=True):
        """
        Discrete laplacian on a regular grid
        """

        if librairie == 'numba' and _numba:

            mnt = change_dtype_if_required(mnt, np.float32)
            idx_x = change_dtype_if_required(idx_x, np.int32)
            idx_y = change_dtype_if_required(idx_y, np.int32)

            laplacian = self._laplacian_numba_idx(mnt, idx_x, idx_y, dx, helbig=helbig)
            librairie = librairie

        else:
            laplacian = self._laplacian_numpy_idx(mnt, idx_x, idx_y, dx, helbig=helbig)
            librairie = "numpy"

        if verbose: print(f"__Laplacian calculation using {librairie}")
        return(laplacian)

    @staticmethod
    def _laplacian_numpy_idx(mnt, idx_x, idx_y, dx, helbig=True):
        a = np.float32((mnt[idx_y-1, idx_x] + mnt[idx_y+1, idx_x] + mnt[idx_y, idx_x-1] + mnt[idx_y, idx_x+1] - 4*mnt[idx_y, idx_x])/dx**2)
        c = np.float32(dx/4) if helbig else 1
        return (a * c)

    @staticmethod
    def _laplacian_loop_numpy_1D_helbig(mnt, idx_x, idx_y, dx):
        laplacian = np.empty(idx_x.shape, np.float32)
        for i in range(idx_x.shape[0]):
            a = (mnt[idx_x[i] - 1, idx_y[i]] + mnt[idx_x[i] + 1, idx_y[i]] + mnt[idx_x[i], idx_y[i] - 1] + mnt[
                idx_x[i], idx_y[i] + 1] - 4 * mnt[idx_x[i], idx_y[i]]) / dx ** 2
            c = dx / 4
            laplacian[i] = a * c
        return (laplacian)

    @staticmethod
    def _laplacian_loop_numpy_1D(mnt, idx_x, idx_y, dx):
        laplacian = np.empty(idx_x.shape, np.float32)
        for i in range(idx_x.shape[0]):
            a = (mnt[idx_x[i] - 1, idx_y[i]] + mnt[idx_x[i] + 1, idx_y[i]] + mnt[idx_x[i], idx_y[i] - 1] + mnt[
                idx_x[i], idx_y[i] + 1] - 4 * mnt[idx_x[i], idx_y[i]]) / dx ** 2
            laplacian[i] = a
        return (laplacian)

    @staticmethod
    def _laplacian_loop_numpy_2D_helbig(mnt, idx_x, idx_y, dx):
        laplacian = np.empty(idx_x.shape, np.float32)
        for j in range(idx_x.shape[0]):
            for i in range(idx_x.shape[1]):
                a = (mnt[idx_y[j, i] - 1, idx_x[j, i]] + mnt[idx_y[j, i] + 1, idx_x[j, i]] + mnt[idx_y[j, i], idx_x[j, i] - 1] + mnt[
                    idx_y[j, i], idx_x[j, i] + 1] - 4 * mnt[idx_y[j, i], idx_x[j, i]]) / dx ** 2
                c = dx / 4
                laplacian[j, i] = a * c
        return (laplacian)

    @staticmethod
    def _laplacian_loop_numpy_2D(mnt, idx_x, idx_y, dx):
        laplacian = np.empty(idx_x.shape, np.float32)
        for j in range(idx_x.shape[0]):
            for i in range(idx_x.shape[1]):
                a = (mnt[idx_y[j, i] - 1, idx_x[j, i]] + mnt[idx_y[j, i] + 1, idx_x[j, i]] + mnt[idx_y[j, i], idx_x[j, i] - 1] + mnt[
                    idx_y[j, i], idx_x[j, i] + 1] - 4 * mnt[idx_y[j, i], idx_x[j, i]]) / dx ** 2
                laplacian[j, i] = a
        return (laplacian)

    def _laplacian_numba_idx(self, mnt, idx_x, idx_y, dx, helbig=True):

        if helbig:
            laplacian_1D = self._laplacian_loop_numpy_1D_helbig
            laplacian_2D = self._laplacian_loop_numpy_2D_helbig
        else:
            laplacian_1D = self._laplacian_loop_numpy_1D
            laplacian_2D = self._laplacian_loop_numpy_2D

        if idx_x.ndim == 1:
            lapl_vect = jit([(float32[:, :], int32[:], int32[:], int64)], nopython=True)(laplacian_1D)

        if idx_x.ndim == 2:
            lapl_vect = jit([(float32[:, :], int32[:, :], int32[:, :], int64)], nopython=True)(laplacian_2D)

        return(lapl_vect(mnt, idx_x, idx_y, dx))

    def laplacian_map(self, mnt, dx, librairie="numpy", helbig=True):

        # Pad mnt to compute laplacian on edges
        mnt_padded = np.pad(mnt, (1, 1), "edge").astype(np.float32)
        shape = mnt_padded.shape

        # Use meshgrid to create indexes with mnt size and use numpy broadcasting when selecting indexes
        xx, yy = np.array(np.meshgrid(list(range(shape[1])), list(range(shape[0])))).astype(np.int32)

        # Compute laplacian on indexes using an index for every grid point (meshgrid)
        laplacian = self.laplacian_idx(mnt_padded, xx[1:-1, 1:-1], yy[1:-1, 1:-1], dx, librairie=librairie, helbig=helbig)

        return(laplacian)

    def mu_helbig_map(self, mnt, dx):
        """
        Adapted from I. Gouttevin

        From Helbig et al. 2017
        """
        mu = np.sqrt(np.sum(np.array(np.gradient(mnt, dx))**2, axis=0)/2)
        return(mu)

    def mu_helbig_idx(self, mnt, dx, idx_x, idx_y):
        mu = self.mu_helbig_map(mnt, dx)
        return(mu[idx_y, idx_x])

    def mu_helbig_average(self, mnt, dx, idx_x, idx_y):
        mu = self.mu_helbig_map(mnt, dx)

        y_left = np.intp(idx_y-79//2)
        y_right = np.intp(idx_y+79//2)
        x_left = np.intp(idx_x-69//2)
        x_right = np.intp(idx_x+69//2)

        return(np.mean(mu[y_left:y_right, x_left:x_right]))

    def x_dsc_topo_helbig_map(self, mnt, dx):

        a = 17.0393
        b = 0.737
        c = 1.0234
        d = 0.3794
        e = 1.9821

        term_1 = 1 - a*self.laplacian_map(mnt, dx)/(1+a*self.laplacian_map(mnt, dx, helbig=True)**b)
        term_2 = c / (1+d*self.mu_helbig_map(mnt, dx)**2)

        return(term_1*term_2)

    def x_dsc_topo_helbig_idx(self, mnt, idx_x, idx_y, dx):

        a = 17.0393
        b = 0.737
        c = 1.0234
        d = 0.3794
        e = 1.9821

        term_1 = 1 - a*self.laplacian_idx(mnt, idx_x, idx_y, dx)/(1+a*self.laplacian_idx(mnt, idx_x, idx_y, dx, helbig=True)**b)
        term_2 = c / (1+d*self.mu_helbig_idx(mnt, dx)**2)

        return(term_1*term_2)

    def sigma_helbig(self, mnt, idx_x, idx_y):

        y_left = np.intp(idx_y-79//2)
        y_right = np.intp(idx_y+79//2)
        x_left = np.intp(idx_x-69//2)
        x_right = np.intp(idx_x+69//2)

        sigma = np.std(mnt[y_left:y_right, x_left:x_right])
        return(sigma)

    def xsi_helbig_idx(self, mnt, idx_x, idx_y):

        y_left = np.int32(idx_y-79//2)
        y_right = np.int32(idx_y+79//2)
        x_left = np.int32(idx_x-69//2)
        x_right = np.int32(idx_x+69//2)

        #print(np.array([mnt[i1:i2, j1:j2] for (x, y) in zip(idx_x, idx_y) for (i1, j1), (i2, j2) in zip(x, y)]).shape)
        xsi = np.sqrt(2) * np.std(mnt[y_left:y_right, x_left:x_right]) / self.mu_helbig_idx(mnt, idx_x, idx_y)
        return(xsi)

    def x_sgp_topo_helbig_idx(self, mnt, idx_x, idx_y, dx, L=2_000):

        a = 3.354688
        b = 1.998767
        c = 0.20286
        d = 5.951

        mu = self.mu_helbig_idx(mnt, dx, idx_x, idx_y)
        xsi = self.xsi_helbig_idx(mnt, idx_x, idx_y)

        x = 1 - (1 - (1/(1+a*mu**b))**c)*np.exp(-d*(L/xsi)**(-2))
        return(x)

