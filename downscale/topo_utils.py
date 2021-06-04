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

    def laplacian_map(self, mnt, dx, librairie="numpy", helbig=True, verbose=True):

        # Pad mnt to compute laplacian on edges
        mnt_padded = np.pad(mnt, (1, 1), "edge").astype(np.float32)
        shape = mnt_padded.shape

        # Use meshgrid to create indexes with mnt size and use numpy broadcasting when selecting indexes
        xx, yy = np.array(np.meshgrid(list(range(shape[1])), list(range(shape[0])))).astype(np.int32)

        # Compute laplacian on indexes using an index for every grid point (meshgrid)
        laplacian = self.laplacian_idx(mnt_padded, xx[1:-1, 1:-1], yy[1:-1, 1:-1], dx, librairie=librairie, helbig=helbig)

        laplacian = change_dtype_if_required(laplacian, np.float32)
        if verbose: print(f"__Laplacian map calculated. Shape: {laplacian.shape}")

        return(laplacian)

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

        laplacian = change_dtype_if_required(laplacian, np.float32)
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

        result = lapl_vect(mnt, idx_x, idx_y, dx)
        result = change_dtype_if_required(result, np.float32)
        return(result)

    def mu_helbig_map(self, mnt, dx):
        """
        Adapted from I. Gouttevin

        From Helbig et al. 2017
        """
        mu = np.sqrt(np.sum(np.array(np.gradient(mnt, dx))**2, axis=0)/2)
        mu = change_dtype_if_required(mu, np.float32)
        return(mu)

    def mu_helbig_idx(self, mnt, dx, idx_x, idx_y):
        mu = self.mu_helbig_map(mnt, dx)
        mu = change_dtype_if_required(mu, np.float32)
        return(mu[idx_y, idx_x])


class Sgp_helbig(Topo_utils):

    def __init__(self):
        super().__init__()

    def mu_helbig_average(self, mnt, dx, idx_x, idx_y, reduce_mnt=True, nb_pixels_x=100, nb_pixels_y=100, verbose=True):

        mu = self.mu_helbig_map(mnt, dx)

        y_left = np.intp(idx_y-79//2)
        y_right = np.intp(idx_y+79//2)
        x_left = np.intp(idx_x-69//2)
        x_right = np.intp(idx_x+69//2)

        if reduce_mnt:

            small_idx_y = idx_y[nb_pixels_y:-nb_pixels_y:, nb_pixels_x:-nb_pixels_x]
            small_idx_x = idx_x[nb_pixels_y:-nb_pixels_y:, nb_pixels_x:-nb_pixels_x]

            y_left = np.intp(small_idx_y - 79 // 2).reshape(small_idx_y.shape[0] * small_idx_y.shape[1])
            y_right = np.intp(small_idx_y + 79 // 2).reshape(small_idx_y.shape[0] * small_idx_y.shape[1])
            x_left = np.intp(small_idx_x - 69 // 2).reshape(small_idx_x.shape[0] * small_idx_x.shape[1])
            x_right = np.intp(small_idx_x + 69 // 2).reshape(small_idx_x.shape[0] * small_idx_x.shape[1])

        mu_flat = np.array([np.mean(mu[i1:j1, i2:j2]) for i1, j1, i2, j2 in zip(y_left, y_right, x_left, x_right)])

        mu = mu_flat.reshape((small_idx_y.shape[0], small_idx_y.shape[1]))

        mu = change_dtype_if_required(mu, np.float32)
        if verbose: print(f"__Subgrid: computed average mu. Output shape: {mu.shape}")

        return(mu)

    @staticmethod
    @jit([float32[:](float32[:, :], int32[:], int32[:], int32[:], int32[:])], cache=True, nopython=True)
    def std_slicing_numba(array, y_left, y_right, x_left, x_right):
        result = np.empty(y_left.shape)
        for index, (i1, j1, i2, j2) in enumerate(zip(y_left, y_right, x_left, x_right)):
            result[index] = np.std(array[i1:j1, i2:j2])
        return (result.astype(np.float32))

    def xsi_helbig_map(self, mnt, mu, idx_x, idx_y, reduce_mnt=True, nb_pixels_x=100, nb_pixels_y=100, librairie="numba", verbose=True):

        y_left = np.int32(idx_y-79//2)
        y_right = np.int32(idx_y+79//2)
        x_left = np.int32(idx_x-69//2)
        x_right = np.int32(idx_x+69//2)

        if reduce_mnt:

            small_idx_y = idx_y[nb_pixels_y:-nb_pixels_y:, nb_pixels_x:-nb_pixels_x]
            small_idx_x = idx_x[nb_pixels_y:-nb_pixels_y:, nb_pixels_x:-nb_pixels_x]

            y_left = np.intp(small_idx_y - 79 // 2).reshape(small_idx_y.shape[0] * small_idx_y.shape[1])
            y_right = np.intp(small_idx_y + 79 // 2).reshape(small_idx_y.shape[0] * small_idx_y.shape[1])
            x_left = np.intp(small_idx_x - 69 // 2).reshape(small_idx_x.shape[0] * small_idx_x.shape[1])
            x_right = np.intp(small_idx_x + 69 // 2).reshape(small_idx_x.shape[0] * small_idx_x.shape[1])

        if librairie == "numba" and _numba:

            mnt = change_dtype_if_required(mnt, np.float32)
            y_left = change_dtype_if_required(y_left, np.int32)
            y_right = change_dtype_if_required(y_right, np.int32)
            x_left = change_dtype_if_required(x_left, np.int32)
            x_right = change_dtype_if_required(x_right, np.int32)

            std_flat = self.std_slicing_numba(mnt, y_left, y_right, x_left, x_right)
            librairie = "numba"
        else:
            std_flat = np.array([np.std(mnt[i1:j1, i2:j2]) for i1, j1, i2, j2 in zip(y_left, y_right, x_left, x_right)])
            librairie = "numpy"

        std = std_flat.reshape((small_idx_y.shape[0], small_idx_y.shape[1]))
        if verbose: print(f"__Subgrid: computed average std. Output shape: {std.shape}. Librairie: {librairie}")

        xsi = np.sqrt(2) * std / mu
        xsi = change_dtype_if_required(xsi, np.float32)
        if verbose: print(f"__Subgrid: computed average xsi. Output shape: {xsi.shape}")
        return(xsi)

    def x_sgp_topo_helbig_idx(self, mnt, idx_x, idx_y, dx, L=2_000, reduce_mnt=True, nb_pixels_x=100, nb_pixels_y=100, verbose=True):

        a = 3.354688
        b = 1.998767
        c = 0.20286
        d = 5.951

        mu = self.mu_helbig_average(mnt, dx, idx_x, idx_y, reduce_mnt=reduce_mnt, nb_pixels_x=nb_pixels_x, nb_pixels_y=nb_pixels_y)
        xsi = self.xsi_helbig_map(mnt, mu, idx_x, idx_y, reduce_mnt=reduce_mnt, nb_pixels_x=nb_pixels_x, nb_pixels_y=nb_pixels_y, librairie="numba")

        x = 1 - (1 - (1/(1+a*mu**b))**c)*np.exp(-d*(L/xsi)**(-2))
        x = change_dtype_if_required(x, np.float32)
        if verbose: print(f"__Subgrid: computed x_sgp_topo. Output shape: {x.shape}")
        return(x)

    def subgrid_map(self, mnt_large, dx=25, L=2_000, reduce_mnt=True, nb_pixels_x=100, nb_pixels_y=100, verbose=True):

        shape_large = mnt_large.shape
        if verbose: print(f"Large mnt shape: {shape_large}. Size reduction on x: 2 * {nb_pixels_x}. Size reduction on x: 2 * {nb_pixels_y} ")
        x = list(range(shape_large[1]))
        y = list(range(shape_large[0]))
        idx_x, idx_y = np.array(np.meshgrid(x, y)).astype(np.int32)

        x_sgp_topo = self.x_sgp_topo_helbig_idx(mnt_large, idx_x, idx_y, dx, L=L, reduce_mnt=reduce_mnt, nb_pixels_x=nb_pixels_x, nb_pixels_y=nb_pixels_y)
        x_sgp_topo = change_dtype_if_required(x_sgp_topo, np.float32)
        return (x_sgp_topo)


class Dwnsc_helbig(Sgp_helbig):

    def __init__(self):
        super().__init__()

    def x_dsc_topo_helbig_map(self, mnt, dx, verbose=True):

        a = 17.0393
        b = 0.737
        c = 1.0234
        d = 0.3794
        e = 1.9821

        laplacian_map = self.laplacian_map(mnt, dx, helbig=True)

        term_1 = 1 - a*laplacian_map/(1+a*np.abs(laplacian_map)**b)
        term_2 = c / (1+d*self.mu_helbig_map(mnt, dx)**e)
        x = term_1*term_2

        if verbose: print(f"MNT shape: {mnt.shape}")
        if verbose: print(f"x_dsc_topo computed. x shape: {x.shape}")

        x = change_dtype_if_required(x, np.float32)
        return(x)

    def x_dsc_topo_helbig_idx(self, mnt, idx_x, idx_y, dx):

        a = 17.0393
        b = 0.737
        c = 1.0234
        d = 0.3794
        e = 1.9821

        term_1 = 1 - a*self.laplacian_idx(mnt, idx_x, idx_y, dx)/(1+a*self.laplacian_idx(mnt, idx_x, idx_y, dx, helbig=True)**b)
        term_2 = c / (1+d*self.mu_helbig_idx(mnt, dx)**2)
        x = term_1*term_2

        x = change_dtype_if_required(x, np.float32)
        return(x)


