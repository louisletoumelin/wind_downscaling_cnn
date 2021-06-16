import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import datetime
import time
from time import time as t
import os
import concurrent.futures

from downscale.Utils.Utils import change_dtype_if_required, change_several_dtype_if_required

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

    def normalize_topo(self, topo_HD, mean, std, dtype=np.float32, librairie="numexpr", verbose=True):
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
        print(f"__Normalize done with mean {len(mean)} means and std") if verbose else None

        if librairie == 'tensorflow':
            topo_HD = tf.constant(topo_HD, dtype=tf.float32)
            mean = tf.constant(mean, dtype=tf.float32)
            std = tf.constant(std, dtype=tf.float32)
            result = tf.subtract(topo_HD, mean)
            result = tf.divide(result, result)
            return (result)
        else:
            topo_HD = np.array(topo_HD, dtype=dtype)
            if librairie == 'numexpr' and self._numexpr:
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
        print("__Mean peak valley computed") if verbose else None
        return(peak_valley_height.astype(np.float32))

    def laplacian_map(self, mnt, dx, librairie="numpy", helbig=True, verbose=True):

        # Pad mnt to compute laplacian on edges
        mnt_padded = np.pad(mnt, (1, 1), "edge").astype(np.float32)
        shape = mnt_padded.shape

        # Use meshgrid to create indexes with mnt size and use numpy broadcasting when selecting indexes
        xx, yy = np.array(np.meshgrid(list(range(shape[1])), list(range(shape[0])))).astype(np.int32)

        # Compute laplacian on indexes using an index for every grid point (meshgrid)
        laplacian = self.laplacian_idx(mnt_padded, xx[1:-1, 1:-1], yy[1:-1, 1:-1], dx, librairie=librairie, helbig=helbig)

        if verbose: print(f"__Laplacian map calculated. Shape: {laplacian.shape}. Librairie: {librairie}")

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
        if verbose: print(f"__Laplacian calculated. Librarie: {librairie}")
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
        return(result)

    def mu_helbig_map(self, mnt, dx, verbose=True):
        """
        Adapted from I. Gouttevin

        From Helbig et al. 2017
        """
        mu = np.sqrt(np.sum(np.array(np.gradient(mnt, dx))**2, axis=0)/2)
        mu = change_dtype_if_required(mu, np.float32)
        if verbose: print("__mu calculation using numpy")
        return(mu)

    def mu_helbig_idx(self, mnt, dx, idx_x, idx_y, verbose=True):
        mu = self.mu_helbig_map(mnt, dx)
        mu = change_dtype_if_required(mu, np.float32)
        if verbose: print("__Selecting indexes on mu")
        return(mu[idx_y, idx_x])

    @staticmethod
    def _get_window_idx_boundaries(idx_x, idx_y, x_win=69//2, y_wind=79//2, reshape=True):

        y_left = np.int32(idx_y - y_wind)
        y_right = np.int32(idx_y + y_wind)
        x_left = np.int32(idx_x - x_win)
        x_right = np.int32(idx_x + x_win)

        if reshape:
            flat_shape = idx_y.shape[0] * idx_y.shape[1]
            return (y_left.reshape(flat_shape), y_right.reshape(flat_shape), x_left.reshape(flat_shape), x_right.reshape(flat_shape))

        return(y_left, y_right, x_left, x_right)

    @staticmethod
    def rolling_window_mean_numpy(in_arr, out_arr, x_win, y_win):
        """
        Inspired from
        https://stackoverflow.com/questions/48215914/vectorized-2-d-moving-window-in-numpy-including-edges
        """
        xn, yn = in_arr.shape
        for x in range(xn):
            xmin = max(0, x - x_win)
            xmax = min(xn, x + x_win + 1)
            for y in range(yn):
                ymin = max(0, y - y_win)
                ymax = min(yn, y + y_win + 1)
                out_arr[x, y] = np.mean(in_arr[xmin:xmax, ymin:ymax])
        return out_arr

    def tpi_map(self, mnt, radius, resolution=25, librairie='numba'):

        x_win, y_win = self.radius_to_square_window(radius, resolution)

        if librairie == 'numba' and _numba:
            window_func = jit("float32[:,:](float32[:,:],float32[:,:],int32, int32)", nopython=True, cache=True)(self.rolling_window_mean_numpy)
        else:
            window_func = self.rolling_window_mean_numpy

        output = np.empty_like(mnt).astype(np.float32)
        mean = window_func(mnt, output, x_win, y_win)

        return(mnt-mean)

    @staticmethod
    def _control_idx_boundaries(idx, min_idx=0, max_idx=None):
        if isinstance(idx, list) or isinstance(idx, np.ndarray):
            result = []
            for index, x in enumerate(idx):
                x = np.where(x < min_idx[index], 0, x)
                x = np.where(x > max_idx[index], max_idx[index], x)
                result.append(x)
            return(result)
        else:
            idx = np.where(idx < min_idx, 0, idx)
            idx = np.where(idx > max_idx, max_idx, idx)
            return(idx)

    @staticmethod
    def radius_to_square_window(radius, resolution):
        if radius % resolution == 0:
            dx = np.int32(radius / resolution)
            return(dx, dx)
        else:
            dx = np.int32(radius // resolution + 1)
            return(dx, dx)

    def tpi_idx(self, mnt, idx_x, idx_y, radius, resolution=25):

        x_win, y_win = self.radius_to_square_window(radius, resolution)

        y_left, y_right, x_left, x_right = self._get_window_idx_boundaries(idx_x, idx_y,
                                                                           x_win=x_win, y_wind=y_win,
                                                                           reshape=False)

        boundaries_mnt = [mnt.shape[0], mnt.shape[0], mnt.shape[1], mnt.shape[1]]
        y_left, y_right, x_left, x_right = self._control_idx_boundaries([y_left, y_right, x_left, x_right],
                                                                        min_idx=[0, 0, 0, 0],
                                                                        max_idx=boundaries_mnt)

        mean_window = np.array([np.mean(mnt[i1:j1, i2:j2]) for i1, j1, i2, j2 in zip(y_left, y_right, x_left, x_right)])

        return(mnt[idx_x, idx_y] - mean_window)


class Sgp_helbig(Topo_utils):

    def __init__(self):
        super().__init__()
    
    def mu_helbig_average(self, mnt, dx, idx_x, idx_y, reduce_mnt=True, type="map", nb_pixels_x=100, nb_pixels_y=100, librairie="numba", verbose=True):

        if reduce_mnt:
            small_idx_y = idx_y[nb_pixels_y:-nb_pixels_y, nb_pixels_x:-nb_pixels_x]
            small_idx_x = idx_x[nb_pixels_y:-nb_pixels_y, nb_pixels_x:-nb_pixels_x]
            shape = small_idx_y.shape
            y_left, y_right, x_left, x_right = self._get_window_idx_boundaries(small_idx_x, small_idx_y)
        else:
            y_left, y_right, x_left, x_right = self._get_window_idx_boundaries(idx_x, idx_y, reshape=False)

        if type == "map":
            mu = self.mu_helbig_map(mnt, dx)

            if librairie == 'numba' and _numba:
                mu, y_left, y_right, x_left, x_right = change_several_dtype_if_required(
                    [mnt, y_left, y_right, x_left, x_right], [np.float32, np.int32, np.int32, np.int32, np.int32])
                jit_mean = jit([float32[:](float32[:, :], int32[:], int32[:], int32[:], int32[:])], nopython=True, cache=True)(self.mean_slicing_numpy)
                mu_flat = jit_mean(mu, y_left, y_right, x_left, x_right)
                librairie = "numba"
            else:
                mu_flat = np.array([np.mean(mu[i1:j1, i2:j2]) for i1, j1, i2, j2 in zip(y_left, y_right, x_left, x_right)])
                librairie = "numpy"
        elif type == "indexes":
            boundaries_mnt = [mnt.shape[0], mnt.shape[0], mnt.shape[1], mnt.shape[1]]
            y_left, y_right, x_left, x_right = self._control_idx_boundaries([y_left, y_right, x_left, x_right],
                                                                            min_idx=[0, 0, 0, 0],
                                                                            max_idx=boundaries_mnt)
            mu_flat = np.array([np.mean(self.mu_helbig_map(mnt[i1:j1, i2:j2], dx)) for i1, j1, i2, j2 in zip(y_left, y_right, x_left, x_right)])

        mu = mu_flat.reshape((shape[0], shape[1])) if reduce_mnt else mu_flat

        mu = change_dtype_if_required(mu, np.float32)
        if verbose: print(f"__Subgrid: computed average mu. Output shape: {mu.shape}. Librairie: {librairie}")

        return(mu)

    @staticmethod
    def mean_slicing_numpy(array, y_left, y_right, x_left, x_right):
        result = np.empty(y_left.shape)
        for index, (i1, j1, i2, j2) in enumerate(zip(y_left, y_right, x_left, x_right)):
            result[index] = np.mean(array[i1:j1, i2:j2])
        return (result.astype(np.float32))

    @staticmethod
    def std_slicing_numpy(array, y_left, y_right, x_left, x_right):
        result = np.empty(y_left.shape)
        for index, (i1, j1, i2, j2) in enumerate(zip(y_left, y_right, x_left, x_right)):
            result[index] = np.std(array[i1:j1, i2:j2])
        return (result.astype(np.float32))

    def xsi_helbig_map(self, mnt, mu, idx_x, idx_y, reduce_mnt=True, nb_pixels_x=100, nb_pixels_y=100, librairie="numba", verbose=True):

        y_left, y_right, x_left, x_right = self._get_window_idx_boundaries(idx_x, idx_y, x_win=69//2, y_wind=79//2, reshape=False)

        if reduce_mnt:
            small_idx_y = idx_y[nb_pixels_y:-nb_pixels_y:, nb_pixels_x:-nb_pixels_x]
            small_idx_x = idx_x[nb_pixels_y:-nb_pixels_y:, nb_pixels_x:-nb_pixels_x]
            y_left, y_right, x_left, x_right = self._get_window_idx_boundaries(small_idx_x, small_idx_y)
            small_shape = small_idx_y.shape

        if librairie == "numba" and _numba:
            mnt, y_left, y_right, x_left, x_right = change_several_dtype_if_required([mnt, y_left, y_right, x_left, x_right], [np.float32, np.int32, np.int32, np.int32, np.int32])
            boundaries_mnt = [mnt.shape[0], mnt.shape[0], mnt.shape[1], mnt.shape[1]]
            y_left, y_right, x_left, x_right = self._control_idx_boundaries([y_left, y_right, x_left, x_right],
                                                                            min_idx=[0, 0, 0, 0],
                                                                            max_idx=boundaries_mnt)
            std_slicing_numba = jit([float32[:](float32[:, :], int32[:], int32[:], int32[:], int32[:])], cache=True, nopython=True)(self.std_slicing_numpy)
            std_flat = std_slicing_numba(mnt, y_left, y_right, x_left, x_right)
            librairie = "numba"
        else:
            std_flat = np.array([np.std(mnt[i1:j1, i2:j2]) for i1, j1, i2, j2 in zip(y_left, y_right, x_left, x_right)])
            librairie = "numpy"

        std = std_flat.reshape((small_shape[0], small_shape[1])) if reduce_mnt else std_flat
        if verbose: print(f"__Subgrid: computed average std. Output shape: {std.shape}. Librairie: {librairie}")

        xsi = np.sqrt(2) * std / mu

        xsi = change_dtype_if_required(xsi, np.float32)
        if verbose: print(f"__Subgrid: computed average xsi. Output shape: {xsi.shape}")
        return(xsi)

    def x_sgp_topo_helbig_idx(self, mnt, idx_x, idx_y, dx, L=2_000, type="map", reduce_mnt=True, nb_pixels_x=100, nb_pixels_y=100, verbose=True):

        a = 3.354688
        b = 1.998767
        c = 0.20286
        d = 5.951

        mu = self.mu_helbig_average(mnt, dx, idx_x, idx_y, type=type, reduce_mnt=reduce_mnt)
        xsi = self.xsi_helbig_map(mnt, mu, idx_x, idx_y, reduce_mnt=reduce_mnt, nb_pixels_x=nb_pixels_x, nb_pixels_y=nb_pixels_y, librairie="numba")

        x = 1 - (1 - (1/(1+a*mu**b))**c)*np.exp(-d*(L/xsi)**(-2))

        x = change_dtype_if_required(x, np.float32)
        if verbose: print(f"__Subgrid: computed x_sgp_topo. Output shape: {x.shape}")

        return(x)

    def subgrid(self, mnt_large, dx=25, L=2_000, idx_x=None, idx_y=None, type="map", reduce_mnt=True, nb_pixels_x=100, nb_pixels_y=100, verbose=True):

        if type == "map":
            shape = mnt_large.shape
            all_x_idx = range(shape[1])
            all_y_idx = range(shape[0])
            idx_x, idx_y = np.array(np.meshgrid(all_x_idx, all_y_idx)).astype(np.int32)
            if verbose: print(f"Large mnt shape: {shape}. Size reduction on x: 2 * {nb_pixels_x}. Size reduction on x: 2 * {nb_pixels_y} ")

        reduce_mnt = False if type == "indexes" else reduce_mnt
        x_sgp_topo = self.x_sgp_topo_helbig_idx(mnt_large, idx_x, idx_y, dx,
                                                L=L,
                                                type=type,
                                                reduce_mnt=reduce_mnt,
                                                nb_pixels_x=nb_pixels_x,
                                                nb_pixels_y=nb_pixels_y)

        return (x_sgp_topo)


class Dwnsc_helbig(Sgp_helbig):

    def __init__(self):
        super().__init__()

    def x_dsc_topo_helbig(self, mnt, dx=25, idx_x=None, idx_y=None, type="map", librairie="numba", verbose=True):

        a = 17.0393
        b = 0.737
        c = 1.0234
        d = 0.3794
        e = 1.9821

        if type == "map":
            laplacian = self.laplacian_map(mnt, dx, librairie=librairie, helbig=True)
            mu = self.mu_helbig_map(mnt, dx)
        elif type == "indexes":
            idx_x, idx_y = change_several_dtype_if_required([idx_x, idx_y], [np.int32, np.int32])
            laplacian = self.laplacian_idx(mnt, idx_x, idx_y, dx, librairie=librairie, helbig=True)
            mu = self.mu_helbig_idx(mnt, dx, idx_x, idx_y)

        term_1 = 1 - a*laplacian/(1 + a*np.abs(laplacian)**b)
        term_2 = c / (1 + d*mu**e)
        x = term_1*term_2

        if verbose:
            print(f"__MNT shape: {mnt.shape}")
            print(f"__x_dsc_topo computed. x shape: {x.shape}")

        x = change_dtype_if_required(x, np.float32)
        return(x)

    def downscale_helbig(self, mnt_large, dx=25, L=2_000, idx_x=None, idx_y=None, type="map", reduce_mnt=True,
                         librairie="numba", nb_pixels_x=100, nb_pixels_y=100, verbose=True):

        if verbose: print(f"\nBegin subgrid parameterization from Helbig et al. 2017")
        x_sgp_topo = self.subgrid(mnt_large,
                                  idx_x=idx_x,
                                  idx_y=idx_y,
                                  dx=dx,
                                  L=L,
                                  type=type,
                                  reduce_mnt=reduce_mnt,
                                  nb_pixels_x=nb_pixels_x,
                                  nb_pixels_y=nb_pixels_y,
                                  verbose=verbose)

        plt.figure()
        plt.imshow(x_sgp_topo)
        plt.title("Subgrid Helbig et al. 2017")

        mnt_small = mnt_large[nb_pixels_y:-nb_pixels_y:, nb_pixels_x:-nb_pixels_x]

        plt.figure()
        plt.imshow(mnt_large)
        plt.title("MNT")

        if verbose: print(f"\nBegin downscaling from Helbig et al. 2017")
        x_dsc_topo = self.x_dsc_topo_helbig(mnt_small, dx=dx, idx_x=idx_x, idx_y=idx_y, type=type, verbose=verbose, librairie=librairie)

        plt.figure()
        plt.imshow(x_dsc_topo)
        plt.title("x_dsc_topo Helbig et al. 2017")

        plt.figure()
        plt.imshow(x_sgp_topo*x_dsc_topo)
        plt.title("Downscaling Helbig et al. 2017")

        return(x_sgp_topo*x_dsc_topo)
