import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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

    def normalize_topo(self, topo_HD, mean, std, dtype=np.float32, library="numexpr", verbose=True):
        """
        Normalize a topography with mean and std.

        Parameters
        ----------
        topo_HD : array
        mean : array
        std : array
        dtype: numpy dtype
        library: str
        verbose: boolean

        Returns
        -------
        Standardized topography : array
        """
        print(f"__Normalize done with mean {len(mean)} means and std") if verbose else None

        if library == 'tensorflow':
            topo_HD = tf.constant(topo_HD, dtype=tf.float32)
            mean = tf.constant(mean, dtype=tf.float32)
            std = tf.constant(std, dtype=tf.float32)
            result = tf.subtract(topo_HD, mean)
            result = tf.divide(result, result)
            return result
        else:
            topo_HD = np.array(topo_HD, dtype=dtype)
            if library == 'numexpr' and self._numexpr:
                return ne.evaluate("(topo_HD - mean) / std")
            else:
                return (topo_HD - mean) / std

    @staticmethod
    def mean_peak_valley(topo, verbose=True):
        """
        2 * std(topography)

        Mean peak valley height

        Parameters
        ----------
        topo : ndarray
            topography

        verbose : boolean
            verbose

        Returns
        -------
        peak_valley_height : ndarray
            Mean peak valley height
        """
        peak_valley_height = 2 * np.nanstd(topo)
        print("__Mean peak valley computed") if verbose else None
        return peak_valley_height.astype(np.float32)

    def laplacian_map(self, mnt, dx, library="numpy", helbig=True, verbose=True):
        if mnt.ndim > 2:
            return np.vectorize(self._laplacian_map, signature='(m,n),(),(),(),()->(m,n)')(mnt,
                                                                                  dx,
                                                                                  library,
                                                                                  helbig,
                                                                                  verbose)
        else:
            return self._laplacian_map(mnt, dx, library, helbig, verbose)

    def _laplacian_map(self, mnt, dx, library, helbig, verbose):

        # Pad mnt to compute laplacian on edges
        mnt_padded = np.pad(mnt, (1, 1), "edge").astype(np.float32)
        print("__MNT padded for laplacian computation") if verbose else None
        shape = mnt_padded.shape

        # Use meshgrid to create indexes with mnt size and use numpy broadcasting when selecting indexes
        xx, yy = np.array(np.meshgrid(list(range(shape[1])), list(range(shape[0])))).astype(np.int32)

        # Compute laplacian on indexes using an index for every grid point (meshgrid)
        laplacian = self.laplacian_idx(mnt_padded, xx[1:-1, 1:-1], yy[1:-1, 1:-1], dx,
                                       library=library, helbig=helbig, verbose=verbose)

        print(f"__Laplacian map calculated. Shape: {laplacian.shape}. Library: {library}") if verbose else None

        return laplacian

    def laplacian_idx(self, mnt, idx_x, idx_y, dx, verbose=True, library='numpy', helbig=True):
        """
        Discrete laplacian on a regular grid
        """

        if library == 'numba' and _numba:

            mnt = change_dtype_if_required(mnt, np.float32)
            idx_x = change_dtype_if_required(idx_x, np.int32)
            idx_y = change_dtype_if_required(idx_y, np.int32)

            laplacian = self._laplacian_numba_idx(mnt, idx_x, idx_y, dx, helbig=helbig)
            library = library

        else:
            laplacian = self._laplacian_numpy_idx(mnt, idx_x, idx_y, dx, helbig=helbig)
            library = "numpy"

        laplacian = change_dtype_if_required(laplacian, np.float32)
        print(f"__Laplacian calculated. Librarie: {library}") if verbose else None
        return laplacian

    @staticmethod
    def _laplacian_numpy_idx(mnt, idx_x, idx_y, dx, helbig=True):
        a = np.float32((mnt[idx_y-1, idx_x] + mnt[idx_y+1, idx_x] + mnt[idx_y, idx_x-1] + mnt[idx_y, idx_x+1] - 4*mnt[idx_y, idx_x])/dx**2)
        c = np.float32(dx/4) if helbig else 1
        return a * c

    @staticmethod
    def _laplacian_loop_numpy_1D_helbig(mnt, idx_x, idx_y, dx):
        laplacian = np.empty(idx_x.shape, np.float32)
        for i in range(idx_x.shape[0]):
            a = (mnt[idx_x[i] - 1, idx_y[i]] + mnt[idx_x[i] + 1, idx_y[i]] + mnt[idx_x[i], idx_y[i] - 1] + mnt[
                idx_x[i], idx_y[i] + 1] - 4 * mnt[idx_x[i], idx_y[i]]) / dx ** 2
            c = dx / 4
            laplacian[i] = a * c
        return laplacian

    @staticmethod
    def _laplacian_loop_numpy_1D(mnt, idx_x, idx_y, dx):
        laplacian = np.empty(idx_x.shape, np.float32)
        for i in range(idx_x.shape[0]):
            a = (mnt[idx_x[i] - 1, idx_y[i]] + mnt[idx_x[i] + 1, idx_y[i]] + mnt[idx_x[i], idx_y[i] - 1] + mnt[
                idx_x[i], idx_y[i] + 1] - 4 * mnt[idx_x[i], idx_y[i]]) / dx ** 2
            laplacian[i] = a
        return laplacian

    @staticmethod
    def _laplacian_loop_numpy_2D_helbig(mnt, idx_x, idx_y, dx):
        laplacian = np.empty(idx_x.shape, np.float32)
        for j in range(idx_x.shape[0]):
            for i in range(idx_x.shape[1]):
                a = (mnt[idx_y[j, i] - 1, idx_x[j, i]] + mnt[idx_y[j, i] + 1, idx_x[j, i]] + mnt[idx_y[j, i], idx_x[j, i] - 1] + mnt[
                    idx_y[j, i], idx_x[j, i] + 1] - 4 * mnt[idx_y[j, i], idx_x[j, i]]) / dx ** 2
                c = dx / 4
                laplacian[j, i] = a * c
        return laplacian

    @staticmethod
    def _laplacian_loop_numpy_2D(mnt, idx_x, idx_y, dx):
        laplacian = np.empty(idx_x.shape, np.float32)
        for j in range(idx_x.shape[0]):
            for i in range(idx_x.shape[1]):
                a = (mnt[idx_y[j, i] - 1, idx_x[j, i]] + mnt[idx_y[j, i] + 1, idx_x[j, i]] + mnt[idx_y[j, i], idx_x[j, i] - 1] + mnt[
                    idx_y[j, i], idx_x[j, i] + 1] - 4 * mnt[idx_y[j, i], idx_x[j, i]]) / dx ** 2
                laplacian[j, i] = a
        return laplacian

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
        return result

    @staticmethod
    def _get_window_idx_boundaries(idx_x, idx_y, x_win=69//2, y_win=79//2, reshape=True):

        y_left = np.int32(idx_y - y_win)
        y_right = np.int32(idx_y + y_win)
        x_left = np.int32(idx_x - x_win)
        x_right = np.int32(idx_x + x_win)

        if reshape:
            flat_shape = idx_y.shape[0] * idx_y.shape[1]
            return y_left.reshape(flat_shape), y_right.reshape(flat_shape), x_left.reshape(flat_shape), x_right.reshape(flat_shape)

        return y_left, y_right, x_left, x_right

    @staticmethod
    def rolling_window_mean_numpy(in_arr, out_arr, x_win, y_win):
        """
        Inspired from
        https://stackoverflow.com/questions/48215914/vectorized-2-d-moving-window-in-numpy-including-edges
        """
        yn, xn = in_arr.shape
        for x in range(xn):
            xmin = max(0, x - x_win)
            xmax = min(xn, x + x_win + 1)
            for y in range(yn):
                ymin = max(0, y - y_win)
                ymax = min(yn, y + y_win + 1)
                out_arr[y, x] = np.mean(in_arr[ymin:ymax, xmin:xmax])
        return out_arr

    def tpi_map(self, mnt, radius, resolution=25, library='numba'):

        x_win, y_win = self.radius_to_square_window(radius, resolution)

        if library == 'numba' and _numba:
            window_func = jit("float32[:,:](float32[:,:],float32[:,:],int32, int32)", nopython=True, cache=True)(self.rolling_window_mean_numpy)
        else:
            window_func = self.rolling_window_mean_numpy

        mnt, x_win, y_win = change_several_dtype_if_required([mnt, x_win, y_win], [np.float32, np.int32, np.int32])
        output = np.empty_like(mnt).astype(np.float32)
        try:
            mean = window_func(mnt, output, x_win, y_win)
        except ZeroDivisionError:
            print("mnt shape", mnt.shape)
            print("output shape", output.shape)
            print("x_win", x_win)
            print("y_win", y_win)
            print("x_win shape", x_win.shape)
            print("x_win shape", x_win.shape)

        return mnt - mean

    @staticmethod
    def _control_idx_boundaries(idx, min_idx=0, max_idx=None):
        if isinstance(idx, list) or isinstance(idx, np.ndarray):
            result = []
            for index, x in enumerate(idx):
                x = np.where(x < min_idx[index], 0, x)
                x = np.where(x > max_idx[index], max_idx[index], x)
                result.append(x)
            return result
        else:
            idx = np.where(idx < min_idx, 0, idx)
            idx = np.where(idx > max_idx, max_idx, idx)
            return idx

    @staticmethod
    def radius_to_square_window(radius, resolution):
        if radius % resolution == 0:
            dx = np.int32(radius / resolution)
            return dx, dx
        else:
            dx = np.int32(radius // resolution + 1)
            return dx, dx

    def tpi_idx(self, mnt, idx_x, idx_y, radius, resolution=25):

        x_win, y_win = self.radius_to_square_window(radius, resolution)

        y_left, y_right, x_left, x_right = self._get_window_idx_boundaries(idx_x, idx_y,
                                                                           x_win=x_win, y_win=y_win,
                                                                           reshape=False)

        boundaries_mnt = [mnt.shape[0], mnt.shape[0], mnt.shape[1], mnt.shape[1]]
        y_left, y_right, x_left, x_right = self._control_idx_boundaries([y_left, y_right, x_left, x_right],
                                                                        min_idx=[0, 0, 0, 0],
                                                                        max_idx=boundaries_mnt)

        mean_window = np.array([np.mean(mnt[i1:j1+1, i2:j2+1]) for i1, j1, i2, j2 in zip(y_left, y_right, x_left, x_right)])
        return mnt[idx_y, idx_x] - mean_window

    def sx_idx(self, mnt, idx_x, idx_y, cellsize=25, dmax=300, in_wind=270, wind_inc=5, wind_width=30):

        x_win, y_win = self.radius_to_square_window(dmax, cellsize)

        y_left, y_right, x_left, x_right = self._get_window_idx_boundaries(idx_x, idx_y,
                                                                           x_win=x_win, y_win=y_win,
                                                                           reshape=False)

        boundaries_mnt = [mnt.shape[0], mnt.shape[0], mnt.shape[1], mnt.shape[1]]
        y_left, y_right, x_left, x_right = self._control_idx_boundaries([y_left, y_right, x_left, x_right],
                                                                        min_idx=[0, 0, 0, 0],
                                                                        max_idx=boundaries_mnt)

        output_shape = np.array(mnt[y_left[0]:y_right[0] + 1, x_left[0]:x_right[0] + 1]).shape
        cntr_y = output_shape[0] // 2
        cntr_x = output_shape[1] // 2
        sx_indexes = [self.sx_map(mnt[i1:j1 + 1, i2:j2 + 1], cellsize, dmax, in_wind, wind_inc, wind_width)[cntr_x, cntr_y] for i1, j1, i2, j2 in zip(y_left, y_right, x_left, x_right)]

        return np.array(sx_indexes)

    @staticmethod
    def _sx_idx(dem_padded, y, x, y_offsets, x_offsets, distances):
        # compute the difference in altitude for all cells along all angles
        altitude_diff = dem_padded[y + y_offsets, x + x_offsets] - dem_padded[y, x]

        # directions are in the first dimension, cells in the last
        slope = altitude_diff / distances
        amax = np.nanmax(slope, -1)

        # Pas sur qu'il faut calculer le minimum ici
        amin = np.nanmin(slope, -1)
        result = np.where(-amin > amax, amin, amax)

        # maybe nanmean would be more correct, but we reproduce the
        # exisiting implementation for now
        result = np.nanmean(np.arctan(result))

        return result

    def sx_map(self, dem, cellsize, dmax, in_wind, wind_inc=5, wind_width=30):

        grid = np.full(dem.shape, np.nan)

        # Pad the input edges with np.nan
        nb_cells = np.ceil(dmax / cellsize).astype(np.int32)
        pad_shape = ((nb_cells, nb_cells),) * 2
        dem_padded = np.pad(dem, pad_shape, 'constant', constant_values=np.nan)

        # Define wind sectors in degrees
        wind_left = in_wind - wind_width / 2
        wind_right = in_wind + wind_width / 2
        winds = np.arange(wind_left, wind_right, wind_inc)

        # The angles we check. Add last dimension so we can broadcast the direction
        # samples
        alpha_rad = np.expand_dims(np.deg2rad(winds), -1)

        # pre-compute the cell indices that are sampled for each direction
        # Je n'aurais pas mis le moins ici
        #y_offsets = -np.round(np.arange(1, nb_cells) * np.cos(alpha_rad)).astype(np.int32)
        y_offsets = np.round(np.arange(1, nb_cells) * np.cos(alpha_rad)).astype(np.int32)
        x_offsets = np.round(np.arange(1, nb_cells) * np.sin(alpha_rad)).astype(np.int32)

        # pre-compute the distances for each sampled cell
        distances = cellsize*np.sqrt(x_offsets ** 2 + y_offsets ** 2).astype(np.float32)

        # set distances that are too large to np.nan so they're not considered
        distances[(distances == 0.) | (distances > dmax)] = np.nan

        for y in range(nb_cells, nb_cells + dem.shape[0]):
            for x in range(nb_cells, nb_cells + dem.shape[1]):
                # result = np.nansum(np.arctan(result)) / len(winds)
                grid[y - nb_cells, x - nb_cells] = self._sx_idx(dem_padded, y, x, y_offsets, x_offsets, distances)
        return grid