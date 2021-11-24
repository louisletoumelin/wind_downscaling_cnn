import numpy as np
import matplotlib.pyplot as plt

from downscale.operators.topo_utils import Topo_utils
from downscale.utils.utils_func import change_dtype_if_required, change_several_dtype_if_required
from downscale.utils.decorators import print_func_executed_decorator, timer_decorator, \
    change_dtype_if_required_decorator

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


class SgpHelbig(Topo_utils):

    def __init__(self):
        super().__init__()

    @print_func_executed_decorator("mu_helbig_map", level_begin="____", level_end="____", end="")
    @timer_decorator("mu_helbig_map", unit='minute', level=".... ")
    def mu_helbig_map(self, mnt, dx, verbose=True):
        if mnt.ndim > 2:
            return np.vectorize(self._mu_helbig_map, signature='(m,n),(),()->(m,n)')(mnt,
                                                                                     dx,
                                                                                     verbose)
        else:
            return self._mu_helbig_map(mnt, dx, verbose)

    @staticmethod
    @change_dtype_if_required_decorator(np.float32)
    def _mu_helbig_map(mnt, dx, verbose):
        """
        Adapted from I. Gouttevin

        From Helbig et al. 2017
        """

        mu = np.sqrt(np.sum(np.array(np.gradient(mnt, dx)) ** 2, axis=0) / 2)
        print("____mu calculation. Library: numpy") if verbose else None

        return mu

    @print_func_executed_decorator("mu_helbig_idx", level_begin="__", level_end="__", end="")
    @timer_decorator("mu_helbig_idx", unit='minute', level="....")
    @change_dtype_if_required_decorator(np.float32)
    def mu_helbig_idx(self, mnt, dx, idx_x, idx_y):
        """This function can not be directly written with numba"""
        mu_helbig_func = self.mu_helbig_map
        mu = [mu_helbig_func(mnt[y - 1:y + 2, x - 1:x + 2], dx, verbose=False)[1, 1] for (x, y) in zip(idx_x, idx_y)]
        return mu

    @staticmethod
    def _idx_from_array_shape(array):
        shape = array.shape
        idx_x = range(shape[1])
        idx_y = range(shape[0])
        idx_x, idx_y = np.array(np.meshgrid(idx_x, idx_y)).astype(np.int32)
        return idx_x, idx_y

    @timer_decorator("mu_average_numba", unit='minute', level=".... ")
    def mu_average_numba(self, mu, y_left, y_right, x_left, x_right):

        mu, y_left, y_right, x_left, x_right = change_several_dtype_if_required(
            [mu, y_left, y_right, x_left, x_right], [np.float32, np.int32, np.int32, np.int32, np.int32])

        jit_mean = jit([float32[:](float32[:, :], int32[:], int32[:], int32[:], int32[:])], nopython=True)(
            self.mean_slicing_numpy)

        return jit_mean(mu, y_left, y_right, x_left, x_right)

    @staticmethod
    @timer_decorator("mu_average_numpy", unit='minute', level=".... ")
    def mu_average_numpy(mu, y_left, y_right, x_left, x_right):
        return np.array([np.mean(mu[i1:j1 + 1, i2:j2 + 1]) for i1, j1, i2, j2 in zip(y_left, y_right, x_left, x_right)])

    @timer_decorator("mu_average_idx", unit='minute', level=".... ")
    def mu_average_idx(self, mnt, y_left, y_right, x_left, x_right, dx, verbose=True):
        boundaries_mnt = [mnt.shape[0], mnt.shape[0], mnt.shape[1], mnt.shape[1]]
        y_left, y_right, x_left, x_right = self._control_idx_boundaries([y_left, y_right, x_left, x_right],
                                                                        min_idx=[0, 0, 0, 0],
                                                                        max_idx=boundaries_mnt)
        return np.array([np.mean(self.mu_helbig_map(mnt[i1:j1 + 1, i2:j2 + 1], dx, verbose=verbose))
                         for i1, j1, i2, j2 in zip(y_left, y_right, x_left, x_right)])

    @print_func_executed_decorator("mu_helbig_average", level_begin="__", level_end="__", end="")
    @timer_decorator("mu_helbig_average", unit='minute', level="....")
    @change_dtype_if_required_decorator(np.float32)
    def mu_helbig_average(self, mnt, dx, idx_x=None, idx_y=None, reduce_mnt=True, type_input="map", x_win=69 // 2,
                          y_win=79 // 2, nb_pixels_x=100, nb_pixels_y=100, library="numba", verbose=True):

        if idx_x is None and idx_y is None:
            idx_x, idx_y = self._idx_from_array_shape(mnt)

        y_left, y_right, x_left, x_right, shape = self.get_and_control_idx_boundary(mnt, idx_x, idx_y,
                                                                                    nb_pixels_x, nb_pixels_y,
                                                                                    reduce_mnt=reduce_mnt,
                                                                                    x_win=x_win,
                                                                                    y_win=y_win)
        if type_input == "map":

            mu = self.mu_helbig_map(mnt, dx, verbose=verbose)

            if library == 'numba' and _numba:
                mu_flat = self.mu_average_numba(mu, y_left, y_right, x_left, x_right)
                library = "numba"
            else:
                mu_flat = self.mu_average_numpy(mu, y_left, y_right, x_left, x_right)
                library = "numpy"

        elif type_input == "indexes":

            mu_flat = self.mu_average_idx(mnt, y_left, y_right, x_left, x_right, dx, verbose=verbose)

        mu = mu_flat.reshape((shape[0], shape[1])) if (type_input == "map" or reduce_mnt) else mu_flat

        print(f"____mu_helbig_average. Output shape: {mu.shape}. Library: {library}") if verbose else None

        return mu

    @staticmethod
    def mean_slicing_numpy(array, y_left, y_right, x_left, x_right):
        result = np.empty(y_left.shape)
        for index, (i1, j1, i2, j2) in enumerate(zip(y_left, y_right, x_left, x_right)):
            result[index] = np.mean(array[i1:j1 + 1, i2:j2 + 1])
        return result.astype(np.float32)

    @staticmethod
    def std_slicing_numpy_loop(array, y_left, y_right, x_left, x_right):
        result = np.empty(y_left.shape)
        for index, (i1, j1, i2, j2) in enumerate(zip(y_left, y_right, x_left, x_right)):
            result[index] = np.std(array[i1:j1 + 1, i2:j2 + 1])
        return result.astype(np.float32)

    def get_and_control_idx_boundary(self, mnt, idx_x, idx_y, nb_pixels_x, nb_pixels_y,
                                     reduce_mnt=True, x_win=69 // 2, y_win=79 // 2):

        reshape = True if idx_x.ndim > 1 else False

        if reduce_mnt:
            small_idx_y = idx_y[nb_pixels_y:-nb_pixels_y:, nb_pixels_x:-nb_pixels_x]
            small_idx_x = idx_x[nb_pixels_y:-nb_pixels_y:, nb_pixels_x:-nb_pixels_x]
            y_left, y_right, x_left, x_right = self._get_window_idx_boundaries(small_idx_x, small_idx_y,
                                                                               reshape=reshape, x_win=x_win,
                                                                               y_win=y_win)
            shape = small_idx_y.shape
        else:
            shape = mnt.shape
            y_left, y_right, x_left, x_right = self._get_window_idx_boundaries(idx_x, idx_y, x_win=x_win, y_win=y_win,
                                                                               reshape=reshape)

        mnt, y_left, y_right, x_left, x_right = change_several_dtype_if_required(
            [mnt, y_left, y_right, x_left, x_right], [np.float32, np.int32, np.int32, np.int32, np.int32])
        boundaries_mnt = [shape[0], shape[0], shape[1], shape[1]]
        y_left, y_right, x_left, x_right = self._control_idx_boundaries([y_left, y_right, x_left, x_right],
                                                                        min_idx=[0, 0, 0, 0],
                                                                        max_idx=boundaries_mnt)
        return y_left, y_right, x_left, x_right, shape

    @print_func_executed_decorator("std_slicing_numba", level_begin="__", level_end="__", end="")
    @timer_decorator("std_slicing_numba", unit='minute', level="....")
    def std_slicing_numba(self, mnt, y_left, y_right, x_left, x_right):
        mnt, y_left, y_right, x_left, x_right = change_several_dtype_if_required([mnt, y_left, y_right, x_left, x_right],
                                                                                 [np.float32, np.int32, np.int32, np.int32, np.int32])
        _std_slicing_numba = jit([float32[:](float32[:, :], int32[:], int32[:], int32[:], int32[:])], nopython=True)(
            self.std_slicing_numpy_loop)
        return _std_slicing_numba(mnt, y_left, y_right, x_left, x_right)

    @staticmethod
    def std_slicing_numpy_list_comprehension(mnt, y_left, y_right, x_left, x_right):
        return np.array([np.std(mnt[i1:j1 + 1, i2:j2 + 1]) for i1, j1, i2, j2 in zip(y_left, y_right, x_left, x_right)])

    @print_func_executed_decorator("xsi_helbig_map", level_begin="__", level_end="__", end="")
    @timer_decorator("xsi_helbig_map", unit='minute', level="....")
    @change_dtype_if_required_decorator(np.float32)
    def xsi_helbig_map(self, mnt, mu, idx_x=None, idx_y=None, reduce_mnt=True, x_win=69 // 2, y_win=79 // 2,
                       nb_pixels_x=100, nb_pixels_y=100, library="numba", verbose=True):

        if idx_x is None and idx_y is None:
            idx_x, idx_y = self._idx_from_array_shape(mnt)

        y_left, y_right, x_left, x_right, shape = self.get_and_control_idx_boundary(mnt, idx_x, idx_y,
                                                                                    nb_pixels_x, nb_pixels_y,
                                                                                    reduce_mnt=reduce_mnt,
                                                                                    x_win=x_win,
                                                                                    y_win=y_win)

        if library == "numba" and _numba:
            std_flat = self.std_slicing_numba(mnt, y_left, y_right, x_left, x_right)
            library = "numba"
        else:
            std_flat = self.std_slicing_numpy_list_comprehension(mnt, y_left, y_right, x_left, x_right)
            library = "numpy"

        std = std_flat.reshape((shape[0], shape[1])) if idx_x.ndim > 1 else std_flat
        print(f"__Subgrid: xsi_helbig_map. Output shape std: {std.shape}. Library: {library}") if verbose else None

        xsi = np.sqrt(2) * std / mu

        print(f"__Subgrid: xsi_helbig_map. Output shape: {xsi.shape}") if verbose else None
        return xsi

    @print_func_executed_decorator("x_sgp_topo",
                                   level_begin="__",
                                   level_end="__",
                                   end="")
    @change_dtype_if_required_decorator(np.float32)
    def x_sgp_topo_helbig_idx(self, mnt, idx_x=None, idx_y=None, dx=25, L=2_000, type_input="map", reduce_mnt=True,
                              nb_pixels_x=100, nb_pixels_y=100, x_win=69 // 2, y_win=79 // 2, verbose=True):

        a = 3.354688
        b = 1.998767
        c = 0.20286
        d = 5.951

        if idx_x is None and idx_y is None:
            idx_x, idx_y = self._idx_from_array_shape(mnt)

        mu = self.mu_helbig_average(mnt, dx, idx_x, idx_y,
                                    type_input=type_input, reduce_mnt=reduce_mnt,
                                    x_win=x_win, y_win=y_win, library="numba", verbose=verbose)

        xsi = self.xsi_helbig_map(mnt, mu, idx_x, idx_y, reduce_mnt=reduce_mnt, nb_pixels_x=nb_pixels_x,
                                  nb_pixels_y=nb_pixels_y, x_win=x_win, y_win=y_win, library="numba", verbose=verbose)

        x = 1 - (1 - (1 / (1 + a * mu ** b)) ** c) * np.exp(-d * (L / xsi) ** (-2))
        print(f"__Subgrid: x_sgp_topo_helbig_idx. Output shape: {x.shape}") if verbose else None

        return x

    @print_func_executed_decorator("subgrid parameterization from Helbig et al. 2017",
                                   level_begin="\n",
                                   level_end="",
                                   end="")
    def subgrid(self, mnt_large, dx=25, L=2_000, x_win=69 // 2, y_win=79 // 2, idx_x=None, idx_y=None, type_input="map",
                reduce_mnt=True, nb_pixels_x=100, nb_pixels_y=100, verbose=True):

        if type_input == "map":
            shape = mnt_large.shape
            all_x_idx = range(shape[1])
            all_y_idx = range(shape[0])
            idx_x, idx_y = np.array(np.meshgrid(all_x_idx, all_y_idx)).astype(np.int32)
            if verbose:
                print(f"Large mnt shape: {shape}. "
                      f"Size reduction on x: 2 * {nb_pixels_x}. "
                      f"Size reduction on x: 2 * {nb_pixels_y} ")

        reduce_mnt = False if type_input == "indexes" else reduce_mnt

        x_sgp_topo = self.x_sgp_topo_helbig_idx(mnt_large, idx_x, idx_y, dx,
                                                L=L,
                                                type_input=type_input,
                                                x_win=x_win,
                                                y_win=y_win,
                                                reduce_mnt=reduce_mnt,
                                                nb_pixels_x=nb_pixels_x,
                                                nb_pixels_y=nb_pixels_y)

        return x_sgp_topo


class DwnscHelbig(SgpHelbig):

    def __init__(self):
        super().__init__()

    @print_func_executed_decorator("downscaling from Helbig et al. 2017",
                                   level_begin="\n",
                                   level_end="",
                                   end="")
    @change_dtype_if_required_decorator(np.float32)
    def x_dsc_topo_helbig(self, mnt, dx=25, idx_x=None, idx_y=None, type_input="map", library="numba", verbose=True):

        a = 17.0393
        b = 0.737
        c = 1.0234
        d = 0.3794
        e = 1.9821

        if type_input == "map":
            laplacian = self.laplacian_map(mnt, dx, library=library, helbig=True)
            mu = self.mu_helbig_map(mnt, dx, verbose=verbose)
        elif type_input == "indexes":
            idx_x, idx_y = change_several_dtype_if_required([idx_x, idx_y], [np.int32, np.int32])
            laplacian = self.laplacian_idx(mnt, idx_x, idx_y, dx, library=library, helbig=True)
            mu = self.mu_helbig_idx(mnt, dx, idx_x, idx_y, verbose=verbose)

        term_1 = 1 - a * laplacian / (1 + a * np.abs(laplacian) ** b)
        term_2 = c / (1 + d * mu ** e)
        x = term_1 * term_2

        if verbose:
            print(f"__MNT shape: {mnt.shape}")
            print(f"__x_dsc_topo computed. x shape: {x.shape}")

        return x

    def downscale_helbig(self, mnt_large, dx=25, L=2_000, idx_x=None, idx_y=None, type_input="map", reduce_mnt=False,
                         library="numba", nb_pixels_x=100, nb_pixels_y=100, plot=True, verbose=True):

        x_sgp_topo = self.subgrid(mnt_large,
                                  idx_x=idx_x,
                                  idx_y=idx_y,
                                  dx=dx,
                                  L=L,
                                  type_input=type_input,
                                  reduce_mnt=reduce_mnt,
                                  nb_pixels_x=nb_pixels_x,
                                  nb_pixels_y=nb_pixels_y,
                                  verbose=verbose)
        if plot:
            plt.figure()
            plt.imshow(x_sgp_topo)
            plt.colorbar()
            plt.title("x_sgp_topo Helbig et al. 2017")

            mnt_small = mnt_large[nb_pixels_y:-nb_pixels_y:, nb_pixels_x:-nb_pixels_x] if reduce_mnt else mnt_large

            plt.figure()
            plt.imshow(mnt_large)
            plt.colorbar()
            plt.title("MNT")

        x_dsc_topo = self.x_dsc_topo_helbig(mnt_small,
                                            dx=dx, idx_x=idx_x, idx_y=idx_y,
                                            type_input=type_input, verbose=verbose, library=library)
        if plot:
            plt.figure()
            plt.imshow(x_dsc_topo)
            plt.colorbar()
            plt.title("x_dsc_topo Helbig et al. 2017")

            plt.figure()
            plt.imshow(x_sgp_topo * x_dsc_topo)
            plt.colorbar()
            plt.title("x_sgp_topo*x_dsc_topo Helbig et al. 2017")

        return x_sgp_topo * x_dsc_topo
