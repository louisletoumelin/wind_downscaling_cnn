import numpy as np
import matplotlib.pyplot as plt

from downscale.Operators.topo_utils import Topo_utils
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


# noinspection PyUnboundLocalVariable
class SgpHelbig(Topo_utils):

    def __init__(self):
        super().__init__()

    @staticmethod
    def mu_helbig_map(mnt, dx, verbose=True):
        """
        Adapted from I. Gouttevin

        From Helbig et al. 2017
        """

        mu = np.sqrt(np.sum(np.array(np.gradient(mnt, dx)) ** 2, axis=0) / 2)

        mu = change_dtype_if_required(mu, np.float32)
        print("__mu calculation using numpy") if verbose else None

        return mu

    def mu_helbig_idx(self, mnt, dx, idx_x, idx_y, verbose=True):

        mu_helbig_func = self.mu_helbig_map
        mu = [mu_helbig_func(mnt[y - 1:y + 2, x - 1:x + 2], dx, verbose=False)[1, 1] for (x, y) in zip(idx_x, idx_y)]

        mu = change_dtype_if_required(np.array(mu), np.float32)
        print("__Selecting indexes on mu") if verbose else None

        return mu

    def mu_helbig_average(self, mnt, dx, idx_x=None, idx_y=None, reduce_mnt=True, type_input="map", x_win=69 // 2,
                          y_win=79 // 2, nb_pixels_x=100, nb_pixels_y=100, library="numba", verbose=True):

        if idx_x is None and idx_y is None:
            shape = mnt.shape
            idx_x = range(shape[1])
            idx_y = range(shape[0])
            idx_x, idx_y = np.array(np.meshgrid(idx_x, idx_y)).astype(np.int32)

        if reduce_mnt:
            small_idx_y = idx_y[nb_pixels_y:-nb_pixels_y, nb_pixels_x:-nb_pixels_x]
            small_idx_x = idx_x[nb_pixels_y:-nb_pixels_y, nb_pixels_x:-nb_pixels_x]
            shape = small_idx_y.shape
            y_left, y_right, x_left, x_right = self._get_window_idx_boundaries(small_idx_x, small_idx_y, x_win=x_win,
                                                                               y_win=y_win)
        else:
            reshape = True if type_input == "map" else False
            shape = mnt.shape
            y_left, y_right, x_left, x_right = self._get_window_idx_boundaries(idx_x, idx_y, reshape=reshape,
                                                                               x_win=x_win, y_win=y_win)

        boundaries_mnt = [shape[0], shape[0], shape[1], shape[1]]
        y_left, y_right, x_left, x_right = self._control_idx_boundaries([y_left, y_right, x_left, x_right],
                                                                        min_idx=[0, 0, 0, 0],
                                                                        max_idx=boundaries_mnt)

        if type_input == "map":
            mu = self.mu_helbig_map(mnt, dx, verbose=verbose)
            if library == 'numba' and _numba:
                mu, y_left, y_right, x_left, x_right = change_several_dtype_if_required(
                    [mu, y_left, y_right, x_left, x_right], [np.float32, np.int32, np.int32, np.int32, np.int32])
                jit_mean = jit([float32[:](float32[:, :], int32[:], int32[:], int32[:], int32[:])], nopython=True)(
                    self.mean_slicing_numpy)
                mu_flat = jit_mean(mu, y_left, y_right, x_left, x_right)
                library = "numba"
            else:
                mu_flat = np.array(
                    [np.mean(mu[i1:j1 + 1, i2:j2 + 1]) for i1, j1, i2, j2 in zip(y_left, y_right, x_left, x_right)])
                library = "numpy"

        elif type_input == "indexes":
            boundaries_mnt = [mnt.shape[0], mnt.shape[0], mnt.shape[1], mnt.shape[1]]
            y_left, y_right, x_left, x_right = self._control_idx_boundaries([y_left, y_right, x_left, x_right],
                                                                            min_idx=[0, 0, 0, 0],
                                                                            max_idx=boundaries_mnt)
            mu_flat = np.array([np.mean(self.mu_helbig_map(mnt[i1:j1 + 1, i2:j2 + 1], dx, verbose=verbose))
                                for i1, j1, i2, j2 in zip(y_left, y_right, x_left, x_right)])

        mu = mu_flat.reshape((shape[0], shape[1])) if (type_input == "map" or reduce_mnt) else mu_flat

        mu = change_dtype_if_required(mu, np.float32)
        print(f"__Subgrid: computed average mu. Output shape: {mu.shape}. Library: {library}") if verbose else None

        return mu

    @staticmethod
    def mean_slicing_numpy(array, y_left, y_right, x_left, x_right):
        result = np.empty(y_left.shape)
        for index, (i1, j1, i2, j2) in enumerate(zip(y_left, y_right, x_left, x_right)):
            result[index] = np.mean(array[i1:j1 + 1, i2:j2 + 1])
        return result.astype(np.float32)

    @staticmethod
    def std_slicing_numpy(array, y_left, y_right, x_left, x_right):
        result = np.empty(y_left.shape)
        for index, (i1, j1, i2, j2) in enumerate(zip(y_left, y_right, x_left, x_right)):
            result[index] = np.std(array[i1:j1 + 1, i2:j2 + 1])
        return result.astype(np.float32)

    def xsi_helbig_map(self, mnt, mu, idx_x=None, idx_y=None, reduce_mnt=True, x_win=69 // 2, y_win=79 // 2,
                       nb_pixels_x=100, nb_pixels_y=100, library="numba", verbose=True):

        if idx_x is None and idx_y is None:
            shape = mnt.shape
            idx_x = range(shape[1])
            idx_y = range(shape[0])
            idx_x, idx_y = np.array(np.meshgrid(idx_x, idx_y)).astype(np.int32)

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
        if library == "numba" and _numba:
            std_slicing_numba = jit([float32[:](float32[:, :], int32[:], int32[:], int32[:], int32[:])], nopython=True)(
                self.std_slicing_numpy)
            std_flat = std_slicing_numba(mnt, y_left, y_right, x_left, x_right)
            library = "numba"
        else:
            std_flat = np.array(
                [np.std(mnt[i1:j1 + 1, i2:j2 + 1]) for i1, j1, i2, j2 in zip(y_left, y_right, x_left, x_right)])
            library = "numpy"

        std = std_flat.reshape((shape[0], shape[1])) if reshape else std_flat
        print(f"__Subgrid: computed average std. Output shape: {std.shape}. Library: {library}") if verbose else None

        xsi = np.sqrt(2) * std / mu

        xsi = change_dtype_if_required(xsi, np.float32)
        print(f"__Subgrid: computed average xsi. Output shape: {xsi.shape}") if verbose else None
        return xsi

    def x_sgp_topo_helbig_idx(self, mnt, idx_x=None, idx_y=None, dx=25, L=2_000, type_input="map", reduce_mnt=True,
                              nb_pixels_x=100, nb_pixels_y=100, x_win=69 // 2, y_win=79 // 2, verbose=True):

        a = 3.354688
        b = 1.998767
        c = 0.20286
        d = 5.951

        if idx_x is None and idx_y is None:
            shape = mnt.shape
            idx_x = range(shape[1])
            idx_y = range(shape[0])
            idx_x, idx_y = np.array(np.meshgrid(idx_x, idx_y)).astype(np.int32)

        mu = self.mu_helbig_average(mnt, dx, idx_x, idx_y,
                                    type_input=type_input, reduce_mnt=reduce_mnt,
                                    x_win=x_win, y_win=y_win, verbose=verbose)
        xsi = self.xsi_helbig_map(mnt, mu, idx_x, idx_y, reduce_mnt=reduce_mnt, nb_pixels_x=nb_pixels_x,
                                  nb_pixels_y=nb_pixels_y, x_win=x_win, y_win=y_win, library="numba", verbose=verbose)

        x = 1 - (1 - (1 / (1 + a * mu ** b)) ** c) * np.exp(-d * (L / xsi) ** (-2))

        x = change_dtype_if_required(x, np.float32)
        print(f"__Subgrid: computed x_sgp_topo. Output shape: {x.shape}") if verbose else None

        return x

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


# noinspection PyUnboundLocalVariable
class DwnscHelbig(SgpHelbig):

    def __init__(self):
        super().__init__()

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

        x = change_dtype_if_required(x, np.float32)
        return x

    def downscale_helbig(self, mnt_large, dx=25, L=2_000, idx_x=None, idx_y=None, type_input="map", reduce_mnt=False,
                         library="numba", nb_pixels_x=100, nb_pixels_y=100, verbose=True):

        print(f"\nBegin subgrid parameterization from Helbig et al. 2017") if verbose else None
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

        plt.figure()
        plt.imshow(x_sgp_topo)
        plt.colorbar()
        plt.title("x_sgp_topo Helbig et al. 2017")

        mnt_small = mnt_large[nb_pixels_y:-nb_pixels_y:, nb_pixels_x:-nb_pixels_x] if reduce_mnt else mnt_large

        plt.figure()
        plt.imshow(mnt_large)
        plt.colorbar()
        plt.title("MNT")

        print(f"\nBegin downscaling from Helbig et al. 2017") if verbose else None
        x_dsc_topo = self.x_dsc_topo_helbig(mnt_small,
                                            dx=dx, idx_x=idx_x, idx_y=idx_y,
                                            type_input=type_input, verbose=verbose, library=library)

        plt.figure()
        plt.imshow(x_dsc_topo)
        plt.colorbar()
        plt.title("x_dsc_topo Helbig et al. 2017")

        plt.figure()
        plt.imshow(x_sgp_topo * x_dsc_topo)
        plt.colorbar()
        plt.title("x_sgp_topo*x_dsc_topo Helbig et al. 2017")

        return x_sgp_topo * x_dsc_topo
