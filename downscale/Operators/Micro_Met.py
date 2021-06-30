"""
Liston, G. E., & Elder, K. (2006). A meteorological distribution
system for high-resolution terrestrial modeling (MicroMet).
Journal of Hydrometeorology, 7(2), 217-234.
"""

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


class MicroMet(Topo_utils):

    def __init__(self):
        super().__init__()

    @staticmethod
    def terrain_slope_map(mnt, dx, verbose=True):
        """Terrain slope following Liston and Elder (2006)"""
        beta = np.arctan(np.sqrt(np.sum(np.array(np.gradient(mnt, dx)) ** 2, axis=0)))
        beta = change_dtype_if_required(beta, np.float32)
        print("__Terrain slope calculation using numpy") if verbose else None
        return beta

    def terrain_slope_idx(self, mnt, dx, idx_x, idx_y, verbose=True):
        beta = self.terrain_slope_map(mnt, dx)
        print("__Selecting indexes on terrain slope") if verbose else None
        return beta[idx_y, idx_x]

    @staticmethod
    def terrain_slope_azimuth_map(mnt, dx, verbose=True):
        gradient_y, gradient_x = np.gradient(mnt, dx)
        arctan_value = np.where(gradient_x != 0, np.arctan(gradient_y / gradient_x), np.where(gradient_y>0, np.pi/2, np.where(gradient_y < 0, gradient_y, -np.pi/2)))
        xi = 3 * np.pi / 2 - arctan_value

        xi = change_dtype_if_required(xi, np.float32)
        print("__Terrain slope azimuth calculation using numpy") if verbose else None

        return xi

    def terrain_slope_azimuth_idx(self, mnt, dx, idx_x, idx_y, verbose=True):
        xi = self.terrain_slope_azimuth_map(mnt, dx)

        print("__Terrain slope azimuth indexes selected") if verbose else None

        return xi[idx_y, idx_x]

    @staticmethod
    def curvature_map(mnt, length_scale=None, scale=False, verbose=True):
        if length_scale is None:
            length_scale = 2 * np.nanstd(mnt)

        s = np.roll(mnt, -1, axis=0)
        n = np.roll(mnt, 1, axis=0)
        w = np.roll(mnt, 1, axis=1)
        e = np.roll(mnt, -1, axis=1)
        s_e = np.roll(np.roll(mnt, -1, axis=1), -1, axis=0)
        n_w = np.roll(np.roll(mnt, 1, axis=1), 1, axis=0)
        s_w = np.roll(np.roll(mnt, 1, axis=1), -1, axis=0)
        n_e = np.roll(np.roll(mnt, -1, axis=1), 1, axis=0)

        curv_e_w = (mnt - (w + e) / 2) / (2 * length_scale)
        curv_n_s = (mnt - (n + s) / 2) / (2 * length_scale)
        curv_ne_sw = (mnt - (n_e + s_w) / 2) / (2 * np.sqrt(2) * length_scale)
        curv_nw_se = (mnt - (n_w + s_e) / 2) / (2 * np.sqrt(2) * length_scale)

        curvature = (curv_e_w + curv_n_s + curv_ne_sw + curv_nw_se) / 4
        curvature = change_dtype_if_required(curvature, np.float32)

        if scale:
            scaling_factor = np.nanmax(np.abs(curvature[1:-1, 1:-1]))
            curvature = curvature / (2 * scaling_factor)
            assert np.all(-0.5 <= curvature[1:-1, 1:-1])
            assert np.all(curvature[1:-1, 1:-1] <= 0.5)

        print("__Curvature calculation using numpy") if verbose else None
        return curvature

    def curvature_idx(self, mnt, idx_x, idx_y, scale=False, verbose=True):
        curvature = [self.curvature_map(mnt[y-2:y+2, x-2:x+2], scale=False, verbose=False)[3] for (x, y) in zip(idx_x, idx_y)]
        curvature = np.array(curvature)

        if scale and (np.any(curvature < -0.5) or not np.any(curvature > 0.5)):
            scaling_factor = np.nanmax(np.abs(curvature[1:-1, 1:-1]))
            curvature = curvature / (2 * scaling_factor)
            assert np.all(-0.5 <= curvature[1:-1, 1:-1])
            assert np.all(curvature[1:-1, 1:-1] <= 0.5)
            print("__Used scaling in omega_s_idx")

        print("__Curvature indexes selected") if verbose else None

        return curvature

    def omega_s_map(self, mnt, dx, wind_dir, scale=False, verbose=True):
        """The slope in the direction of the wind"""
        beta = self.terrain_slope_map(mnt, dx)
        xi = self.terrain_slope_azimuth_map(mnt, dx)
        omega_s = beta * np.cos(wind_dir - xi)
        omega_s = change_dtype_if_required(omega_s, np.float32)

        if scale:
            scaling_factor = np.nanmax(np.abs(omega_s[1:-1, 1:-1]))
            omega_s = omega_s / (2 * scaling_factor)
            assert np.all(-0.5 <= omega_s[-1:1, -1:1])
            assert np.all(-0.5 <= omega_s[-1:1, -1:1])

        print("__Omega_s calculation using numpy") if verbose else None

        return omega_s

    def omega_s_idx(self, mnt, dx, wind_dir, idx_x, idx_y, scale=False, verbose=True):

        omega_s = [self.omega_s_map(mnt[y-2:y+2, x-2:x+2], dx, wind_dir, verbose=False, scale=False)[3] for (x, y) in zip(idx_x, idx_y)]
        omega_s = np.array(omega_s)

        if scale and (np.any(omega_s < -0.5) or not np.any(omega_s > 0.5)):
            scaling_factor = np.nanmax(np.abs(omega_s[1:-1, 1:-1]))
            omega_s = omega_s / (2 * scaling_factor)
            assert np.all(-0.5 <= omega_s[1:-1, 1:-1])
            assert np.all(omega_s[1:-1, 1:-1] <= 0.5)
            print("__Used scaling in omega_s_idx")

        print("__Omega_s indexes selected") if verbose else None

        return omega_s

    def wind_weighting_factor_map(self, mnt, dx, wind_dir, gamma_s=0.58, gamma_c=0.42, verbose=True):
        """gamma_s = 0.58 and gamma_c = 0.42"""
        omega_c = self.curvature_map(mnt, length_scale=None, scale=True)
        omega_s = self.omega_s_map(mnt, dx, wind_dir, scale=True)
        w = 1 + gamma_s * omega_s + gamma_c * omega_c

        w = change_dtype_if_required(w, np.float32)
        print("__Wind weighting factor calculation using numpy") if verbose else None

        return w

    def diverting_factor_map(self, mnt, dx, wind_dir, verbose=True):

        term1 = -0.5 * self.omega_s_map(mnt, dx, wind_dir, scale=True)
        azimuth = self.terrain_slope_azimuth_map(mnt, dx)
        theta_d = term1 * np.sin(2 * (azimuth - wind_dir))

        theta_d = change_dtype_if_required(theta_d, np.float32)
        print("__Wind weighting factor calculation using numpy") if verbose else None

        return theta_d
