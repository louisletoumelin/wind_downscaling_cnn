"""
Liston, G. E., & Elder, K. (2006). A meteorological distribution
system for high-resolution terrestrial modeling (MicroMet).
Journal of Hydrometeorology, 7(2), 217-234.
"""

import numpy as np
import matplotlib.pyplot as plt
from downscale.Operators.topo_utils import Topo_utils
from downscale.Utils.Utils import change_dtype_if_required, lists_to_arrays_if_required


class MicroMet(Topo_utils):

    def __init__(self):
        super().__init__()

    @staticmethod
    def terrain_slope_map(mnt, dx, verbose=True):
        """
        tan-1((dz/dx)**2 + (dz/dy)**2)**(1/2)

        Terrain slope following Liston and Elder (2006)
        """
        beta = np.arctan(np.sqrt(np.sum(np.array(np.gradient(mnt, dx)) ** 2, axis=0)))

        beta = change_dtype_if_required(beta, np.float32)
        print("__Terrain slope calculation. Library: numpy") if verbose else None

        return beta

    def terrain_slope_idx(self, mnt, dx, idx_x, idx_y, verbose=True):
        """
        tan-1((dz/dx)**2 + (dz/dy)**2)**(1/2)

        Terrain slope following Liston and Elder (2006)
        """
        idx_x, idx_y = lists_to_arrays_if_required([idx_x, idx_y])

        beta = [self.terrain_slope_map(mnt[y - 2:y + 2, x - 2:x + 2], dx, verbose=False)[2, 2] for (x, y) in
              zip(idx_x, idx_y)]
        beta = np.array(beta)

        print("__Terrain slope indexes selected") if verbose else None
        print("__INFO: do not have to scale Terrain slope") if verbose else None

        return beta

    @staticmethod
    def terrain_slope_azimuth_map(mnt, dx, verbose=True):
        """
        3 pi /2 - tan-1(dz/dy / dz/dx)

        following Liston and Elder (2006)
        """
        gradient_y, gradient_x = np.gradient(mnt, dx)
        arctan_value = np.where(gradient_x != 0, np.arctan(gradient_y / gradient_x), np.where(gradient_y>0, np.pi/2, np.where(gradient_y < 0, gradient_y, -np.pi/2)))
        xi = 3 * np.pi / 2 - arctan_value

        xi = change_dtype_if_required(xi, np.float32)
        print("__Terrain slope azimuth calculation. Library: numpy") if verbose else None

        return xi

    def terrain_slope_azimuth_idx(self, mnt, dx, idx_x, idx_y, verbose=True):
        """
        3 pi /2 - tan-1(dz/dy / dz/dx)

        following Liston and Elder (2006)
        """
        idx_x, idx_y = lists_to_arrays_if_required([idx_x, idx_y])
        xi = [self.terrain_slope_azimuth_map(mnt[y - 2:y + 2, x - 2:x + 2], dx, verbose=False)[2, 2] for (x, y) in
                     zip(idx_x, idx_y)]
        xi = np.array(xi)

        print("__Terrain slope azimuth indexes selected") if verbose else None
        print("__INFO: do not have to scale Terrain slope azimuth") if verbose else None

        return xi

    @staticmethod
    def curvature_map(mnt, length_scale=None, scaling_factor=None, scale=False, verbose=True):
        """
        Curvature following Liston and Elder (2006)

        (1/4) * ((z-(n+s)/2)/(2*n) + (z-(e+w)/2)/(2*n) + (z-(nw+se)/2)/((2*sqrt(2)*n)) + (z-(ne+sw)/2)/(2*sqrt(2)*n))

        with n = 2*std(dem)


        Parameters
        ----------
        mnt : dnarray
            Digital elevation model (topography)
        idx_x : dnarray
            Indexes along the x axis
        idx_y : ndarray
            Indexes along the y axis
        method : string
            "safe" or anything else. If safe, first compute curvature on a map and the select indexes.
            This allows for clean scaling of curvature if abs(curvatur) > 0.5 (Liston and Elder (2006))
            If an other string is passed, the computation will be faster but the scaling might be unadequate.
        scale : boolean
            If True, scale the output such as -0.5<curvature<0.5
        length_scale : float
            Length used in curvature computation
        scaling_factor : float
            Length used to scale outputs
        verbose : boolean
            Print verbose

        Returns
        -------
        curvature : ndarray
            Curvature
        """
        if length_scale is None:
            std_mnt = np.nanstd(mnt)
            length_scale = 2 * std_mnt if std_mnt != 0 else 1

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
            scaling_factor = np.nanmax(np.abs(curvature[1:-1, 1:-1])) if scaling_factor is None else scaling_factor
            curvature = curvature / (2 * scaling_factor)
            assert np.all(-0.5 <= curvature[1:-1, 1:-1])
            assert np.all(curvature[1:-1, 1:-1] <= 0.5)

        print("__Curvature calculation. Library: numpy") if verbose else None
        return curvature

    def curvature_idx(self, mnt, idx_x, idx_y, method="safe", scale=False, length_scale=None, scaling_factor=None, verbose=True):
        """
        Curvature following Liston and Elder (2006)

        (1/4) * ((z-(n+s)/2)/(2*n) + (z-(e+w)/2)/(2*n) + (z-(nw+se)/2)/((2*sqrt(2)*n)) + (z-(ne+sw)/2)/(2*sqrt(2)*n))

        with n = 2*std(dem)


        Parameters
        ----------
        mnt : dnarray
            Digital elevation model (topography)
        idx_x : dnarray
            Indexes along the x axis
        idx_y : ndarray
            Indexes along the y axis
        method : string
            "safe" or anything else. If safe, first compute curvature on a map and the select indexes.
            This allows for clean scaling of curvature if abs(curvatur) > 0.5 (Liston and Elder (2006))
            If an other string is passed, the computation will be faster but the scaling might be unadequate.
        scale : boolean
            If True, scale the output such as -0.5<curvature<0.5
        length_scale : float
            Length used in curvature computation
        scaling_factor : float
            Length used to scale outputs
        verbose : boolean
            Print verbose

        Returns
        -------
        curvature : ndarray
            Curvature
        """
        idx_x, idx_y = lists_to_arrays_if_required([idx_x, idx_y])
        if method == "safe":
            curvature_map = self.curvature_map(mnt, scale=scale, length_scale=length_scale, scaling_factor=scaling_factor, verbose=verbose)

            print("__Curvature indexes selected. Method: Safe") if verbose else None
            return curvature_map[idx_y, idx_x]

        else:
            curvature = [self.curvature_map(mnt[y-2:y+2, x-2:x+2], scale=False, verbose=False)[3, 3] for (x, y) in zip(idx_x, idx_y)]
            curvature = np.array(curvature)

            if scale and (np.any(curvature < -0.5) or np.any(curvature > 0.5)):
                scaling_factor = np.nanmax(np.abs(curvature[1:-1, 1:-1])) if scaling_factor is None else scaling_factor
                curvature = curvature / (2 * scaling_factor)
                assert np.all(-0.5 <= curvature[1:-1, 1:-1])
                assert np.all(curvature[1:-1, 1:-1] <= 0.5)
                print("__Used scaling in curvature_idx. Method: NOT safe") if verbose else None
            else:
                print("__WARNING: Did not scale curvature") if verbose else None

            print("__Curvature indexes selected. Method: NOT safe") if verbose else None

            return curvature

    def omega_s_map(self, mnt, dx, wind_dir, scale=False, scaling_factor=None, verbose=True):
        """
        Test 3. Good result
        """
        """The slope in the direction of the wind"""
        beta = self.terrain_slope_map(mnt, dx, verbose=verbose)
        xi = self.terrain_slope_azimuth_map(mnt, dx, verbose=verbose)
        omega_s = beta * np.cos(wind_dir - xi)

        if scale:
            scaling_factor = np.nanmax(np.abs(omega_s[1:-1, 1:-1])) if scaling_factor is None else scaling_factor
            omega_s = omega_s / (2 * scaling_factor)
            assert np.all(-0.5 <= omega_s[-1:1, -1:1])
            assert np.all(-0.5 <= omega_s[-1:1, -1:1])

        omega_s = change_dtype_if_required(omega_s, np.float32)
        print("__Omega_s calculation. Library: numpy") if verbose else None

        return omega_s

    def omega_s_idx(self, mnt, dx, wind_dir, idx_x, idx_y, method="safe", scale=False, scaling_factor=None, verbose=True):
        """
        Test 3. Good result
        """
        """
        Omega_s following Liston and Elder (2006)

        Parameters
        ----------
        mnt : dnarray
            Digital elevation model (topography)
        dx : float
            Digital elevation model resolution (distance between grid points)
        idx_x : dnarray
            Indexes along the x axis
        idx_y : ndarray
            Indexes along the y axis
        method : string
            "safe" or anything else. If safe, first compute omega_s on a map and the select indexes.
            This allows for clean scaling of omega_s if abs(omega_s) > 0.5 (Liston and Elder (2006))
            If an other string is passed, the computation will be faster but the scaling might be unadequate.
        scale : boolean
            If True, scale the output such as -0.5<omega_s<0.5
        scaling_factor : float
            Length used to scale outputs
        verbose : boolean
            Print verbose

        Returns
        -------
        omega_s : ndarray
            Omega_s
        """
        idx_x, idx_y = lists_to_arrays_if_required([idx_x, idx_y])
        if method == "safe":
            omega_s_map_ = self.omega_s_map(mnt, dx, wind_dir, scale=scale, scaling_factor=scaling_factor, verbose=verbose)

            print("__Omega_s indexes selected. Method: Safe") if verbose else None
            return omega_s_map_[idx_y, idx_x]

        else:
            omega_s_idx_ = [self.omega_s_map(mnt[y - 2:y + 2, x - 2:x + 2], dx, wind_dir, scale=False, verbose=False)[3, 3] for (x, y) in
                         zip(idx_x, idx_y)]
            omega_s_idx_ = np.array(omega_s_idx_)

            if scale and (np.any(omega_s_idx_ < -0.5) or np.any(omega_s_idx_ > 0.5)):
                scaling_factor = np.nanmax(np.abs(omega_s_idx_[1:-1, 1:-1])) if scaling_factor is None else scaling_factor
                omega_s_idx_ = omega_s_idx_ / (2 * scaling_factor)
                assert np.all(-0.5 <= omega_s_idx_[1:-1, 1:-1])
                assert np.all(omega_s_idx_[1:-1, 1:-1] <= 0.5)
                print("__Used scaling in omega_s_idx. Method: NOT safe") if verbose else None
            else:
                print("__WARNING: Did not scale omega_s") if verbose else None

            print("__Omega_s indexes selected. Method: NOT safe") if verbose else None

            return omega_s_idx_

    def wind_weighting_factor_map(self, mnt, dx, wind_dir, gamma_s=0.58, gamma_c=0.42, scale=True, verbose=True):
        """
        Test 3. Good result
        """
        """gamma_s = 0.58 and gamma_c = 0.42"""
        omega_c = self.curvature_map(mnt, length_scale=None, scale=scale, verbose=verbose)
        omega_s = self.omega_s_map(mnt, dx, wind_dir, scale=scale, verbose=verbose)
        w = 1 + gamma_s * omega_s + gamma_c * omega_c

        w = change_dtype_if_required(w, np.float32)
        print("__Wind weighting factor calculation. Library: numpy") if verbose else None

        return w

    def wind_weighting_factor_idx(self, mnt, dx, wind_dir, idx_x, idx_y, gamma_s=0.58, gamma_c=0.42, method="safe",
                                  scale=True, length_scale=None, scaling_factor=None, verbose=True):
        """
        Test 3. Good result
        """
        """gamma_s = 0.58 and gamma_c = 0.42"""
        idx_x, idx_y = lists_to_arrays_if_required([idx_x, idx_y])
        omega_c = self.curvature_idx(mnt, idx_x, idx_y,
                                     length_scale=length_scale, scaling_factor=scaling_factor, method=method,
                                     scale=scale, verbose=verbose)
        omega_s = self.omega_s_idx(mnt, dx, wind_dir, idx_x, idx_y,
                                   method=method, scale=scale, scaling_factor=scaling_factor, verbose=verbose)
        w = 1 + gamma_s * omega_s + gamma_c * omega_c

        w = change_dtype_if_required(w, np.float32)
        print("__Wind weighting factor calculation. Library: numpy") if verbose else None

        return w

    def diverting_factor_map(self, mnt, dx, wind_dir, scale=True, verbose=True):
        """
        Test 3. Good result
        """

        term1 = -0.5 * self.omega_s_map(mnt, dx, wind_dir, scale=scale, verbose=verbose)
        azimuth = self.terrain_slope_azimuth_map(mnt, dx, verbose=verbose)
        theta_d = term1 * np.sin(2 * (azimuth - wind_dir))

        theta_d = change_dtype_if_required(theta_d, np.float32)
        print("__Wind weighting factor calculation. Library: numpy") if verbose else None

        return theta_d

    def diverting_factor_idx(self, mnt, dx, wind_dir, idx_x, idx_y,
                             method="safe", scale=False, scaling_factor=None, verbose=True):
        """
        Test 1. no nans OK
        Test 2. good shape OK
        Test 3. Good result
        Test 4. Good data format
        """
        idx_x, idx_y = lists_to_arrays_if_required([idx_x, idx_y])
        term1 = -0.5 * self.omega_s_idx(mnt, dx, wind_dir, idx_x, idx_y,
                                        method=method, scale=scale, scaling_factor=scaling_factor, verbose=verbose)
        azimuth = self.terrain_slope_azimuth_idx(mnt, dx, idx_x, idx_y, verbose=verbose)
        theta_d = term1 * np.sin(2 * (azimuth - wind_dir))

        theta_d = change_dtype_if_required(theta_d, np.float32)
        print("__Wind weighting factor calculation. Library: numpy.") if verbose else None

        return theta_d
