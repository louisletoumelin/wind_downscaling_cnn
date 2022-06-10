"""
Liston, G. E., & Elder, K. (2006). A meteorological distribution
system for high-resolution terrestrial modeling (MicroMet).
Journal of Hydrometeorology, 7(2), 217-234.
"""

import numpy as np
from downscale.operators.topo_utils import Topo_utils
from downscale.utils.utils_func import change_dtype_if_required, lists_to_arrays_if_required
from downscale.utils.decorators import change_dtype_if_required_decorator, print_func_executed_decorator


class SlopeMicroMet(Topo_utils):

    def __init__(self):
        super().__init__()

    @staticmethod
    @change_dtype_if_required_decorator(np.float32)
    def terrain_slope_map(mnt, dx, verbose=True):
        """
        tan-1[ { (dz/dx)**2 + (dz/dy)**2 }**(1/2) ]
        """
        print("____Library: numpy") if verbose else None
        return np.arctan(np.sqrt(np.sum(np.array(np.gradient(mnt, dx)) ** 2, axis=0)))

    @print_func_executed_decorator("terrain_slope_idx",
                                   level_begin="__",
                                   level_end="__",
                                   end="")
    @change_dtype_if_required_decorator(np.float32)
    def terrain_slope_idx(self, mnt, dx, idx_x, idx_y, verbose=True):
        """
        tan-1((dz/dx)**2 + (dz/dy)**2)**(1/2)

        Terrain slope following Liston and Elder (2006)
        """
        idx_x, idx_y = lists_to_arrays_if_required([idx_x, idx_y])

        print("__INFO: do not have to scale Terrain slope") if verbose else None

        return [self.terrain_slope_map(mnt[y - 2:y + 3, x - 2:x + 3], dx, verbose=False)[2, 2] for (x, y) in
                zip(idx_x, idx_y)]

    def terrain_slope(self, mnt, dx, idx_x=None, idx_y=None, verbose=True):
        if ((idx_x is None) and (idx_y is None)) or np.ndim(idx_x) > 1:
            return self.terrain_slope_map(mnt, dx, verbose=verbose)
        else:
            return self.terrain_slope_idx(mnt, dx, idx_x, idx_y, verbose=verbose)


class AzimuthMicroMet(Topo_utils):

    def __init__(self):
        super().__init__()

    @staticmethod
    @change_dtype_if_required_decorator(np.float32)
    def terrain_slope_azimuth_map(mnt, dx, verbose=True):
        """
        3 pi /2 - tan-1[ (dz/dy) / (dz/dx) ]
        """
        gradient_y, gradient_x = np.gradient(mnt, dx)
        arctan_value = np.where(gradient_x != 0,
                                np.arctan(gradient_y / gradient_x),
                                np.where(gradient_y > 0,
                                         np.pi / 2,
                                         np.where(gradient_y < 0,
                                                  gradient_y,
                                                  -np.pi / 2)))

        print("____Library: numpy") if verbose else None

        return 3 * np.pi / 2 - arctan_value

    @change_dtype_if_required_decorator(np.float32)
    def terrain_slope_azimuth_idx(self, mnt, dx, idx_x, idx_y, verbose=True):
        """
        3 pi /2 - tan-1(dz/dy / dz/dx)
        """
        idx_x, idx_y = lists_to_arrays_if_required([idx_x, idx_y])

        print("__INFO: do not have to scale Terrain slope azimuth") if verbose else None

        return [self.terrain_slope_azimuth_map(mnt[y - 2:y + 3, x - 2:x + 3], dx, verbose=False)[2, 2] for (x, y) in
              zip(idx_x, idx_y)]

    def terrain_slope_azimuth(self, mnt, dx, idx_x, idx_y, verbose=True):
        if ((idx_x is None) and (idx_y is None)) or np.ndim(idx_x) > 1:
            return self.terrain_slope_azimuth_map(mnt, dx, verbose=verbose)
        else:
            return self.terrain_slope_azimuth_idx(mnt, dx, idx_x, idx_y, verbose=verbose)


class CurvatureMicroMet(Topo_utils):

    def __init__(self):
        super().__init__()

    @staticmethod
    @change_dtype_if_required_decorator(np.float32)
    def get_length_scale_curvature(mnt):
        std_mnt = np.nanstd(mnt)
        return 2 * std_mnt if std_mnt != 0 else 1

    @change_dtype_if_required_decorator(np.float32)
    def curvature_map(self, mnt, length_scale=None, scaling_factor=None, scale=False, verbose=True):
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

        length_scale = self.get_length_scale_curvature(mnt) if length_scale is None else length_scale

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

        if scale:
            scaling_factor = np.nanmax(np.abs(curvature[1:-1, 1:-1])) if scaling_factor is None else scaling_factor
            curvature = curvature / (2 * scaling_factor)
            assert np.all(-0.5 <= curvature[1:-1, 1:-1])
            assert np.all(curvature[1:-1, 1:-1] <= 0.5)
            print("___curvature is now scaled between -0.5 and 0.5 "
                  "(assumption not true on the borders)") if verbose else None

        print("____Library: numpy") if verbose else None
        return curvature

    @change_dtype_if_required_decorator(np.float32)
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
            curvature_map = self.curvature_map(mnt,
                                               scale=scale, length_scale=length_scale,
                                               scaling_factor=scaling_factor, verbose=verbose)
            print("____Method: Safe (scaling factor calculate on the all map)") if verbose else None
            return curvature_map[idx_y, idx_x]
        else:
            length_scale = self.get_length_scale_curvature(mnt)
            if np.ndim(idx_x) == 0:
                idx_x = [idx_x]
                idx_y = [idx_y]
            curvature = [self.curvature_map(mnt[y-2:y+3, x-2:x+3],
                                            scale=False,
                                            length_scale=length_scale,
                                            verbose=False)[2, 2] for (x, y) in zip(idx_x, idx_y)]
            curvature = np.array(curvature, dtype=np.float32)

            if scale and (np.any(curvature < -0.5) or np.any(curvature > 0.5)):
                scaling_factor = np.nanmax(np.abs(curvature[1:-1, 1:-1])) if scaling_factor is None else scaling_factor
                curvature = curvature / (2 * scaling_factor)
                assert np.all(-0.5 <= curvature[1:-1, 1:-1])
                assert np.all(curvature[1:-1, 1:-1] <= 0.5)
                print("__Used scaling in curvature_idx. Method: NOT safe "
                      "(scaling factor not defined on the all map)") if verbose else None
            else:
                print("__WARNING: Did not scale curvature") if verbose else None

            print("__Curvature indexes selected. Method: NOT safe") if verbose else None

            return curvature

    def curvature(self, mnt, idx_x=None, idx_y=None,
                  method="safe", scale=False, length_scale=None, scaling_factor=None, verbose=True):
        if ((idx_x is None) and (idx_y is None)) or np.ndim(idx_x) > 1:
            return self.curvature_map(mnt, length_scale=length_scale, scaling_factor=scaling_factor,
                                      scale=scale, verbose=verbose)
        else:
            return self.curvature_idx(mnt, idx_x, idx_y,
                                      method=method, scale=scale, length_scale=length_scale,
                                      scaling_factor=scaling_factor, verbose=verbose)


class MicroMet(SlopeMicroMet, AzimuthMicroMet, CurvatureMicroMet):

    def __init__(self):
        super().__init__()

    @change_dtype_if_required_decorator(np.float32)
    def _omega_s_map(self, mnt, dx, wind_dir, scale=False, scaling_factor=None, verbose=True):
        """The slope in the direction of the wind"""
        beta = self.terrain_slope_map(mnt, dx, verbose=verbose)
        xi = self.terrain_slope_azimuth_map(mnt, dx, verbose=verbose)
        omega_s = beta * np.cos(wind_dir - xi)

        if scale:
            scaling_factor = np.nanmax(np.abs(omega_s[1:-1, 1:-1])) if scaling_factor is None else scaling_factor
            omega_s = omega_s / (2 * scaling_factor)
            assert np.all(-0.5 <= omega_s[-1:1, -1:1])
            assert np.all(omega_s[-1:1, -1:1] <= 0.5)

        print("____Library: numpy") if verbose else None

        return omega_s

    def _omega_s_idx(self, mnt, dx, wind_dir, idx_x, idx_y, method="safe", scale=False, scaling_factor=None, verbose=True):
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
            omega_s_idx_ = [self.omega_s_map(mnt[y - 2:y + 3, x - 2:x + 3], dx, wind_dir, scale=False, verbose=False)[2, 2] for (x, y) in
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

    def omega_s(self, mnt, dx, wind_dir, idx_x=None, idx_y=None, method="safe", scale=False, scaling_factor=None, verbose=True):
        if ((idx_x is None) and (idx_y is None)) or np.ndim(idx_x) > 1:
            return self._omega_s_map(mnt, dx, wind_dir, scale=scale, scaling_factor=scaling_factor, verbose=verbose)
        else:
            return self._omega_s_idx(mnt, dx, wind_dir, idx_x, idx_y,
                                    method=method, scale=scale, scaling_factor=scaling_factor, verbose=verbose)

    @change_dtype_if_required_decorator(np.float32)
    def wind_weighting_factor(self, mnt, dx, wind_dir, idx_x=None, idx_y=None, gamma_s=0.58, gamma_c=0.42,
                              method="safe", scale=True, length_scale=None, scaling_factor=None, verbose=True):
        """gamma_s = 0.58 and gamma_c = 0.42"""
        idx_x, idx_y = lists_to_arrays_if_required([idx_x, idx_y])
        omega_c = self.curvature(mnt, idx_x=idx_x, idx_y=idx_y, method=method, scale=scale,
                                 length_scale=length_scale, scaling_factor=scaling_factor, verbose=verbose)
        omega_s = self.omega_s(mnt, dx, wind_dir, idx_x=idx_x, idx_y=idx_y, method=method, scale=scale,
                               scaling_factor=scaling_factor, verbose=verbose)

        print("____Library: numpy") if verbose else None

        return 1 + gamma_s * omega_s + gamma_c * omega_c

    @change_dtype_if_required_decorator(np.float32)
    def diverting_factor(self, mnt, dx, wind_dir, idx_x=None, idx_y=None,
                         method="safe", scale=False, scaling_factor=None, verbose=True):
        """Need test map and idx"""
        idx_x, idx_y = lists_to_arrays_if_required([idx_x, idx_y])
        omega_s = self.omega_s(mnt, dx, wind_dir, idx_x=idx_x, idx_y=idx_y, method=method,
                                    scale=scale, scaling_factor=scaling_factor, verbose=verbose)
        azimuth = self.terrain_slope_azimuth(mnt, dx, idx_x, idx_y, verbose=verbose)

        print("____Library: numpy") if verbose else None

        return -0.5 * omega_s * np.sin(2 * (azimuth - wind_dir))
