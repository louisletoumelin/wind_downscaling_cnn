import numpy as np
from time import time as t

from downscale.utils.utils_func import change_dtype_if_required
from downscale.utils.decorators import print_func_executed_decorator, timer_decorator, change_dtype_if_required_decorator
from downscale.utils.context_managers import timer_context

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


class Wind_utils:

    def __init__(self):
        pass

    @staticmethod
    def apply_log_profile(z_in=None, z_out=None, wind_in=None, z0=None, library="numexpr",
                          verbose=True, z_in_verbose=None, z_out_verbose=None):

        """
        Apply log profile to a wind time serie.

        Parameters
        ----------
        z_in : Input elevation
        z_out : Output elevation
        wind_in : Input wind
        z0 : Roughness length for momentum
        verbose: boolean
        z_in_verbose: str
        z_out_verbose: str

        Returns
        -------
        wind_out: Output wind, after logarithmic profile
        """

        print(f"Applied log profile: {z_in_verbose} => {z_out_verbose}") if verbose else None

        if _numexpr and library == "numexpr":
            return ne.evaluate("(wind_in * log(z_out / z0)) / log(z_in / z0)")
        else:
            return (wind_in * np.log(z_out / z0)) / np.log(z_in / z0)

    def _exposed_wind_speed_xarray(self, xarray_data, z_out=10):
        dims = xarray_data.Wind.dims
        return xarray_data.assign(exp_Wind=(dims,
                                            self._exposed_wind_speed_num(wind_speed=xarray_data.Wind.values,
                                                                         z_out=z_out,
                                                                         z0=xarray_data.Z0.values,
                                                                         z0_rel=xarray_data.Z0REL.values)))

    def _exposed_wind_speed_num(self, wind_speed=None, z_out=None, z0=None, z0_rel=None, library="numexpr"):
        if self._numexpr and library == "numexpr":
            acceleration_factor = ne.evaluate(
                "log((z_out) / z0) * (z0 / (z0_rel+z0))**0.0706 / (log((z_out) / (z0_rel+z0)))")
            acceleration_factor = ne.evaluate("where(acceleration_factor > 0, acceleration_factor, 1)")
            exp_Wind = ne.evaluate("wind_speed * acceleration_factor")

        else:
            acceleration_factor = np.log(z_out / z0) * (z0 / (z0_rel + z0)) ** 0.0706 / (
                np.log(z_out / (z0_rel + z0)))
            acceleration_factor = np.where(acceleration_factor > 0, acceleration_factor, 1)
            exp_Wind = wind_speed * acceleration_factor

        return exp_Wind, acceleration_factor

    @print_func_executed_decorator("expose wind speed", level_begin="\n__", level_end="__")
    @timer_decorator("expose wind speed", unit="second", level=". . . . ")
    def exposed_wind_speed(self, wind_speed=None, z_out=None, z0=None, z0_rel=None,
                           library="numexpr", xarray_data=None, level_verbose="____", verbose=True):
        """
        Expose wind speed (deoperate subgrid parameterization from NWP)

        Parameters
        ----------
        wind_speed : ndarray
            Wind speed from NWP
        z_out : ndarray
            Elevation at which desinfluencing is performed
        z0 : ndarray
            Roughness length for momentum
        z0_rel : ndarray
            Roughness length for momentum associated with mean topography
        apply_directly_to_nwp : boolean, optional
            If True, updates the nwp directly by adding a new variable (Default: False)
        library: str
        verbose: boolean

        Returns
        -------
        exp_Wind: Unexposed wind
        acceleration_factor: Acceleration related to unexposition (usually >= 1)
        """

        if library == "xarray":
            return self._exposed_wind_speed_xarray(xarray_data, z_out=z_out)

        else:
            return self._exposed_wind_speed_num(wind_speed=wind_speed, z_out=z_out, z0=z0,
                                                z0_rel=z0_rel, library=library)

    @staticmethod
    def _scale_wind_for_ideal_case(wind_speed, wind_dir, input_speed, input_dir, verbose=True):
        """
        Specify specific wind speed and direction for ideal cases.

        Parameters
        ----------
        wind_speed : array
        wind_dir : array
        input_speed : constant
        input_dir : constant

        Returns
        -------
        wind_speed : array
            Constant wind speed.
        wind_direction : array
            Constant wind direction.
        """
        wind_speed = np.full(wind_speed.shape, input_speed)
        wind_dir = np.full(wind_dir.shape, input_dir)
        print("__Wind speed scaled for ideal cases") if verbose else None
        return wind_speed, wind_dir

    @print_func_executed_decorator("wind speed ratio (acceleration)", level_begin="\n__", level_end="__")
    @timer_decorator("wind speed ratio (acceleration)", unit="second", level=". . . . ")
    def wind_speed_ratio(self, num=None, den=None, library="numexpr", verbose=True):
        if self._numexpr and library == "numexpr":
            print("____Library: Numexpr") if verbose else None
            a1 = ne.evaluate("where(den > 0, num / den, 1)")
        else:
            print("____Library: Numpy") if verbose else None
            a1 = np.where(den > 0, num / den, 1)
        return a1

    def _3D_wind_speed(self, U=None, V=None, W=None, out=None, library="numexpr", verbose_level="____", verbose=True):
        """
        UVW = np.sqrt(U**2 + V**2 + W**2)

        Parameters
        ----------
        U : dnarray
            Horizontal wind speed component U
        V : string, optional
            Horizontal wind speed component V
        W : string, optional
            Vertical wind speed component W
        out : ndarray, optional
            If specified, the output of the caluclation is directly written in out,
            which is best for memory consumption (Default: None)

        Returns
        -------
        UVW : ndarray
            Wind speed
        """

        if verbose:
            t0 = t()

        if out is None:
            if self._numexpr and library == "numexpr":
                print("____Library: Numexpr") if verbose else None
                wind_speed = ne.evaluate("sqrt(U**2 + V**2 + W**2)")
            else:
                print("____Library: Numpy") if verbose else None
                wind_speed = np.sqrt(U ** 2 + V ** 2 + W ** 2)

            if verbose:
                print(f"{verbose_level}Wind speed computed")
                print(f". . . . Time to calculate wind speed: {np.round(t() - t0)} seconds")

            return wind_speed

        else:
            if self._numexpr and library == "numexpr":
                ne.evaluate("sqrt(U**2 + V**2 + W**2)", out=out)
            else:
                np.sqrt(U ** 2 + V ** 2 + W ** 2, out=out)

            if verbose:
                print(f"{verbose_level}Wind speed computed")
                print(f". . . . Time to calculate wind speed: {np.round(t()-t0)}")

    def _2D_wind_speed(self, U=None, V=None, out=None, library="numexpr", verbose_level="____", verbose=True):
        """
        UV = np.sqrt(U**2 + V**2)

        Parameters
        ----------
        U : dnarray
            Horizontal wind speed component U
        V : string, optional
            Horizontal wind speed component V
        out : ndarray, optional
            If specified, the output of the caluclation is directly written in out,
            which is best for memory consumption (Default: None)

        Returns
        -------
        UV : ndarray
            Wind speed
        """

        if verbose:
            t0 = t()

        if out is None:
            if self._numexpr and library == "numexpr":
                return ne.evaluate("sqrt(U**2 + V**2)")
            else:
                return np.sqrt(U ** 2 + V ** 2)

        else:
            if self._numexpr and library == "numexpr":
                ne.evaluate("sqrt(U**2 + V**2)", out=out)
            else:
                np.sqrt(U ** 2 + V ** 2, out=out)

            if verbose:
                print(f"{verbose_level}Wind speed computed")
                print(f". . . . Time to calculate wind speed: {np.round(t()-t0)}")

    def _compute_wind_speed_num(self, U=None, V=None, W=None, verbose=True):
        if W is None:
            wind_speed = self._2D_wind_speed(U=U, V=V, verbose=verbose)
        else:
            wind_speed = self._3D_wind_speed(U=U, V=V, W=W, verbose=verbose)
        return wind_speed

    @change_dtype_if_required_decorator(np.float32)
    def _compute_wind_speed_num_out(self, U=None, V=None, W=None, out=None, verbose=True):
        if W is None:
            self._2D_wind_speed(U=U, V=V, out=out, verbose=verbose)
        else:
            self._3D_wind_speed(U=U, V=V, W=W, out=out, verbose=verbose)

    def compute_wind_speed_xarray(self, xarray_data, u_name="U", v_name="V", verbose=True):
        dims = xarray_data[u_name].dims
        return xarray_data.assign(Wind=(dims, self._compute_wind_speed_num(xarray_data[u_name].values,
                                                                           xarray_data[v_name].values,
                                                                      verbose=verbose)))

    def compute_wind_speed(self, U=None, V=None, W=None,
                           library='num', out=None,
                           xarray_data=None, u_name="U", v_name="V",
                           verbose=True, time_level=". . . . ", name_to_print="compute wind speed", unit="second"):
        """
        Calculates wind speed from wind speed components.

        First detects the number of wind component then calculates wind speed.
        The calculation can be performed on numexpr, numpy or xarray dataset

        Parameters
        ----------
        U : dnarray
            Horizontal wind speed component U
        V : string, optional
            Horizontal wind speed component V
        W : string, optional
            Vertical wind speed component V
        out : ndarray, optional
            If specified, the output of the calculation is directly written in out,
            which is best for memory consumption (Default: None)
        library : str, optional
            Select the librairie to use for calculation. If 'num' first test numexpr, if not available select numpy.
            If 'xarray', a xarray dataframe needs to be specified with the corresponding names for wind components.
            (Default: 'num')

        Returns
        -------
        UV : ndarray
            Wind speed
        """
        # Numexpr or numpy
        with timer_context(name_to_print, level=time_level, unit=unit, verbose=verbose):
            if library == 'num':
                if out is None:
                    return self._compute_wind_speed_num(U=U, V=V, W=W, verbose=verbose)
                else:
                    self._compute_wind_speed_num_out(U=U, V=V, W=W, out=out, verbose=verbose)

            if library == 'xarray':
                return self.compute_wind_speed_xarray(xarray_data, u_name=u_name, v_name=v_name, verbose=verbose)

    def compute_speed_and_direction_xarray(self, xarray_data, u_name="U", v_name="V", verbose=True):

        assert u_name in xarray_data
        assert v_name in xarray_data

        xarray_data = self.compute_wind_speed(library="xarray", xarray_data=xarray_data,
                                              u_name=u_name, v_name=v_name, verbose=verbose)
        xarray_data = self.direction_from_u_and_v(library="xarray", xarray_data=xarray_data,
                                                  u_name=u_name, v_name=v_name, verbose=verbose)
        print("____Wind and Wind_DIR calculated on xarray") if verbose else None

        return xarray_data

    @print_func_executed_decorator("wind speed scaling", level_begin="\n__", level_end="__")
    @timer_decorator("wind speed scaling", unit="second", level=". . . . ")
    def wind_speed_scaling(self, scaling_wind, prediction, type_scaling="linear", library="numexpr", verbose=True):
        """
        Linear: scaling_wind * prediction / 3

        Decrease acceleration and deceleration:
        Arctan_30_1: 30*np.arctan(scaling_wind/30) * prediction / 3

        Decrease acceleration only:
        Arctan_30_2: 30*np.arctan((scaling_wind* prediction / 3)/30)

        Parameters
        ----------
        scaling_wind : ndarray
            Scaling wind (ex: NWP wind)
        prediction : ndarray
            CNN ouptut
        linear : boolean, optional
            Linear scaling (Default: True)

        Returns
        -------
        prediction : ndarray
            Scaled wind
        """
        scaling_wind = scaling_wind.astype(np.float32)
        prediction = prediction.astype(np.float32)
        thirty = np.float32(30)
        three = np.float32(3)

        if type_scaling == "linear":
            if self._numexpr and library == "numexpr":
                print("____Library: Numexpr")
                prediction = ne.evaluate("scaling_wind * prediction / 3")
            else:
                print("____Library: Numpy")
                prediction = scaling_wind * prediction / 3

        if type_scaling == "Arctan_30_1":
            if self._numexpr and library == "numexpr":
                prediction = ne.evaluate("30*arctan(scaling_wind/30) * prediction / 3")
            else:
                prediction = 30*np.arctan(scaling_wind/30) * prediction / 3

        if type_scaling == "Arctan_30_2":
            if self._numexpr and library == "numexpr":
                print("____Library: Numexpr")
                prediction = ne.evaluate("thirty*arctan((scaling_wind * prediction / three)/thirty)")
            else:
                print("____Library: Numpy")
                prediction = 30*np.arctan((scaling_wind * prediction / 3)/30)

        if type_scaling == "Arctan_10_2":
            if self._numexpr and library == "numexpr":
                prediction = ne.evaluate("10*arctan((scaling_wind * prediction / 3)/10)")
            else:
                prediction = 10*np.arctan((scaling_wind * prediction / 3)/10)

        if type_scaling == "Arctan_20_2":
            if self._numexpr and library == "numexpr":
                prediction = ne.evaluate("30*arctan((scaling_wind * prediction / 3)/30)")
            else:
                prediction = 20*np.arctan((scaling_wind * prediction / 3)/20)

        if type_scaling == "Arctan_38_2_2":
            if self._numexpr and library == "numexpr":
                prediction = ne.evaluate("38.2*arctan((scaling_wind * prediction / 3)/38.2)")
            else:
                prediction = 38.2*np.arctan((scaling_wind * prediction / 3)/38.2)

        if type_scaling == "Arctan_40_2":
            if self._numexpr and library == "numexpr":
                prediction = ne.evaluate("30*arctan((scaling_wind * prediction / 3)/30)")
            else:
                prediction = 40*np.arctan((scaling_wind * prediction / 3)/40)

        if type_scaling == "Arctan_50_2":
            if self._numexpr and library == "numexpr":
                prediction = ne.evaluate("30*arctan((scaling_wind * prediction / 3)/30)")
            else:
                prediction = 50*np.arctan((scaling_wind * prediction / 3)/50)

        prediction = change_dtype_if_required(prediction, np.float32)
        print(f"____Type {type_scaling}") if verbose else None
        return prediction

    @change_dtype_if_required_decorator(np.float32)
    @print_func_executed_decorator("computing angular deviation", level_begin="____", level_end="____")
    @timer_decorator("computing angular deviation", unit="second", level=". . . . ")
    def angular_deviation(self, U, V, unit_output="radian", library="numexpr", verbose=True):
        """
        Angular deviation from incoming flow.

        The incoming flow is supposed from the West so that V=0. If deviated, V != 0.
        The angular deviation is then np.arctan(V / U)

        Parameters
        ----------
        U : dnarray
            Horizontal wind speed component U
        V : string, optional
            Horizontal wind speed component V

        Returns
        -------
        alpha : ndarray
            Angular deviation [rad]
        """
        if self._numexpr and library == "numexpr":
            print("____Library: Numexpr") if verbose else None
            alpha = ne.evaluate("where(U == 0, where(V == 0, 0, V/abs(V) * 3.14159 / 2), arctan(V / U))")
        else:
            print("____Library: Numpy") if verbose else None
            alpha = np.where(U == 0,
                             np.where(V == 0, 0, np.sign(V) * np.pi / 2),
                             np.arctan(V / U))

        if unit_output == "degree":
            alpha = self._rad_to_deg_num(alpha)

        return alpha

    @change_dtype_if_required_decorator(np.float32)
    @print_func_executed_decorator("computing wind direction from angular deviation", level_begin="____", level_end="____")
    @timer_decorator("computing wind direction from angular deviation", unit="second", level=". . . . ")
    def direction_from_alpha(self, wind_dir, alpha, unit_direction ="degree", unit_alpha="radian", unit_output="radian",
                             library="numexpr", verbose=True):
        """
        wind_dir - alpha

        Wind direction modified by angular deviation due to wind/topography interaction.
        Warning: this function might return a new wind direction in a rotated coordinates.

        Parameters
        ----------
        wind_dir : dnarray
            Initial NWP wind direction
        alpha : dnarray
            Angular deviation
        input_dir_in_degree : boolean, optionnal
            If True, converts the input wind direction from radans to degrees (Default: True)

        Returns
        -------
        alpha : ndarray
            Modified wind direction
        """
        if unit_direction == "degree":
            wind_dir = self._deg_to_rad_num(wind_dir, library="numexpr")

        if unit_alpha == "degree":
            alpha = self._deg_to_rad_num(alpha, library="numexpr")

        if self._numexpr and library == "numexpr":
            print("____Library: Numexpr") if verbose else None
            UV_DIR = ne.evaluate("wind_dir - alpha")
        else:
            print("____Library: Numpy") if verbose else None
            UV_DIR = wind_dir - alpha

        if unit_output == "degree":
            UV_DIR = self._rad_to_deg_num(UV_DIR)

        return UV_DIR

    @change_dtype_if_required_decorator(np.float32)
    def _u_zonal_component(self, UV=None, UV_DIR=None, library="numexpr", unit_direction="degree", verbose=True):

        assert unit_direction == "radian"

        if self._numexpr and library == "numexpr":
            print(f"____Library: {library}") if verbose else None
            return ne.evaluate("-sin(UV_DIR) * UV")
        else:
            print("____Library: numpy") if verbose else None
            return -np.sin(UV_DIR) * UV

    @change_dtype_if_required_decorator(np.float32)
    def _v_meridional_component(self, UV=None, UV_DIR=None, library="numexpr", unit_direction="degree", verbose=True):

        assert unit_direction == "radian"

        if self._numexpr and library == "numexpr":
            print(f"____Library: {library}") if verbose else None
            return ne.evaluate("-cos(UV_DIR) * UV")
        else:
            print("____Library: numpy") if verbose else None
            return -np.cos(UV_DIR) * UV

    @print_func_executed_decorator("degree to radians", level_begin="________", level_end="________")
    @timer_decorator("converting degrees to radians", unit="second", level=". . . . ")
    @change_dtype_if_required_decorator(np.float32)
    def _deg_to_rad_num(self, UV_DIR, library="numexpr"):
        if self._numexpr and library == "numexpr":
            return ne.evaluate("3.14159 * UV_DIR/180")
        else:
            return np.deg2rad(UV_DIR)

    @print_func_executed_decorator("radians to degree", level_begin="________", level_end="________")
    @timer_decorator("converting radians to degrees", unit="second", level=". . . . ")
    @change_dtype_if_required_decorator(np.float32)
    def _rad_to_deg_num(self, UV_DIR, library="numexpr"):
        if self._numexpr and library == "numexpr":
            return ne.evaluate("180 * UV_DIR/3.14159")
        else:
            return np.rad2deg(UV_DIR)

    def _horizontal_wind_component_num(self, UV, UV_DIR, library="numexpr", unit_direction="degree", verbose=True):
        if unit_direction == "degree":
            UV_DIR = self._deg_to_rad_num(UV_DIR, library=library)
            unit_direction = "radian"
        U = self._u_zonal_component(UV=UV, UV_DIR=UV_DIR, library=library, unit_direction=unit_direction,
                                    verbose=verbose)
        V = self._v_meridional_component(UV=UV, UV_DIR=UV_DIR, library=library, unit_direction=unit_direction,
                                         verbose=verbose)
        return U, V

    def _horizontal_wind_component_xarray(self, xarray_data, wind_name="UV", wind_dir_name="UV_DIR",
                                          unit_direction="degree", verbose=True):
        if unit_direction == "degree":
            xarray_data = xarray_data.assign(theta=lambda x: (np.pi / 180) * (x[wind_dir_name] % 360))
        xarray_data = xarray_data.assign(U=lambda x: -x[wind_name] * np.sin(x["theta"]))
        xarray_data = xarray_data.assign(V=lambda x: -x[wind_name] * np.cos(x["theta"]))
        return xarray_data

    @print_func_executed_decorator("computing horizontal wind components", level_begin="____", level_end="____")
    @timer_decorator("computing horizontal wind components", unit="second", level=". . . . ")
    def horizontal_wind_component(self, UV=None, UV_DIR=None,
                                  library='numexpr', xarray_data=None, wind_name="Wind",
                                  wind_dir_name="Wind_DIR", unit_direction="radian", verbose=True):
        """
        U = -np.sin(UV_DIR) * UV
        V = -np.cos(UV_DIR) * UV

        Computes U and V component from wind direction and wind speed

        Parameters
        ----------
        UV : dnarray
            Wind speed
        UV_DIR : dnarray
            Wind_direction
        working_with_xarray : boolean, optionnal
            If True, computes U and V on an xarray dataframe (provided with names of variables)
            and returns the dataframe

        Returns
        -------
        U : dnarray
            Horizontal wind speed component U
        V : string, optional
            Horizontal wind speed component V

        References
        ----------
        Liston, G. E., & Elder, K. (2006). A meteorological distribution system for high-resolution terrestrial modeling (MicroMet). Journal of Hydrometeorology, 7(2), 217-234.
        """
        if library == "xarray":
            print("____Library: xarray") if verbose else None
            return self._horizontal_wind_component_xarray(xarray_data,
                                                          wind_name=wind_name,
                                                          wind_dir_name=wind_dir_name,
                                                          unit_direction=unit_direction,
                                                          verbose=False)
        else:
            return self._horizontal_wind_component_num(UV, UV_DIR, library=library,
                                                       unit_direction=unit_direction,
                                                       verbose=verbose)

    @change_dtype_if_required_decorator(np.float32)
    def _direction_from_u_and_v_num(self, U, V, library="numexpr", verbose=True):
        if self._numexpr and library == "numexpr":
            print("____Library: Numexpr") if verbose else None
            return ne.evaluate("(180 + (180/3.14159)*arctan2(U,V)) % 360")
        else:
            print("____Library: Numpy") if verbose else None
            return np.mod(180 + np.rad2deg(np.arctan2(U, V)), 360)

    def _direction_from_u_and_v_xarray(self, xarray_data=None, u_name="U", v_name="V"):
        dims = xarray_data[u_name].dims
        return xarray_data.assign(Wind_DIR=(dims,
                                            self._direction_from_u_and_v_num(xarray_data[u_name].values,
                                                                             xarray_data[v_name].values,
                                                                             verbose=False)))

    def _direction_from_u_and_v_degree(self, U=None, V=None, library="numexpr", verbose=True,
                                       xarray_data=None, u_name="U", v_name="V"):
        if library == "xarray":
            print("____Library: xarray") if verbose else None
            return self._direction_from_u_and_v_xarray(xarray_data, u_name=u_name, v_name=v_name)
        else:
            return self._direction_from_u_and_v_num(U=U, V=V, library=library, verbose=verbose)

    def direction_from_u_and_v(self, U=None, V=None, unit_output="degree",
                               library="numexpr", xarray_data=None, u_name="U", v_name="V",
                               verbose=True, name_to_print="direction from u and v", time_level=". . . . ",
                               unit="second"):
        """
        UV_DIR = 180 + (180/3.14159)*arctan2(U,V)) % 360

        Compute wind direction from U and V components

        Parameters
        ----------
        U : dnarray
            Horizontal wind speed component U
        V : string, optional
            Horizontal wind speed component V
        output_in_degree : boolean, optionnal
            If True, the wind direction is converted to degrees

        Returns
        -------
        UV_DIR : dnarray
            Wind direction
        """
        with timer_context(name_to_print, level=time_level, unit=unit, verbose=verbose):
            if unit_output == "degree":
                return self._direction_from_u_and_v_degree(U=U, V=V, library=library,
                                                           xarray_data=xarray_data, u_name=u_name, v_name=v_name,
                                                           verbose=verbose)
            else:
                raise NotImplementedError
