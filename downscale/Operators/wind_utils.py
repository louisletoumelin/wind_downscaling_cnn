import numpy as np

from downscale.Utils.Utils import change_dtype_if_required

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

try:
    import dask
    _dask = True
except ModuleNotFoundError:
    _dask = False

class Wind_utils:
    _numexpr = _numexpr
    _numba = _numba
    _dask = _dask

    def __init__(self):
        pass
    
    @staticmethod
    def apply_log_profile(z_in=None, z_out=None, wind_in=None, z0=None,
                          verbose=True, z_in_verbose=None, z_out_verbose=None):

        """
        Apply log profile to a wind time serie.

        Parameters
        ----------
        z_in : Input elevation
        z_out : Output elevation
        wind_in : Input wind
        z0 : Roughness length for momentum

        Returns
        -------
        wind_out: Output wind, after logarithmic profile
        """

        if verbose: print(f"Applied log profile: {z_in_verbose} => {z_out_verbose}")

        if _numexpr:
            return (ne.evaluate("(wind_in * log(z_out / z0)) / log(z_in / z0)"))
        else:
            return ((wind_in * np.log(z_out / z0)) / np.log(z_in / z0))
        
    def exposed_wind_speed(self, wind_speed=None, z_out=None, z0=None, z0_rel=None, apply_directly_to_nwp=False, verbose=True):
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

        Returns
        -------
        exp_Wind: Unexposed wind
        acceleration_factor: Acceleration related to unexposition (usually >= 1)
        """

        if apply_directly_to_nwp:
            self.nwp.data_xr = self.nwp.data_xr.assign(
                exp_Wind=lambda x: x.Wind * (np.log(10 / x.Z0) / np.log(10 / x.Z0REL)) * (x.Z0 / x.Z0REL) ** (0.0708))

        if not (apply_directly_to_nwp):

            if self._numexpr:
                acceleration_factor = ne.evaluate(
                    "log((z_out) / z0) * (z0 / (z0_rel+z0))**0.0706 / (log((z_out) / (z0_rel+z0)))")
                acceleration_factor = ne.evaluate("where(acceleration_factor > 0, acceleration_factor, 1)")
                exp_Wind = ne.evaluate("wind_speed * acceleration_factor")

            else:
                acceleration_factor = np.log(z_out / z0) * (z0 / (z0_rel + z0)) ** 0.0706 / (
                    np.log((z_out) / (z0_rel + z0)))
                acceleration_factor = np.where(acceleration_factor > 0, acceleration_factor, 1)
                exp_Wind = wind_speed * acceleration_factor

            print(f"____Acceleration maximum expose {np.nanmax(acceleration_factor)}")
            return (exp_Wind, acceleration_factor)
        
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
        wind_speed = wind_speed * 0 + input_speed
        wind_dir = wind_dir * 0 + input_dir
        if verbose: print("__Wind speed scaled for ideal cases")
        return (wind_speed, wind_dir)
    
    def wind_speed_ratio(self, num=None, den=None):
        if self._numexpr:
            a1 = ne.evaluate("where(den > 0, num / den, 1)")
        else:
            a1 = np.where(den > 0, num / den, 1)
        return (a1)

    def _3D_wind_speed(self, U=None, V=None, W=None, out=None, verbose=True):
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
        if out is None:
            if self._numexpr:
                wind_speed = ne.evaluate("sqrt(U**2 + V**2 + W**2)")
            else:
                wind_speed = np.sqrt(U ** 2 + V ** 2 + W ** 2)
            if verbose: print("__UVW calculated")
            return (wind_speed)

        else:
            if self._numexpr:
                ne.evaluate("sqrt(U**2 + V**2 + W**2)", out=out)
            else:
                np.sqrt(U ** 2 + V ** 2 + W ** 2, out=out)
            if verbose: print("__UVW calculated")

    def _2D_wind_speed(self, U=None, V=None, out=None, verbose=True):
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
        if out is None:
            if self._numexpr:
                wind_speed = ne.evaluate("sqrt(U**2 + V**2)")
            else:
                wind_speed = np.sqrt(U ** 2 + V ** 2)
            if verbose: print("__UV calculated")
            return (wind_speed)

        else:
            if self._numexpr:
                ne.evaluate("sqrt(U**2 + V**2)", out=out)
            else:
                np.sqrt(U ** 2 + V ** 2, out=out)
            if verbose: print("__UV calculated")

    def compute_wind_speed(self, U=None, V=None, W=None,
                           computing='num', out=None,
                           xarray_data=None, u_name="U", v_name="V", verbose=True):
        """
        Calculates wind speed from wind speed components.

        First detects the number of wind component then calculates wind speed.
        The calculation can be performed on nmexpr, numpy or xarray dataset

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
        computing : str, optional
            Select the librairie to use for calculation. If 'num' first test numexpr, if not available select numpy.
            If 'xarray', a xarray dataframe needs to be specified with the corresponding names for wind components.
            (Default: 'num')

        Returns
        -------
        UV : ndarray
            Wind speed
        """
        # Numexpr or numpy
        if computing == 'num':
            if out is None:
                if W is None:
                    wind_speed = self._2D_wind_speed(U=U, V=V, verbose=verbose)
                else:
                    wind_speed = self._3D_wind_speed(U=U, V=V, W=W, verbose=verbose)
                wind_speed = change_dtype_if_required(wind_speed, np.float32)
                return (wind_speed)
            else:
                if W is None:
                    self._2D_wind_speed(U=U, V=V, out=out, verbose=verbose)
                else:
                    self._3D_wind_speed(U=U, V=V, W=W, out=out, verbose=verbose)

        if computing == 'xarray':
            xarray_data = xarray_data.assign(Wind=lambda x: np.sqrt(x[u_name] ** 2 + x[v_name] ** 2))
            xarray_data = xarray_data.assign(
                Wind_DIR=lambda x: np.mod(180 + np.rad2deg(np.arctan2(x[u_name], x[v_name])), 360))
            if verbose: print("__Wind and Wind_DIR calculated on xarray")
            return (xarray_data)

    def wind_speed_scaling(self, scaling_wind, prediction, linear=True, verbose=True):
        """
        scaling_wind * prediction / 3

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
        if linear:
            if self._numexpr:
                prediction = ne.evaluate("scaling_wind * prediction / 3")
            else:
                prediction = scaling_wind * prediction / 3
        prediction = change_dtype_if_required(prediction, np.float32)
        if verbose: print('__Wind speed scaling done')
        return (prediction)

    def angular_deviation(self, U, V, verbose=True):
        """
        Angular deviation from incoming flow.

        THe incoming flow is supposed from the West so that V=0. If deviated, V != 0.
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
        if self._numexpr:
            alpha = ne.evaluate("where(U == 0, where(V == 0, 0, V/abs(V) * 3.14159 / 2), arctan(V / U))")
        else:
            alpha = np.where(U == 0,
                             np.where(V == 0, 0, np.sign(V) * np.pi / 2),
                             np.arctan(V / U))
        alpha = change_dtype_if_required(alpha, np.float32)
        if verbose: print("__Angular deviation calculated")
        return (alpha)

    def direction_from_alpha(self, wind_dir, alpha, input_dir_in_degree=True, verbose=True):
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
        if input_dir_in_degree:
            if self._numexpr:
                UV_DIR = ne.evaluate("(3.14159/180) * wind_dir - alpha")
            else:
                UV_DIR = (np.pi / 180) * wind_dir - alpha
        UV_DIR = change_dtype_if_required(UV_DIR, np.float32)
        if verbose: print("__Wind_DIR calculated from alpha")
        return (UV_DIR)

    def horizontal_wind_component(self, UV=None, UV_DIR=None,
                                  working_with_xarray=False, xarray_data=None, wind_name="Wind",
                                  wind_dir_name="Wind_DIR", verbose=True):
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
        """
        if not (working_with_xarray):

            if self._numexpr:
                U = ne.evaluate("-sin(UV_DIR) * UV")
                V = ne.evaluate("-cos(UV_DIR) * UV")

            else:
                U = -np.sin(UV_DIR) * UV
                V = -np.cos(UV_DIR) * UV

            if verbose: print('__Horizontal wind component calculated')
            U = change_dtype_if_required(U, np.float32)
            V = change_dtype_if_required(V, np.float32)

            return (U, V)

        if working_with_xarray:
            xarray_data = xarray_data.assign(theta=lambda x: (np.pi / 180) * (x[wind_dir_name] % 360))
            xarray_data = xarray_data.assign(U=lambda x: -x[wind_name] * np.sin(x["theta"]))
            xarray_data = xarray_data.assign(V=lambda x: -x[wind_name] * np.cos(x["theta"]))
            if verbose: print('__Horizontal wind component calculated on xarray')
            return (xarray_data)

    def direction_from_u_and_v(self, U, V, output_in_degree=True, verbose=True):
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
        if output_in_degree:
            if self._numexpr:
                UV_DIR = ne.evaluate("(180 + (180/3.14159)*arctan2(U,V)) % 360")
            else:
                UV_DIR = np.mod(180 + np.rad2deg(np.arctan2(U, V)), 360)
            if verbose: print('__Wind_DIR calculated from U and V')
            UV_DIR = change_dtype_if_required(UV_DIR, np.float32)
            return (UV_DIR)