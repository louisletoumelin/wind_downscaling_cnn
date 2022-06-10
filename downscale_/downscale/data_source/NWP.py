import numpy as np
import pandas as pd
import xarray as xr

from downscale.data_source.data_2D import Data_2D
from downscale.utils.context_managers import print_all_context

try:
    import pyproj
    _pyproj = True
except ModuleNotFoundError:
    _pyproj = False

try:
    import dask
    _dask = True
except ModuleNotFoundError:
    _dask = False


class NWP(Data_2D):

    _pyproj = _pyproj
    _dask = _dask

    def __init__(self, path_to_file=None, begin=None, end=None,
                 variables_of_interest=['Wind', 'Wind_DIR', 'LAT', 'LON', 'ZS'], prm={}):

        path_to_file = path_to_file if path_to_file is not None else prm.get("selected_path", None)
        save_path = prm.get("save_path")
        path_Z0_2018 = prm.get("path_Z0_2018")
        path_Z0_2019 = prm.get("path_Z0_2019")
        path_to_coord_L93 = prm.get("path_to_coord_L93")
        load_z0 = prm.get("load_z0")
        save = prm.get("save_z0")
        name = prm.get("name_nwp")

        with print_all_context("NWP", level=0, unit="second", verbose=prm.get("verbose", None)):

            # inherit from data
            super().__init__(path_to_file, name)

            # List of variables
            self.variables_of_interest = variables_of_interest
            self.save_path = save_path
            self.begin = begin
            self.end = end

            # Path to file can be a string or a list of strings
            self.load_nwp_files(path_to_file=path_to_file,
                                preprocess_function=self._preprocess_ncfile,
                                prm=prm)

            # Select timeframe
            self.select_timeframe(begin=begin, end=end)

            # Select variables of interest
            self._select_specific_variables()

            # Compute
            self.compute_array_dask()

            # Add L93 coordinates
            self.add_l93_coordinates(path_to_coord_L93=path_to_coord_L93, prm=prm)

            # Modify variables of interest
            self.variables_of_interest = variables_of_interest + ['X_L93', 'Y_L93']
            self._select_specific_variables()

            # Add z0 variables
            self._add_Z0(path_Z0_2018, path_Z0_2019, save=save, load=load_z0, prm=prm, verbose=True)

            # float32
            self.data_xr = self.data_xr.astype("float32", copy=False)

    def compute_array_dask(self):
        try:
            self.data_xr = self.data_xr.compute()
        except:
            print('Did not use compute method on xarray')

    def load_nwp_files(self, path_to_file=None, preprocess_function=None, parallel=False,
                       prm=None, verbose=True):  # self._preprocess_ncfile
        if _dask:
            # Open netcdf file
            self.data_xr = xr.open_mfdataset(path_to_file,
                                             preprocess=preprocess_function,
                                             combine="nested",
                                             concat_dim='time',
                                             parallel=parallel).astype(np.float32, copy=False)
            print(f"__Function xr.open_mfdataset. Parallel: {parallel}") if verbose else None
        else:
            print("__Dask = False")
            self.data_xr = xr.open_dataset(path_to_file)
            print(f"__File loaded: {path_to_file}")
            self.data_xr = preprocess_function(self.data_xr)
            print("__Function xr.open_dataset") if verbose else None

    def add_l93_coordinates(self, path_to_coord_L93=None, verbose=True, prm={}):
        if _pyproj:
            self.data_xr = self.gps_to_l93(data_xr=self.data_xr,
                                           shape=self.shape,
                                           lon='LON',
                                           lat='LAT',
                                           height=self.height,
                                           length=self.length)
            print("__Projected lat/lon coordinates into l93 with pyproj") if verbose else None
        else:
            self._add_X_Y_L93(path_to_coord_L93)
            print("__Added l93 coordinates from npy. Did not project lat/lon") if verbose else None

    def _add_X_Y_L93(self, path_to_coord_L93):
        X_L93 = np.load(path_to_coord_L93 + '_X_L93.npy')
        Y_L93 = np.load(path_to_coord_L93 + '_Y_L93.npy')
        try:
            self.data_xr['X_L93'] = (('yy', 'xx'), X_L93[0:225, 0:175])
            self.data_xr['Y_L93'] = (('yy', 'xx'), Y_L93[0:225, 0:175])
        except ValueError:
            self.data_xr['X_L93'] = (('yy', 'xx'), X_L93)
            self.data_xr['Y_L93'] = (('yy', 'xx'), Y_L93)

    def _interpolate_Z0(self, path_Z0_2018, path_Z0_2019, verbose=True, save=False):

        # Open netcdf files
        z0_10_days_2018 = xr.open_mfdataset(path_Z0_2018, parallel=False)
        z0_10_days_2019 = xr.open_mfdataset(path_Z0_2019, parallel=False)

        # Create time indexes
        full_indexes = []
        for year in [2017, 2018, 2019, 2020]:
            start_time = f"{year}-01-01T00:00:00"
            end_time = f"{year}-12-31T23:00:00"
            full_indexes.append(pd.date_range(start=start_time, end=end_time, freq="1H"))

        # Select Z0 and Z0REL and downsample to 1H
        # Nearest neighbors interpolation for 2018
        Z0_1h_2018_nearest = z0_10_days_2018[['Z0', 'Z0REL']].interp(time=full_indexes[1], method="nearest",
                                                                     kwargs={"fill_value": "extrapolate"})
        if verbose: print(' .. interpolated nearest 2018')

        # Linear interpolation for 2018
        Z0_1h_2018_linear = z0_10_days_2018[['Z0', 'Z0REL']].interp(time=full_indexes[1], method="linear")
        if verbose: print(' .. interpolated linear 2018')

        # Nearest neighbors interpolation for 2019
        Z0_1h_2019_nearest = z0_10_days_2019[['Z0', 'Z0REL']].interp(time=full_indexes[2], method="nearest",
                                                                     kwargs={"fill_value": "extrapolate"})
        if verbose: print(' .. interpolated nearest 2019')

        # Linear interpolation for 2018
        Z0_1h_2019_linear = z0_10_days_2019[['Z0', 'Z0REL']].interp(time=full_indexes[2], method="linear")
        if verbose: print(' .. interpolated linear 2019')

        # Select 10 days begin and end
        begin_10_days_2018 = z0_10_days_2018.time.data[0]
        end_10_days_2018 = z0_10_days_2018.time.data[-1]
        short_time_2018 = pd.date_range(start=begin_10_days_2018, end=end_10_days_2018, freq="1H")

        begin_10_days_2019 = z0_10_days_2019.time.data[0]
        end_10_days_2019 = z0_10_days_2019.time.data[-1]
        short_time_2019 = pd.date_range(start=begin_10_days_2019, end=end_10_days_2019, freq="1H")

        # Linear interpolation between points and nearest extrapolation outside
        inside_2018 = Z0_1h_2018_linear.time.isin(short_time_2018)
        inside_2019 = Z0_1h_2019_linear.time.isin(short_time_2019)
        Z0_1h_2018 = Z0_1h_2018_linear.where(inside_2018, Z0_1h_2018_nearest)
        Z0_1h_2019 = Z0_1h_2019_linear.where(inside_2019, Z0_1h_2019_nearest)

        # 2017 is the mean of 2018 and 2019
        try:
            if verbose: print(' .. Start compute method on Z0 interpolation')
            Z0_var_1h_2017 = (Z0_1h_2019['Z0'].data.compute() + Z0_1h_2018['Z0'].data.compute()) / 2
            Z0REL_1h_2017 = (Z0_1h_2019['Z0REL'].data.compute() + Z0_1h_2018['Z0REL'].data.compute()) / 2
            if verbose: print(' .. End compute method on Z0 interpolation')
        except AttributeError:
            if verbose: print(' .. Did not use compute method in Z0 interpolation')
            Z0_var_1h_2017 = (Z0_1h_2019['Z0'].data + Z0_1h_2018['Z0'].data) / 2
            Z0REL_1h_2017 = (Z0_1h_2019['Z0REL'].data + Z0_1h_2018['Z0REL'].data) / 2

        # Create file for 2017
        Z0_1h_2017 = xr.Dataset(data_vars={"Z0": (["time", "yy", "xx"], Z0_var_1h_2017),
                                           "Z0REL": (["time", "yy", "xx"], Z0REL_1h_2017),
                                           },

                                coords={"time": full_indexes[0],
                                        "xx": np.array(list(range(176))),
                                        "yy": np.array(list(range(226)))})

        # Create a file for 2020 (including 29 February)
        shape_2020 = (len(full_indexes[3]), 226, 176)
        Z0_1h_2020 = xr.Dataset(data_vars={"Z0": (["time", "yy", "xx"], np.empty(shape_2020) * np.nan),
                                           "Z0REL": (["time", "yy", "xx"], np.empty(shape_2020) * np.nan),
                                           },

                                coords={"time": full_indexes[3],
                                        "xx": np.array(list(range(176))),
                                        "yy": np.array(list(range(226)))})

        index_2020 = pd.DataFrame(np.ones(full_indexes[3].shape), index=full_indexes[3])
        index_2020_small = index_2020[np.logical_not((index_2020.index.month == 2) & (index_2020.index.day == 29))]

        Z0_1h_2020_small = xr.Dataset(data_vars={"Z0": (["time", "yy", "xx"], Z0_1h_2017['Z0'].values),
                                                 "Z0REL": (["time", "yy", "xx"], Z0_1h_2017['Z0REL'].values),
                                                 },

                                      coords={"time": index_2020_small.index,
                                              "xx": np.array(list(range(176))),
                                              "yy": np.array(list(range(226)))})

        # Update empty dataset with values from 2017
        Z0_1h_2020.update(Z0_1h_2020_small)

        # Full array
        if verbose: print(' .. Concat netcdf Z0 files')
        array_Z0 = xr.concat([Z0_1h_2017, Z0_1h_2018, Z0_1h_2019, Z0_1h_2020], dim='time')

        if save: Z0_1h_2020.to_netcdf(self.save_path + 'processed_Z0.nc')
        return array_Z0

    def _add_Z0(self, path_Z0_2018, path_Z0_2019, save=False, load=False, prm={}, verbose=True):

        if (path_Z0_2018 is None) and (path_Z0_2019 is None):
            return

        if verbose: print('\n__Start adding Z0')

        if isinstance(self.begin, str):
            year, month, day = self.begin.split('-')
            month = month.lstrip("0")
            day = day.lstrip("0")

        elif isinstance(self.begin, pd._libs.tslibs.timestamps.Timestamp):
            begin = str(self.begin).split()[0]
            year, month, day = begin.split('-')
            month = month.lstrip("0")
            day = day.lstrip("0")

        elif isinstance(self.begin, np.datetime64):
            begin = str(self.begin).split('T')[0]
            year, month, day = begin.split('-')
            month = month.lstrip("0")
            day = day.lstrip("0")

        if load:
            chunks = {"time": 12} if prm["_dask"] else None
            array_Z0 = xr.open_dataset(self.save_path + f'processed_Z0_{year}_32bits.nc', chunks=chunks).astype(
                "float32", copy=False)
        else:
            # Interpolate
            array_Z0 = self._interpolate_Z0(path_Z0_2018, path_Z0_2019, verbose=verbose, save=save)

        # Drop additional pixels
        array_Z0 = array_Z0.where((array_Z0.xx < 175), drop=True)
        array_Z0 = array_Z0.where((array_Z0.yy < 225), drop=True)
        self.data_xr = self.data_xr.where((array_Z0.xx < 175), drop=True)
        self.data_xr = self.data_xr.where((array_Z0.yy < 225), drop=True)
        if verbose: print(' .. Discard additionnal pixels')

        # Save the file locally
        if save:
            array_Z0_2017 = array_Z0.sel(time=slice('2017-01-01', '2017-12-31'))
            array_Z0_2018 = array_Z0.sel(time=slice('2018-01-01', '2018-12-31'))
            array_Z0_2019 = array_Z0.sel(time=slice('2019-01-01', '2019-12-31'))
            try:
                array_Z0_2017.compute().to_netcdf(self.save_path + 'processed_Z0_2017.nc')
                array_Z0_2018.compute().to_netcdf(self.save_path + 'processed_Z0_2018.nc')
                array_Z0_2019.compute().to_netcdf(self.save_path + 'processed_Z0_2019.nc')
            except AttributeError:
                array_Z0_2017.to_netcdf(self.save_path + 'processed_Z0_2017.nc')
                array_Z0_2018.to_netcdf(self.save_path + 'processed_Z0_2018.nc')
                array_Z0_2019.to_netcdf(self.save_path + 'processed_Z0_2019.nc')

        # Select specific dates
        array_Z0 = array_Z0.sel(time=self.data_xr.time)
        if verbose: print(' .. Selected times for Z0')

        # Add Z0 to NWP
        self.data_xr['Z0'] = (('time', 'yy', 'xx'), array_Z0['Z0'].values)
        self.data_xr['Z0REL'] = (('time', 'yy', 'xx'), array_Z0['Z0REL'].values)

        if verbose: print('__End adding Z0')

    def _select_specific_variables(self):
        try:
            self.data_xr = self.data_xr[self.variables_of_interest]
        except:
            pass
        try:
            if 'LAT' in self.variables_of_interest:
                self.data_xr['LAT'] = self.data_xr['LAT'].isel(time=0)
            if 'LON' in self.variables_of_interest:
                self.data_xr['LON'] = self.data_xr['LON'].isel(time=0)
            if 'ZS' in self.variables_of_interest:
                self.data_xr['ZS'] = self.data_xr['ZS'].isel(time=0)
        except:
            pass

    def _preprocess_ncfile(self, netCDF_file):
        try:
            netCDF_file = netCDF_file.assign_coords(time=("time", netCDF_file.time.data))
        except:
            netCDF_file = netCDF_file.assign_coords(time=("oldtime", netCDF_file.time.data))

        netCDF_file = netCDF_file.assign_coords(xx=("xx", list(range(netCDF_file.dims['xx']))))
        netCDF_file = netCDF_file.assign_coords(yy=("yy", list(range(netCDF_file.dims['yy']))))

        try:
            netCDF_file = netCDF_file.rename({'oldtime': 'time'})
        except:
            pass

        netCDF_file = netCDF_file[self.variables_of_interest]
        return netCDF_file

    @property
    def shape(self):
        return self.data_xr['LON'].shape

    @staticmethod
    def gps_to_l93(data_xr=None, shape=None, lon='LON', lat='LAT', height=None, length=None):
        """Converts LAT/LON information from a NWP to L93"""

        import pyproj

        # Initialization²
        X_L93 = np.zeros(shape)
        Y_L93 = np.zeros(shape)

        # Load transformer
        gps_to_l93_func = pyproj.Transformer.from_crs(4326, 2154, always_xy=True)

        # Transform coordinates of each points
        for i in range(height):
            for j in range(length):
                projected_points = [point for point in
                                    gps_to_l93_func.itransform(
                                        [(data_xr[lon][i, j], data_xr[lat][i, j])])]
                X_L93[i, j], Y_L93[i, j] = projected_points[0]

        # Create a new variable with new coordinates
        data_xr["X_L93"] = (("yy", "xx"), X_L93)
        data_xr["Y_L93"] = (("yy", "xx"), Y_L93)
        return data_xr

    def convert_to_mnt_format(self, extract_wind=True):

        assert "X_L93" in self.data_xr
        assert "Y_L93" in self.data_xr

        self.data_xr = self.data_xr.assign_coords(x=("xx", self.data_xr.X_L93.data[0, :]))
        self.data_xr = self.data_xr.assign_coords(y=("yy", self.data_xr.Y_L93.data[:, 0]))
        self.data_xr = self.data_xr.drop(("xx", "yy"), dim=None)

        if extract_wind:
            self.data_xr = self.data_xr["Wind"]
            self.data_xr = self.data_xr.astype(np.float32)

        self.data_xr = self.data_xr.rename({"xx": "x", "yy": "y"})

    def select_timeframe(self, begin=None, end=None):

        if (begin is None) and (end is None):
            begin = self.begin
            end = self.end

        if isinstance(begin, np.datetime64) and isinstance(end, np.datetime64):
            print("____Used np.datetime64 to select timeframe")
            self.data_xr = self.data_xr.sel(time=slice(np.datetime64(begin), np.datetime64(end)))

        elif isinstance(begin, pd._libs.tslibs.timestamps.Timestamp) and isinstance(end, pd._libs.tslibs.timestamps.Timestamp):
            print("____Used Timestamp converted to str to select timeframe")
            self.data_xr = self.data_xr.sel(time=slice(str(begin), str(end)))

        elif isinstance(begin, str) and isinstance(end, str):
            print("____Used str to select timeframe")
            self.data_xr = self.data_xr.sel(time=slice(begin, end))

        print(f"____Selected timeframe for NWP, begin: {begin}")
        print(f"____Selected timeframe for NWP, end: {end}")
