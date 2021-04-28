import numpy as np
import pandas as pd
import xarray as xr
from time import time as t

try:
    import pyproj
    _pyproj = True
except:
    _pyproj = False
try:
    import dask
    _dask = True
except:
    _dask = False
try:
    from shapely.geometry import Point
    _shapely_geometry = True
except:
    _shapely_geometry = False

try:
    import geopandas as gpd
    _geopandas = True
except:
    _geopandas = False


from Data_2D import Data_2D


class NWP(Data_2D):

    _dask = _dask
    _pyproj = _pyproj
    _shapely_geometry = _shapely_geometry
    _geopandas = _geopandas

    def __init__(self, path_to_file, name=None, begin=None, end=None, save_path=None, path_Z0_2018=None, path_Z0_2019=None,
                 variables_of_interest=['Wind', 'Wind_DIR', 'LAT', 'LON', 'ZS'], verbose=True, path_to_file_npy=None,
                 save=False, load_z0=False):
        if verbose:
            t0 = t()
        super().__init__(path_to_file, name)

        # List of variables
        self.variables_of_interest = variables_of_interest

        # save_path
        self.save_path = save_path
        self.begin = begin
        self.begin = begin
        self.end = end

        # Path to file can be a string or a list of strings
        if _dask:
            # Open netcdf file
            self.data_xr = xr.open_mfdataset(path_to_file, preprocess=self._preprocess_ncfile, concat_dim='time',
                                             parallel=False).astype(np.float32, copy=False)
        else:
            print("\nNot using dask to open netcdf files\n")
            self.data_xr = xr.open_dataset(path_to_file)
            self.data_xr = self._preprocess_ncfile(self.data_xr)

        # Select timeframe
        self.select_timeframe()

        # Select variables of interest
        self._select_specific_variables()

        # Add L93 coordinates
        if _pyproj:
            self.gps_to_l93()
        else:
            self._add_X_Y_L93(path_to_file_npy)

        # Modify variables of interest
        self.variables_of_interest = variables_of_interest + ['X_L93', 'Y_L93']
        self._select_specific_variables()

        # Add z0 variables
        if (path_Z0_2018 is not None) and (path_Z0_2019 is not None):
            self._add_Z0(path_Z0_2018, path_Z0_2019, save=save, load=load_z0, verbose=True)

        if verbose:
            t1 = t()
            print(f"\nNWP created in {np.round(t1-t0, 2)} seconds\n")

    def _add_X_Y_L93(self, path_to_file_npy):
        X_L93 = np.load(path_to_file_npy + "_X_L93.npy")
        Y_L93 = np.load(path_to_file_npy + "_Y_L93.npy")
        data_xr = self.data_xr
        data_xr['X_L93'] = (('yy', 'xx'), X_L93)
        data_xr['Y_L93'] = (('yy', 'xx'), Y_L93)
        self.data_xr = data_xr

    def _interpolate_Z0(self, path_Z0_2018, path_Z0_2019, verbose=True, save=False):
        # Open netcdf files
        z0_10_days_2018 = xr.open_mfdataset(path_Z0_2018, parallel=False)
        z0_10_days_2019 = xr.open_mfdataset(path_Z0_2019, parallel=False)

        # Create time indexes
        full_indexes = []
        for year in [2017, 2018, 2019]:
            start_time = f"{year}-01-01T00:00:00"
            end_time = f"{year}-12-31T23:00:00"
            full_indexes.append(pd.date_range(start=start_time, end=end_time, freq="1H"))

        # Select Z0 and Z0REL and downsample to 1H
        Z0_1h_2018_nearest = z0_10_days_2018[['Z0', 'Z0REL']].interp(time=full_indexes[1], method="nearest",
                                                                  kwargs={"fill_value": "extrapolate"})
        if verbose: print(' .. interpolated nearest 2018')
        Z0_1h_2018_linear = z0_10_days_2018[['Z0', 'Z0REL']].interp(time=full_indexes[1], method="linear")
        if verbose: print(' .. interpolated linear 2018')
        Z0_1h_2019_nearest = z0_10_days_2019[['Z0', 'Z0REL']].interp(time=full_indexes[2], method="nearest",
                                                                  kwargs={"fill_value": "extrapolate"})
        if verbose: print(' .. interpolated nearest 2019')
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
        # Full array
        if verbose: print(' .. Concat netcdf Z0 files')
        array_Z0 = xr.concat([Z0_1h_2017, Z0_1h_2018, Z0_1h_2019], dim='time')

        if save: array_Z0.to_netcdf(self.save_path + 'processed_Z0.nc')

        return(array_Z0)

    def _add_Z0(self, path_Z0_2018, path_Z0_2019, save=False, load=False, verbose=True):
        if verbose: print('\nStart adding Z0')
        year, month, day = self.begin.split('-')
        if load:
            array_Z0 = xr.open_dataset(self.save_path + f'processed_Z0_{year}.nc', chunks={"time": 12})
        else:
            # Interpolate
            array_Z0 = self._interpolate_Z0(path_Z0_2018, path_Z0_2019, verbose=verbose, save=save)

        # Drop additional pixels
        array_Z0 = array_Z0.where((array_Z0.xx < 175), drop=True)
        array_Z0 = array_Z0.where((array_Z0.yy < 225), drop=True)

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

        if verbose: print('\nEnd adding Z0')

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
        try:
            self.data_xr = self.data_xr.compute()
        except:
            print('Did not use compute method on xarray')

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
        return (netCDF_file)

    @property
    def shape(self):
        return (self.data_xr['LON'].shape)

    def gps_to_l93(self):
        """Converts LAT/LON information from a NWP to L93"""

        # Initialization
        X_L93 = np.zeros(self.shape)
        Y_L93 = np.zeros(self.shape)

        # Load transformer
        gps_to_l93_func = pyproj.Transformer.from_crs(4326, 2154, always_xy=True)

        # Transform coordinates of each points
        for i in range(self.height):
            for j in range(self.length):
                projected_points = [point for point in
                                    gps_to_l93_func.itransform(
                                        [(self.data_xr['LON'][i, j], self.data_xr['LAT'][i, j])])]
                X_L93[i, j], Y_L93[i, j] = projected_points[0]

        # Create a new variable with new coordinates
        self.data_xr["X_L93"] = (("yy", "xx"), X_L93)
        self.data_xr["Y_L93"] = (("yy", "xx"), Y_L93)

    def select_timeframe(self, begin=None, end=None):
        if (begin is not None) and (end is not None):
            self.data_xr = self.data_xr.sel(time=slice(begin, end))
        else:
            self.data_xr = self.data_xr.sel(time=slice(self.begin, self.end))

