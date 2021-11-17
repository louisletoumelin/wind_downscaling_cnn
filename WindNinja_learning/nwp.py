import numpy as np
import xarray as xr


def preprocess_function(netCDF_file):
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

    return netCDF_file


def load_netcdf_and_preprocess(path, dtype=np.float32):
    dataset = xr.open_mfdataset(path,
                                preprocess=preprocess_function,
                                concat_dim='time').astype(dtype, copy=False)
    return dataset


def select_time_range_xr(dataset, begin, end):
    return dataset.sel(time=slice(begin, end))


def select_station_grid_point_in_NWP(nwp, x_idx_nwp, y_idx_nwp):
    return nwp.isel(xx=x_idx_nwp, yy=y_idx_nwp)