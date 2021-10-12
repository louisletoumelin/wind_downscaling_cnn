from netCDF4 import date2num
from netCDF4 import stringtochar

AROME2 = xr.open_dataset(prm["data_path"] + "AROME/FORCING_alp_2019060107_2019070106.nc")

# Rename dimension time
if "time" in AROME2.dims:
    AROME2 = AROME2.rename({"time": "Time"})
if "oldtime" in AROME2.dims:
    AROME2 = AROME2.rename({"oldtime": "Time"})

datetime = AROME2.Time.to_pandas()
datetime = datetime.dt.strftime("%Y-%m-%d_%H:%M:%S")
datetime = datetime.values
AROME2["Times"] = ("Time", datetime.astype(np.dtype(('S', 19))))

# Rename variables and dimensions
AROME2 = AROME2.rename({'yy': 'south_north', 'xx': 'west_east'})

# Attributes
#for attribute in test.attrs:
#    AROME2.attrs[attribute] = test.attrs[attribute]
AROME2.attrs["TITLE"] = " WRF wrf test"
AROME2.attrs["MAP_PROJ"] = 1
AROME2.attrs["SIMULATION_START_DATE"] = "2019-06-01_07:00"
AROME2.attrs["CEN_LAT"] = 45.127639
AROME2.attrs["CEN_LON"] = 6.111555
AROME2.attrs["DX"] = 1300
AROME2.attrs["DY"] = 1300
AROME2.attrs['MOAD_CEN_LAT'] = AROME2.Projection_parameters.latitude_of_projection_origin
AROME2.attrs['STAND_LON'] = AROME2.Projection_parameters.longitude_of_central_meridian
AROME2.attrs['TRUELAT1'] = AROME2.Projection_parameters.standard_parallel
AROME2.attrs['TRUELAT2'] = AROME2.Projection_parameters.standard_parallel
AROME2.attrs['BOTTOM-TOP_GRID_DIMENSION'] = 1  # 1 (surface) z-layer

# Modify Temperature
if "t2" in AROME2.data_vars:
    AROME2["t2"] = AROME2["t2"] + 273.15
    AROME2 = AROME2.rename({"t2": "T2"})

# Create cloud cover variable
if "QCLOUD" not in AROME2.data_vars:
    shape_0, shape_2, shape_3 = AROME2["T2"].values.shape
    AROME2["QCLOUD"] = (("Time", "bottom_top", "south_north", "west_east"), np.zeros_like(AROME2["T2"].values).reshape(shape_0, 1, shape_2, shape_3))

# Create XLAT and XLON
AROME2["XLAT"] = (("Time", "south_north", "west_east"), (np.array([AROME2["LAT"].values for k in range(len(AROME2["Times"].values))])))
AROME2["XLONG"] = (("Time", "south_north", "west_east"), (np.array([AROME2["LON"].values for k in range(len(AROME2["Times"].values))])))

from downscale.Operators.Processing import Wind_utils

wu = Wind_utils()

# Compute zonal and meridional wind components
if "U10" not in AROME2.data_vars or "V10" not in AROME2.data_vars:
    AROME2 = wu.horizontal_wind_component(working_with_xarray=True, xarray_data=AROME2)
    AROME2 = AROME2.rename({"U": "U10", "V": "V10"})
    # AROME2 = AROME2.expand_dims("DateStrLen")

# Keep only the variable your are intersted in
AROME2 = AROME2[["U10", "V10", "T2", "XLAT", "XLONG", "QCLOUD", "Times"]]
AROME2 = AROME2.set_coords("XLAT")
AROME2 = AROME2.set_coords("XLONG")
#AROME2 = AROME2.rename_vars({"Time":"XTIME"})
#units = 'minutes since 2000-01-01 00:00:00'
#dates = [pd.to_datetime(date).to_pydatetime() for date in AROME2["XTIME"].values]
#AROME2["XTIME"] = (("Time"), date2num(dates,units))
#AROME2["XTIME"].attrs["units"] = units

# Select the domain
AROME3 = AROME2.isel(Time=slice(0, 2)).isel(west_east=slice(109 - 2, 109 + 2), south_north=slice(62 - 2, 62 + 2))
AROME3["T2"] = AROME3["T2"].astype(np.float32)
AROME3["U10"] = AROME3["U10"].astype(np.float32)
AROME3["V10"] = AROME3["V10"].astype(np.float32)
AROME3["XLAT"] = AROME3["XLAT"].astype(np.float32)
AROME3["XLONG"] = AROME3["XLONG"].astype(np.float32)
AROME3["QCLOUD"] = AROME3["QCLOUD"].astype(np.float32)
"""
AROME3.to_netcdf('wrfout_d18_2019-06-01_07_00_00.nc',
                 format='NETCDF3_CLASSIC',
                 encoding={
                     'Times': {
                         "char_dim_name": 'DateStrLen'
                     }
                 },
                 unlimited_dims={'Time': True})
"""
ncout = Dataset('myfile9.nc','w','NETCDF3')
ncout.createDimension('Time',AROME3.dims["Time"])
ncout.createDimension('DateStrLen',19)
ncout.createDimension('south_north',AROME3.dims["south_north"])
ncout.createDimension('west_east',AROME3.dims["west_east"])
ncout.createDimension('bottom_top',AROME3.dims["bottom_top"])

XLONG = ncout.createVariable('XLONG','float32',('Time', 'south_north', 'west_east'))
XLONG[:] = AROME3["XLONG"].values

XLAT = ncout.createVariable('XLAT','float32',('Time', 'south_north', 'west_east'))
XLAT[:] = AROME3["XLAT"].values

U10 = ncout.createVariable('U10','float32',('Time', 'south_north', 'west_east'))
U10[:] = AROME3["U10"].values

V10 = ncout.createVariable('V10','float32',('Time', 'south_north', 'west_east'))
V10[:] = AROME3["U10"].values

T2 = ncout.createVariable('T2','float32',('Time', 'south_north', 'west_east'))
T2[:] = AROME3["T2"].values

Times = ncout.createVariable('Times','S1',('Time', 'DateStrLen'))
Times[:] = stringtochar(AROME3["Times"].values)

ncout.close()