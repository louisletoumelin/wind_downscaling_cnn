import numpy as np
import pyproj
from scipy.spatial import cKDTree
from osgeo import gdal
import concurrent.futures


def project_coordinates(lon=None, lat=None, crs_in=4326, crs_out=2154):

    gps_to_l93_func = pyproj.Transformer.from_crs(crs_in, crs_out, always_xy=True)
    projected_points = [point for point in gps_to_l93_func.itransform([(lon, lat)])][0]
    return projected_points


def crop_and_save_dem(x_min, y_max, x_max, y_min,
                      unit="m",
                      name="lac_blanc_test",
                      input_topo='C:/path/to/file/COP30_L93_cropped.tif',
                      output_dir='C:/path/to/folder/',
                      crs_in=4326,
                      crs_out=2154):

    if unit == "degree":
        x_min, y_max = project_coordinates(lon=x_min, lat=y_max, crs_in=crs_in, crs_out=crs_out)
        x_max, y_min = project_coordinates(lon=x_max, lat=y_min, crs_in=crs_in, crs_out=crs_out)

    bbox = (x_min, y_max, x_max, y_min)
    ds = gdal.Open(input_topo)
    gdal.Translate(output_dir+name+".tif", ds, projWin=bbox)


def x_y_to_stacked_xy(x_array, y_array):
    stacked_xy = np.dstack((x_array, y_array))
    return stacked_xy


def grid_to_flat(stacked_xy):
    x_y_flat = [tuple(i) for line in stacked_xy for i in line]
    return x_y_flat


def find_nearest_neighbor_in_grid(x_grid, y_grid, list_coord_station, number_of_neighbors=1):

    def K_N_N_point(point):
        distance, idx = tree.query(point, k=number_of_neighbors)
        return distance, idx

    # Coordinates where to find neighbors
    if x_grid.ndim == 1 and y_grid.ndim == 1:
        x_grid, y_grid = np.meshgrid(x_grid, y_grid)

    stacked_xy = x_y_to_stacked_xy(x_grid, y_grid)
    grid_flat = grid_to_flat(stacked_xy)
    tree = cKDTree(grid_flat)

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list_nearest = executor.map(K_N_N_point, list_coord_station)
    except ModuleNotFoundError:
        list_nearest = map(K_N_N_point, list_coord_station)

    list_nearest = np.array([np.array(station) for station in list_nearest])
    list_index = [(x, y) for x in range(len(x_grid)) for y in range(len(y_grid))]

    index_nearest_neighbor = list_index[np.intp(list_nearest[0, 1])]

    return index_nearest_neighbor


"""
crop_and_save_dem(dem.x.values[120*i], dem.y.values[120*j], dem.x.values[120*(i+1)+1], dem.y.values[120*(j+1)+1], name="lac_blanc_test_5", unit="m")

destination = 'C:/Users/louis/git/wind_downscaling_CNN/Data/1_Raw/MNT/IGN_25m/Test_wind_ninja/windninja_simu_nc.nc'
source = 'C:/Users/louis/git/wind_downscaling_CNN/Data/1_Raw/MNT/IGN_25m/Test_wind_ninja/topo_lac_blanc_50_10_33m_vel.asc'
gdal.Translate(destination, source)

path_to_WindNinja = "C:/WindNinja/WindNinja-3.7.2/bin"

#WindNinja cli C:/XXXX/cli wxModelInitialization diurnal.cfg --elevation file
#C:/XXXX/canyon fire.asc --vegetation grass --num threads 4

t0 = time()
# mesh_choice		 = medium
# --existing_case_directory C:/Users/louis/git/wind_downscaling_CNN/Data/1_Raw/MNT/Copernicus/NINJAFOAM_lac_blanc_test_5_13012_3/
exp = "C:/WindNinja/WindNinja-3.7.2/etc/windninja/example-files/cli_momentumSolver_diurnal.cfg  --input_speed 9 --input_direction 192 --uni_air_temp 10 --uni_cloud_cover 30  --write_wx_station_csv true"
os.system(path_to_WindNinja+"WindNinja_cli "+exp)
t1 = time()
"""
