import numpy as np


class Data_2D:

    def __init__(self, path_to_file=None, name=None):
        self.path_to_file = path_to_file
        self.name = name

    @staticmethod
    def x_y_to_stacked_xy(x_array, y_array):
        stacked_xy = np.dstack((x_array, y_array))
        return stacked_xy

    @staticmethod
    def grid_to_flat(stacked_xy):
        x_y_flat = [tuple(i) for line in stacked_xy for i in line]
        return x_y_flat

    @property
    def length(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

    @staticmethod
    def project_coordinates(lon=None, lat=None, crs_in=4326, crs_out=2154):
        import pyproj
        gps_to_l93_func = pyproj.Transformer.from_crs(crs_in, crs_out, always_xy=True)
        projected_points = [point for point in gps_to_l93_func.itransform([(lon, lat)])][0]
        return projected_points

    def crop_mnt(self, x_min, y_max, x_max, y_min, unit="degree",
                 input_topo='C:/Users/louis/git/wind_downscaling_CNN/Data/1_Raw/MNT/IGN_25m/ign_L93_25m_alpesIG.tif',
                 output_topo='C:/Users/louis/git/wind_downscaling_CNN/Data/1_Raw/MNT/IGN_25m/topo_lac_blanc_2.tif',
                 crs_in=4326,
                 crs_out=2154):
        """
        x_min, y_max, x_max, y_min = BDclim.select_bounding_box_around_station(station_name="AGUIL. DU MIDI", dx=2_000, dy=2_000)
        IGN.crop_mnt(x_min, y_max, x_max, y_min, unit="m")
        """
        from osgeo import gdal
        if unit == "degree":
            x_min, y_max = self.project_coordinates(lon=x_min, lat=y_max, crs_in=crs_in, crs_out=crs_out)
            x_max, y_min = self.project_coordinates(lon=x_max, lat=y_min, crs_in=crs_in, crs_out=crs_out)
        bbox = (x_min, y_max, x_max, y_min)
        ds = gdal.Open(input_topo)
        gdal.Translate(output_topo, ds, projWin=bbox)
