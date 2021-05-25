import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import rotate
import concurrent.futures

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

class Rotation:

    _numexpr = _numexpr
    _numba = _numba
    _dask = _dask

    def __init__(self):
        pass

    @staticmethod
    def rotate_scipy(topography, wind_dir):
        return (rotate(topography, wind_dir, reshape=False, mode='constant', cval=np.nan))

    def rotate_vectorize_scipy(self, topography, wind_dir):
        return (np.vectorize(self.rotate_scipy, signature='(m,n),()->(m,n)')(topography, wind_dir))

    def rotate_topography_scipy(self, topography, wind_dir, clockwise=False, verbose=True):
        """Rotate a topography to a specified angle

        If wind_dir = 270° then angle = 270+90 % 360 = 360 % 360 = 0
        For wind coming from the West, there is no rotation
        """
        if verbose: print('__Begin rotate topographies')

        if not (clockwise):
            rotated_topography = self.rotate_vectorize_scipy(topography, 90 + wind_dir)
        if clockwise:
            rotated_topography = self.rotate_vectorize_scipy(topography, -90 - wind_dir)

        if verbose: print('__End rotate topographies')
        return (rotated_topography)

    def select_rotation(self, data=None, wind_dir=None, type_rotation='scipy', clockwise=False, librairie='numba',
                        verbose=True, all_mat=None, topo_rot=None, topo_i=None, angles=None,
                        wind_large=None, wind=None):

        if type_rotation=='scipy':
            return(self.rotate_topography_scipy(data, wind_dir, clockwise=clockwise, verbose=verbose))

        if type_rotation=='topo_indexes':
            return(self.rotate_topo_indexes(librairie=librairie, all_mat=all_mat, topo_rot=topo_rot, topo_i=topo_i, angles=angles))

        if type_rotation=='wind_indexes':
            return(self.rotate_wind_indexes(librairie=librairie, all_mat=all_mat, wind_large=wind_large, wind=wind, angles=angles))


    def rotate_wind_indexes(self, librairie='numba', all_mat=None, wind_large=None, wind=None, angles=None):

        if librairie == 'numba' and _numba:
            return(self.rotate_wind_indexes_numba(all_mat, wind_large, wind, angles))

        if librairie == 'num':
            return(self.rotate_wind_indexes_numpy(all_mat, wind_large, wind, angles))

    def rotate_topo_indexes(self, librairie='numba', all_mat=None, topo_rot=None, topo_i=None, angles=None):

        if librairie=='numba' and _numba:
            # (360, 5451, 2) (1, 21, 21, 5451) (21, 21, 140, 140) (1, 21, 21)
            return(self.rotate_topo_indexes_numba(all_mat, topo_rot, topo_i, angles))

        if librairie =='num':
            return(self.rotate_topo_indexes_numpy(all_mat, topo_rot, topo_i, angles))

    if _numba:
        @staticmethod
        @jit([float32[:, :, :, :, :, :](int32[:, :, :], float32[:, :, :, :, :, :], float32[:, :, :, :, :], int32[:, :, :])],nopython=True)
        def rotate_wind_indexes_numba(all_mat, wind_large, wind, angles):
            for time in range(wind_large.shape[0]):
                for y in range(wind_large.shape[1]):
                    for x in range(wind_large.shape[2]):
                        angle_i = angles[time, y, x]
                        for number, (index_y, index_x) in enumerate(zip(all_mat[angle_i, :, 0], all_mat[angle_i, :, 1])):
                            wind_large[time, y, x, index_y, index_x, :] = wind[time, y, x, number, :]
            return (wind_large)

    @staticmethod
    def rotate_wind_indexes_numpy(all_mat, wind_large, wind, angles):
        for time in range(wind_large.shape[0]):
            for y in range(wind_large.shape[1]):
                for x in range(wind_large.shape[2]):
                    angle_i = angles[time, y, x]
                    for number, (index_y, index_x) in enumerate(zip(all_mat[angle_i, :, 0], all_mat[angle_i, :, 1])):
                        wind_large[time, y, x, index_y, index_x, :] = wind[time, y, x, number, :]
        return (wind_large)

    if _numba:
        @staticmethod
        @jit([float32[:, :, :, :](int32[:, :, :], float32[:, :, :, :], float32[:, :, :, :], int32[:, :, :])],nopython=True)
        def rotate_topo_indexes_numba(all_mat, topo_rot, topo_i, angles):
            for time in range(topo_rot.shape[0]):
                for y in range(topo_i.shape[0]):
                    for x in range(topo_i.shape[1]):
                        angle = angles[time, y, x]
                        for number in range(79 * 69):
                            topo_rot[time, y, x, number] = topo_i[
                                y, x, all_mat[angle, number, 0], all_mat[angle, number, 1]]
            return (topo_rot)

    @staticmethod
    def rotate_topo_indexes_numpy(all_mat, topo_rot, topo_i, angles):
        for time in range(topo_rot.shape[0]):
            for y in range(topo_i.shape[0]):
                for x in range(topo_i.shape[1]):
                    angle = angles[time, y, x]
                    for number in range(79 * 69):
                        topo_rot[time, y, x, number] = topo_i[
                            y, x, all_mat[angle, number, 0], all_mat[angle, number, 1]]
        return (topo_rot)




    def _rotate_topo_for_all_station(self):
        """Not used
        Rotate the topography at all stations for each 1 degree angle of wind direction"""

        def rotate_topo_for_all_degrees(self, station):
            dict_topo[station]["rotated_topo_HD"] = {}
            MNT_data, _, _ = observation.extract_MNT_around_station(self, station, mnt, 400, 400)
            for angle in range(360):
                tile = self.rotate_topography(MNT_data, angle)
                dict_topo[station]["rotated_topo_HD"][str(angle)] = []
                dict_topo[station]["rotated_topo_HD"][str(angle)].append(tile)
            return (dict_topo)

        dict_topo = {}
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(rotate_topo_for_all_degrees, self.observation.stations['name'].values)
        except:
            print(
                "Parallel computation using concurrent.futures didn't work, so rotate_topo_for_all_degrees will not be parallelized.")
            map(rotate_topo_for_all_degrees, self.observation.stations['name'].values)
        self.dict_rot_topo = dict_topo