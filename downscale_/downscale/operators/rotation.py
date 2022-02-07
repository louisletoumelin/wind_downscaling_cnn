import numpy as np
from scipy.ndimage import rotate
import concurrent.futures

from downscale.utils.decorators import print_func_executed_decorator, timer_decorator

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
    import tensorflow_addons as tfa
    _tfa = True
except ModuleNotFoundError:
    _tfa = False

class Rotation:

    def __init__(self):
        pass

    @staticmethod
    def rotate_scipy(topography, wind_dir):
        return rotate(topography, wind_dir, reshape=False, mode='constant', cval=np.nan)

    def rotate_vectorize_scipy(self, topography, wind_dir):
        return np.vectorize(self.rotate_scipy, signature='(m,n),()->(m,n)')(topography, wind_dir)

    def rotate_topography_scipy(self, topography, wind_dir, clockwise=False, verbose=True):
        """Rotate a topography to a specified angle

        If wind_dir = 270° then angle = 270+90 % 360 = 360 % 360 = 0
        For wind coming from the West, there is no rotation
        """

        if not clockwise:
            rotated_topography = self.rotate_vectorize_scipy(topography, 90 + wind_dir)
        if clockwise:
            rotated_topography = self.rotate_vectorize_scipy(topography, -90 - wind_dir)

        if verbose: print('____Library: scipy')
        return rotated_topography

    def rotate_vectorize_tfa(self, topography, wind_dir):
        """
        topography.shape == (1, nb_px_nwp_y * nb_px_nwp_x, length_y, length_x, 1)
        """
        signature = '(nb_images,length_y,length_x,channel),(nb_images)->(nb_images,length_y,length_x,channel)'
        return np.vectorize(self._rotate_tfa, signature=signature)(topography, wind_dir)

    @staticmethod
    def _rotate_tfa(data, wind_dir):
        return tfa.image.rotate(data, wind_dir, interpolation="nearest", fill_mode="constant", fill_value=np.nan)

    def rotate_tfa_on_GPU(self, data, wind_dir, verbose=True):
        rotated_images = self.rotate_vectorize_tfa(data, wind_dir)
        print("____rotated on GPU") if verbose else None
        return rotated_images

    def rotate_tfa_on_CPU(self, data, wind_dir, verbose=True):
        rotated_images = self.rotate_vectorize_tfa(data, wind_dir)
        print("____rotated on CPU") if verbose else None
        return rotated_images

    def rotate_tfa(self, data, wind_dir, clockwise=False, GPU=False, verbose=True):
        """Rotate a topography to a specified angle

        If wind_dir = 270° then angle = 270+90 % 360 = 360 % 360 = 0
        For wind coming from the West, there is no rotation
        """

        wind_dir_corrected = 90 + wind_dir if not clockwise else -90 - wind_dir
        if GPU:
            rotated_topography = self.rotate_tfa_on_GPU(data,
                                             wind_dir_corrected)
        else:
            rotated_topography = self.rotate_tfa_on_CPU(data,
                                             wind_dir_corrected)

        if verbose: print('____Library: tfa')
        return rotated_topography

    @print_func_executed_decorator("rotating", level_begin="\n__", level_end="__")
    @timer_decorator("rotating", unit="minute", level=". . . . ")
    def select_rotation(self, data=None, wind_dir=None, type_rotation='scipy', clockwise=False, library='numba',
                        verbose=True, all_mat=None, topo_rot=None, topo_i=None, angles=None,
                        wind_large=None, wind=None, GPU=False, fill_value=np.nan, interp_rotate_tfa="nearest"):

        if type_rotation == 'scipy':
            return self.rotate_topography_scipy(data, wind_dir, clockwise=clockwise, verbose=verbose)

        if type_rotation == "tfa" and _tfa:
            return self.rotate_tfa(data, wind_dir, clockwise=clockwise, GPU=GPU, verbose=verbose)

        if type_rotation == 'topo_indexes':
            return self.rotate_topo_indexes(library=library, all_mat=all_mat, topo_rot=topo_rot, topo_i=topo_i,
                                            angles=angles)

        if type_rotation == 'wind_indexes':
            return self.rotate_wind_indexes(library=library, all_mat=all_mat, wind_large=wind_large, wind=wind,
                                            angles=angles)

    def rotate_wind_indexes(self, library='numba', all_mat=None, wind_large=None, wind=None, angles=None):

        if library == 'numba' and _numba:
            return self.rotate_wind_indexes_numba(all_mat, wind_large, wind, angles)
        else:
            return self.rotate_wind_indexes_numpy(all_mat, wind_large, wind, angles)

    def rotate_topo_indexes(self, library='numba', all_mat=None, topo_rot=None, topo_i=None, angles=None):

        if library == 'numba' and _numba:
            # (360, 5451, 2) (1, 21, 21, 5451) (21, 21, 140, 140) (1, 21, 21)
            return self.rotate_topo_indexes_numba(all_mat, topo_rot, topo_i, angles)
        else:
            return self.rotate_topo_indexes_numpy(all_mat, topo_rot, topo_i, angles)

    if _numba:
        @staticmethod
        @jit([float32[:, :, :, :, :, :](int32[:, :, :], float32[:, :, :, :, :, :], float32[:, :, :, :, :],
                                        int32[:, :, :])], nopython=True)
        def rotate_wind_indexes_numba(all_mat, wind_large, wind, angles):
            for time in range(wind_large.shape[0]):
                for y in range(wind_large.shape[1]):
                    for x in range(wind_large.shape[2]):
                        angle_i = angles[time, y, x]
                        for number, (index_y, index_x) in enumerate(
                                zip(all_mat[angle_i, :, 0], all_mat[angle_i, :, 1])):
                            wind_large[time, y, x, index_y, index_x, :] = wind[time, y, x, number, :]
            return wind_large

    @staticmethod
    def rotate_wind_indexes_numpy(all_mat, wind_large, wind, angles):
        for time in range(wind_large.shape[0]):
            for y in range(wind_large.shape[1]):
                for x in range(wind_large.shape[2]):
                    angle_i = angles[time, y, x]
                    for number, (index_y, index_x) in enumerate(zip(all_mat[angle_i, :, 0], all_mat[angle_i, :, 1])):
                        wind_large[time, y, x, index_y, index_x, :] = wind[time, y, x, number, :]
        return wind_large

    if _numba:
        @staticmethod
        @jit([float32[:, :, :, :](int32[:, :, :], float32[:, :, :, :], float32[:, :, :, :], int32[:, :, :])],
             nopython=True)
        def rotate_topo_indexes_numba(all_mat, topo_rot, topo_i, angles):
            for time in range(topo_rot.shape[0]):
                for y in range(topo_i.shape[0]):
                    for x in range(topo_i.shape[1]):
                        angle = angles[time, y, x]
                        for number in range(79 * 69):
                            topo_rot[time, y, x, number] = topo_i[
                                y, x, all_mat[angle, number, 0], all_mat[angle, number, 1]]
            return topo_rot

    @staticmethod
    def rotate_topo_indexes_numpy(all_mat, topo_rot, topo_i, angles):
        for time in range(topo_rot.shape[0]):
            for y in range(topo_i.shape[0]):
                for x in range(topo_i.shape[1]):
                    angle = angles[time, y, x]
                    for number in range(79 * 69):
                        topo_rot[time, y, x, number] = topo_i[
                            y, x, all_mat[angle, number, 0], all_mat[angle, number, 1]]
        return topo_rot

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
            return dict_topo

        dict_topo = {}
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(rotate_topo_for_all_degrees, self.observation.stations['name'].values)
        except:
            print(
                "Parallel computation using concurrent.futures didn't work, so rotate_topo_for_all_degrees will not be parallelized.")
            map(rotate_topo_for_all_degrees, self.observation.stations['name'].values)
        self.dict_rot_topo = dict_topo

