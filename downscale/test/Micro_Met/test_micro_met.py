import numpy as np
from numpy.testing import *
import os
import warnings
import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)

from downscale.Operators.Micro_Met import MicroMet


@pytest.fixture
def mnt():
    directory = os.getcwd()
    try:
        mnt = np.load("../Data_test/mnt_small.npy")
    except FileNotFoundError:
        mnt = np.load(directory + "/Data_test/mnt_small.npy")
    return mnt


@pytest.fixture
def idx_x():
    idx_x = np.array([10, 30, 72])
    return np.array(idx_x)


@pytest.fixture
def idx_y():
    idx_y = [28, 29, 53]
    return idx_y


@pytest.fixture
def m():
    micromet_instance = MicroMet()
    return micromet_instance


@pytest.fixture
def result_terrain_slope_map(mnt, m):
    result = m.terrain_slope_map(mnt, 25, verbose=False)
    return result


@pytest.fixture
def result_terrain_slope_idx(mnt, m, idx_x, idx_y):
    result = m.terrain_slope_idx(mnt, 25, idx_x, idx_y, verbose=False)
    return result


@pytest.fixture
def result_terrain_slope_azimuth_map(mnt, m):
    result = m.terrain_slope_azimuth_map(mnt, 25, verbose=False)
    return result


@pytest.fixture
def result_terrain_slope_azimuth_idx(mnt, m, idx_x, idx_y):
    result = m.terrain_slope_azimuth_idx(mnt, 25, idx_x, idx_y, verbose=False)
    return result


@pytest.fixture
def result_curvature_map_scale_true(mnt, m):
    result = m.curvature_map(mnt, scale=True, verbose=False)
    return result


@pytest.fixture
def result_curvature_map_scale_false(mnt, m):
    result = m.curvature_map(mnt, scale=False, verbose=False)
    return result


@pytest.fixture
def result_omega_s_map_scale_true(mnt, m):
    result = m.omega_s_map(mnt, 25, 270, scale=True, verbose=False)
    return result


@pytest.fixture
def result_omega_s_map_scale_false(mnt, m):
    result = m.curvature_map(mnt, 25, 270, scale=False, verbose=False)
    return result


@pytest.fixture
def result_diverting_factor_map_scale_true(mnt, m):
    result = m.diverting_factor_map(mnt, 25, 270, scale=True, verbose=False)
    return result


@pytest.fixture
def result_diverting_factor_map_scale_false(mnt, m):
    result = m.diverting_factor_map(mnt, 25, 270, scale=False, verbose=False)
    return result


@pytest.fixture
def result_wind_weighting_factor_map_scale_true(mnt, m):
    result = m.wind_weighting_factor_map(mnt, 25, 270, scale=True, verbose=False)
    return result


@pytest.fixture
def result_wind_weighting_factor_map_scale_false(mnt, m):
    result = m.wind_weighting_factor_map(mnt, 25, 270, scale=False, verbose=False)
    return result


@pytest.fixture
def result_curvature_idx_scale_true_safe(mnt, m, idx_x, idx_y):
    result = m.curvature_idx(mnt, idx_x, idx_y, scale=True, method="safe", verbose=False)
    return result


@pytest.fixture
def result_curvature_idx_scale_false_safe(mnt, m, idx_x, idx_y):
    result = m.curvature_idx(mnt, idx_x, idx_y, scale=False, method="safe", verbose=False)
    return result


@pytest.fixture
def result_curvature_idx_scale_true_other(mnt, m, idx_x, idx_y):
    result = m.curvature_idx(mnt, idx_x, idx_y, scale=True, method="other", verbose=False)
    return result


@pytest.fixture
def result_curvature_idx_scale_false_other(mnt, m, idx_x, idx_y):
    result = m.curvature_idx(mnt, idx_x, idx_y, scale=False, method="other", verbose=False)
    return result


@pytest.fixture
def result_omega_s_idx_scale_true_safe(mnt, m, idx_x, idx_y):
    result = m.omega_s_idx(mnt, 25, 270, idx_x, idx_y, scale=True, method="safe", verbose=False)
    return result


@pytest.fixture
def result_omega_s_idx_scale_false_safe(mnt, m, idx_x, idx_y):
    result = m.omega_s_idx(mnt, 25, 270, idx_x, idx_y, scale=False, method="safe", verbose=False)
    return result


@pytest.fixture
def result_omega_s_idx_scale_true_other(mnt, m, idx_x, idx_y):
    result = m.omega_s_idx(mnt, 25, 270, idx_x, idx_y, scale=True, method="other", verbose=False)
    return result


@pytest.fixture
def result_omega_s_idx_scale_false_other(mnt, m, idx_x, idx_y):
    result = m.omega_s_idx(mnt, 25, 270, idx_x, idx_y, scale=False, method="other", verbose=False)
    return result


@pytest.fixture
def result_wind_weighting_factor_idx_scale_true_safe(mnt, m, idx_x, idx_y):
    result = m.wind_weighting_factor_idx(mnt, 25, 270, idx_x, idx_y, scale=True, method="safe", verbose=False)
    return result


@pytest.fixture
def result_wind_weighting_factor_idx_scale_false_safe(mnt, m, idx_x, idx_y):
    result = m.wind_weighting_factor_idx(mnt, 25, 270, idx_x, idx_y, scale=False, method="safe", verbose=False)
    return result


@pytest.fixture
def result_wind_weighting_factor_idx_scale_true_other(mnt, m, idx_x, idx_y):
    result = m.wind_weighting_factor_idx(mnt, 25, 270, idx_x, idx_y, scale=True, method="other", verbose=False)
    return result


@pytest.fixture
def result_wind_weighting_factor_idx_scale_false_other(mnt, m, idx_x, idx_y):
    result = m.wind_weighting_factor_idx(mnt, 25, 270, idx_x, idx_y, scale=False, method="other", verbose=False)
    return result


@pytest.fixture
def result_diverting_factor_idx_scale_true_safe(mnt, m, idx_x, idx_y):
    result = m.diverting_factor_idx(mnt, 25, 270, idx_x, idx_y, scale=True, method="safe", verbose=False)
    return result


@pytest.fixture
def result_diverting_factor_idx_scale_false_safe(mnt, m, idx_x, idx_y):
    result = m.diverting_factor_idx(mnt, 25, 270, idx_x, idx_y, scale=False, method="safe", verbose=False)
    return result


@pytest.fixture
def result_diverting_factor_idx_scale_true_other(mnt, m, idx_x, idx_y):
    result = m.diverting_factor_idx(mnt, 25, 270, idx_x, idx_y, scale=True, method="other", verbose=False)
    return result


@pytest.fixture
def result_diverting_factor_idx_scale_false_other(mnt, m, idx_x, idx_y):
    result = m.diverting_factor_idx(mnt, 25, 270, idx_x, idx_y, scale=False, method="other", verbose=False)
    return result


def test_terrain_slope_map_no_nans(result_terrain_slope_map):
    assert np.isnan(result_terrain_slope_map).sum() == 0


def test_terrain_slope_map_good_shape(result_terrain_slope_map, mnt):
    assert result_terrain_slope_map.shape == mnt.shape


def test_terrain_slope_map_good_dtype(result_terrain_slope_map):
    assert result_terrain_slope_map.dtype == np.float32


def test_terrain_slope_idx_no_nans(result_terrain_slope_idx):
    assert np.isnan(result_terrain_slope_idx).sum() == 0


def test_terrain_slope_idx_good_shape(result_terrain_slope_idx, idx_x):
    assert result_terrain_slope_idx.shape == idx_x.shape


def test_terrain_slope_idx_good_dtype(result_terrain_slope_idx):
    assert result_terrain_slope_idx.dtype == np.float32


def test_terrain_slope_azimuth_map_no_nans(result_terrain_slope_azimuth_map):
    assert np.isnan(result_terrain_slope_azimuth_map).sum() == 0


def test_terrain_slope_azimuth_map_good_shape(result_terrain_slope_azimuth_map, mnt):
    assert result_terrain_slope_azimuth_map.shape == mnt.shape


def test_terrain_slope_azimuth_map_good_dtype(result_terrain_slope_azimuth_map):
    assert result_terrain_slope_azimuth_map.dtype == np.float32


def test_terrain_slope_azimuth_idx_no_nans(result_terrain_slope_azimuth_idx):
    assert np.isnan(result_terrain_slope_azimuth_idx).sum() == 0


def test_terrain_slope_azimuth_idx_good_shape(result_terrain_slope_azimuth_idx, idx_x):
    assert result_terrain_slope_azimuth_idx.shape == idx_x.shape


def test_terrain_slope_azimuth_idx_good_dtype(result_terrain_slope_azimuth_idx):
    assert result_terrain_slope_azimuth_idx.dtype == np.float32


def test_curvature_map_no_nans(result_curvature_map_scale_true, result_curvature_map_scale_false):
    assert np.isnan(result_curvature_map_scale_true).sum() == 0
    assert np.isnan(result_curvature_map_scale_false).sum() == 0


def test_curvature_map_good_shape(result_curvature_map_scale_true, result_curvature_map_scale_false, mnt):
    assert result_curvature_map_scale_true.shape == mnt.shape
    assert result_curvature_map_scale_false.shape == mnt.shape


def test_curvature_map_good_dtype(result_curvature_map_scale_true, result_curvature_map_scale_false):
    assert result_curvature_map_scale_true.dtype == np.float32
    assert result_curvature_map_scale_false.dtype == np.float32


def test_curvature_idx_no_nans(result_curvature_idx_scale_true_safe, result_curvature_idx_scale_false_safe,
                               result_curvature_idx_scale_true_other, result_curvature_idx_scale_false_other):
    assert np.isnan(result_curvature_idx_scale_true_safe).sum() == 0
    assert np.isnan(result_curvature_idx_scale_false_safe).sum() == 0
    assert np.isnan(result_curvature_idx_scale_true_other).sum() == 0
    assert np.isnan(result_curvature_idx_scale_false_other).sum() == 0


def test_curvature_idx_good_shape(result_curvature_idx_scale_true_safe, result_curvature_idx_scale_false_safe,
                                  result_curvature_idx_scale_true_other,
                                  result_curvature_idx_scale_false_other, idx_x):
    assert result_curvature_idx_scale_true_safe.shape == idx_x.shape
    assert result_curvature_idx_scale_false_safe.shape == idx_x.shape
    assert result_curvature_idx_scale_true_other.shape == idx_x.shape
    assert result_curvature_idx_scale_false_other.shape == idx_x.shape


def test_curvature_idx_good_dtype(result_curvature_idx_scale_true_safe, result_curvature_idx_scale_false_safe,
                                  result_curvature_idx_scale_true_other, result_curvature_idx_scale_false_other):
    assert result_curvature_idx_scale_true_safe.dtype == np.float32
    assert result_curvature_idx_scale_false_safe.dtype == np.float32
    assert result_curvature_idx_scale_true_other.dtype == np.float32
    assert result_curvature_idx_scale_false_other.dtype == np.float32


def test_omega_s_map_no_nans(result_omega_s_map_scale_true, result_omega_s_map_scale_false):
    assert np.isnan(result_omega_s_map_scale_true).sum() == 0
    assert np.isnan(result_omega_s_map_scale_false).sum() == 0


def test_omega_s_map_good_shape(result_omega_s_map_scale_true, result_omega_s_map_scale_false, mnt):
    assert result_omega_s_map_scale_true.shape == mnt.shape
    assert result_omega_s_map_scale_false.shape == mnt.shape


def test_omega_s_map_good_dtype(result_omega_s_map_scale_true, result_omega_s_map_scale_false):
    assert result_omega_s_map_scale_true.dtype == np.float32
    assert result_omega_s_map_scale_false.dtype == np.float32


def test_omega_s_idx_no_nans(result_omega_s_idx_scale_true_safe, result_omega_s_idx_scale_false_safe,
                             result_omega_s_idx_scale_true_other, result_omega_s_idx_scale_false_other):
    assert np.isnan(result_omega_s_idx_scale_true_safe).sum() == 0
    assert np.isnan(result_omega_s_idx_scale_false_safe).sum() == 0
    assert np.isnan(result_omega_s_idx_scale_true_other).sum() == 0
    assert np.isnan(result_omega_s_idx_scale_false_other).sum() == 0


def test_omega_s_idx_good_shape(result_omega_s_idx_scale_true_safe, result_omega_s_idx_scale_false_safe,
                                result_omega_s_idx_scale_true_other,
                                result_omega_s_idx_scale_false_other, idx_x):
    assert result_omega_s_idx_scale_true_safe.shape == idx_x.shape
    assert result_omega_s_idx_scale_false_safe.shape == idx_x.shape
    assert result_omega_s_idx_scale_true_other.shape == idx_x.shape
    assert result_omega_s_idx_scale_false_other.shape == idx_x.shape


def test_omega_s_idx_good_dtype(result_omega_s_idx_scale_true_safe, result_omega_s_idx_scale_false_safe,
                                result_omega_s_idx_scale_true_other, result_omega_s_idx_scale_false_other):
    assert result_omega_s_idx_scale_true_safe.dtype == np.float32
    assert result_omega_s_idx_scale_false_safe.dtype == np.float32
    assert result_omega_s_idx_scale_true_other.dtype == np.float32
    assert result_omega_s_idx_scale_false_other.dtype == np.float32


def test_wind_weighting_factor_map_no_nans(result_wind_weighting_factor_map_scale_true,
                                           result_wind_weighting_factor_map_scale_false):
    assert np.isnan(result_wind_weighting_factor_map_scale_true).sum() == 0
    assert np.isnan(result_wind_weighting_factor_map_scale_false).sum() == 0


def test_wind_weighting_factor_map_good_shape(result_wind_weighting_factor_map_scale_true,
                                              result_wind_weighting_factor_map_scale_false, mnt):
    assert result_wind_weighting_factor_map_scale_true.shape == mnt.shape
    assert result_wind_weighting_factor_map_scale_false.shape == mnt.shape


def test_wind_weighting_factor_map_good_dtype(result_wind_weighting_factor_map_scale_true,
                                              result_wind_weighting_factor_map_scale_false):
    assert result_wind_weighting_factor_map_scale_true.dtype == np.float32
    assert result_wind_weighting_factor_map_scale_false.dtype == np.float32


def test_wind_weighting_factor_idx_no_nans(result_wind_weighting_factor_idx_scale_true_safe,
                                           result_wind_weighting_factor_idx_scale_false_safe,
                                           result_wind_weighting_factor_idx_scale_true_other,
                                           result_wind_weighting_factor_idx_scale_false_other):
    assert np.isnan(result_wind_weighting_factor_idx_scale_true_safe).sum() == 0
    assert np.isnan(result_wind_weighting_factor_idx_scale_false_safe).sum() == 0
    assert np.isnan(result_wind_weighting_factor_idx_scale_true_other).sum() == 0
    assert np.isnan(result_wind_weighting_factor_idx_scale_false_other).sum() == 0


def test_wind_weighting_factor_idx_good_shape(result_wind_weighting_factor_idx_scale_true_safe,
                                              result_wind_weighting_factor_idx_scale_false_safe,
                                              result_wind_weighting_factor_idx_scale_true_other,
                                              result_wind_weighting_factor_idx_scale_false_other, idx_x):
    assert result_wind_weighting_factor_idx_scale_true_safe.shape == idx_x.shape
    assert result_wind_weighting_factor_idx_scale_false_safe.shape == idx_x.shape
    assert result_wind_weighting_factor_idx_scale_true_other.shape == idx_x.shape
    assert result_wind_weighting_factor_idx_scale_false_other.shape == idx_x.shape


def test_wind_weighting_factor_idx_good_dtype(result_wind_weighting_factor_idx_scale_true_safe,
                                              result_wind_weighting_factor_idx_scale_false_safe,
                                              result_wind_weighting_factor_idx_scale_true_other,
                                              result_wind_weighting_factor_idx_scale_false_other):
    assert result_wind_weighting_factor_idx_scale_true_safe.dtype == np.float32
    assert result_wind_weighting_factor_idx_scale_false_safe.dtype == np.float32
    assert result_wind_weighting_factor_idx_scale_true_other.dtype == np.float32
    assert result_wind_weighting_factor_idx_scale_false_other.dtype == np.float32


def test_diverting_factor_map_no_nans(result_diverting_factor_map_scale_true, result_diverting_factor_map_scale_false):
    assert np.isnan(result_diverting_factor_map_scale_true).sum() == 0
    assert np.isnan(result_diverting_factor_map_scale_false).sum() == 0


def test_diverting_factor_map_good_shape(result_diverting_factor_map_scale_true,
                                         result_diverting_factor_map_scale_false, mnt):
    assert result_diverting_factor_map_scale_true.shape == mnt.shape
    assert result_diverting_factor_map_scale_false.shape == mnt.shape


def test_diverting_factor_map_good_dtype(result_diverting_factor_map_scale_true,
                                         result_diverting_factor_map_scale_false):
    assert result_diverting_factor_map_scale_true.dtype == np.float32
    assert result_diverting_factor_map_scale_false.dtype == np.float32


def test_diverting_factor_idx_no_nans(result_diverting_factor_idx_scale_true_safe,
                                      result_diverting_factor_idx_scale_false_safe,
                                      result_diverting_factor_idx_scale_true_other,
                                      result_diverting_factor_idx_scale_false_other):
    assert np.isnan(result_diverting_factor_idx_scale_true_safe).sum() == 0
    assert np.isnan(result_diverting_factor_idx_scale_false_safe).sum() == 0
    assert np.isnan(result_diverting_factor_idx_scale_true_other).sum() == 0
    assert np.isnan(result_diverting_factor_idx_scale_false_other).sum() == 0


def test_diverting_factor_idx_good_shape(result_diverting_factor_idx_scale_true_safe,
                                         result_diverting_factor_idx_scale_false_safe,
                                         result_diverting_factor_idx_scale_true_other,
                                         result_diverting_factor_idx_scale_false_other, idx_x):
    assert result_diverting_factor_idx_scale_true_safe.shape == idx_x.shape
    assert result_diverting_factor_idx_scale_false_safe.shape == idx_x.shape
    assert result_diverting_factor_idx_scale_true_other.shape == idx_x.shape
    assert result_diverting_factor_idx_scale_false_other.shape == idx_x.shape


def test_diverting_factor_idx_good_dtype(result_diverting_factor_idx_scale_true_safe,
                                         result_diverting_factor_idx_scale_false_safe,
                                         result_diverting_factor_idx_scale_true_other,
                                         result_diverting_factor_idx_scale_false_other):
    assert result_diverting_factor_idx_scale_true_safe.dtype == np.float32
    assert result_diverting_factor_idx_scale_false_safe.dtype == np.float32
    assert result_diverting_factor_idx_scale_true_other.dtype == np.float32
    assert result_diverting_factor_idx_scale_false_other.dtype == np.float32


def test_terrain_slope_map_and_idx_consistent():
    m = MicroMet()
    array_test = np.array([[13, 24, 32, 50, 36, 28, 28],
                           [12, 11, 41, 51, 38, 10, 20],
                           [50, 13, 28, 48, 28, 5, 18],
                           [7, 32, 41, 53, 59, 28, 8],
                           [18, 19, 48, 20, 29, 24, 93],
                           [19, 29, 30, 28, 18, 49, 50]])

    result_map = m.terrain_slope_map(array_test, 1, verbose=False)[3, 4]
    result_idx = m.terrain_slope_idx(array_test, 1, [4], [3], verbose=False)

    assert_allclose(result_map, result_idx)


def test_terrain_slope_map_good_result():
    m = MicroMet()

    array_test = np.array([[13, 24, 32, 50],
                           [12, 11, 41, 51],
                           [50, 13, 28, 48],
                           [7, 32, 41, 53]])

    result_map = m.terrain_slope_map(array_test, 1, verbose=False)
    expected_results = np.array([[1.48050674, 1.50876899, 1.5076349, 1.51538309],
                                 [1.51687339, 1.50640289, 1.52108546, 1.47161907],
                                 [1.54383732, 1.50513128, 1.51371554, 1.52090016],
                                 [1.55069422, 1.5315932, 1.51102598, 1.49402444]])

    assert_allclose(result_map, expected_results)


def test_terrain_slope_azimuth_map_and_idx_consistent():
    m = MicroMet()
    array_test = np.array([[13, 24, 32, 50, 36, 28, 28],
                           [12, 11, 41, 51, 38, 10, 20],
                           [50, 13, 28, 48, 28, 5, 18],
                           [7, 32, 41, 53, 59, 28, 8],
                           [18, 19, 48, 20, 29, 24, 93],
                           [19, 29, 30, 28, 18, 49, 50]])

    result_map = m.terrain_slope_azimuth_map(array_test, 1, verbose=False)[3, 4]
    result_idx = m.terrain_slope_azimuth_idx(array_test, 1, [4], [3], verbose=False)

    assert_allclose(result_map, result_idx)


def test_terrain_slope_azimuth_map_good_result():
    m = MicroMet()

    array_test = np.array([[13, 24, 32, 50],
                           [12, 11, 41, 51],
                           [50, 13, 28, 48],
                           [7, 32, 41, 53]])

    result_map = m.terrain_slope_azimuth_map(array_test, 1, verbose=False)
    expected_results = np.array([[4.80304887, 5.65210592, 4.10684432, 4.65689048],
                                 [6.22918381, 5.07493322, 4.81205763, 4.81205763],
                                 [4.64492396, 5.47453552, 4.71238898, 4.66243058],
                                 [5.75655804, 3.87149231, 3.82100646, 4.31759786]])

    assert_allclose(result_map, expected_results)


def test_curvature_map_and_idx_consistent():
    m = MicroMet()
    array_test = np.array([[13, 24, 32, 50, 36, 28, 28],
                           [12, 11, 41, 51, 38, 10, 20],
                           [50, 13, 28, 48, 28, 5, 18],
                           [7, 32, 41, 53, 59, 28, 8],
                           [18, 19, 48, 20, 29, 24, 93],
                           [19, 29, 30, 28, 18, 49, 50]])

    result_map = m.curvature_map(array_test, verbose=False)[3, 4]
    result_idx = m.curvature_idx(array_test, [4], [3], verbose=False)

    assert_allclose(result_map, result_idx)


def test_curvature_map_good_result():
    m = MicroMet()

    array_test = np.array([[13, 24, 32, 50],
                           [12, 11, 41, 51],
                           [50, 13, 28, 48],
                           [7, 32, 41, 53]])

    result_map_1 = m.curvature_map(array_test, verbose=False)[1, 1]
    expected_results_1 = -0.19803762

    result_map_2 = m.curvature_map(array_test, verbose=False)[1, 2]
    expected_results_2 = 0.121523505

    assert_almost_equal(result_map_1, expected_results_1)
    assert_almost_equal(result_map_2, expected_results_2)


def test_curvature_idx_good_result():
    m = MicroMet()

    array_test = np.array([[13, 24, 32, 50],
                           [12, 11, 41, 51],
                           [50, 13, 28, 48],
                           [7, 32, 41, 53]])

    result_map_1 = m.curvature_idx(array_test, 1, 1, verbose=False)
    expected_results_1 = -0.19803762

    result_map_2 = m.curvature_idx(array_test, 2, 1, verbose=False)
    expected_results_2 = 0.121523505

    assert_almost_equal(result_map_1, expected_results_1)
    assert_almost_equal(result_map_2, expected_results_2)


def test_omega_s_map_good_result():
    m = MicroMet()

    array_test = np.array([[13, 24, 32, 50],
                           [12, 11, 41, 51],
                           [50, 13, 28, 48],
                           [7, 32, 41, 53]])

    result_map_1 = np.round(m.omega_s_map(array_test, 1, 270, verbose=False), 2)
    result_map_2 = np.round(m.omega_s_map(array_test, 1, 270, scale=True, scaling_factor=10, verbose=False), 2)

    result_1 = np.array([[1.48050674, 1.50876899, 1.5076349, 1.51538309],
                         [1.51687339, 1.50640289, 1.52108546, 1.47161907],
                         [1.54383732, 1.50513128, 1.51371554, 1.52090016],
                         [1.55069422, 1.5315932, 1.51102598, 1.49402444]])

    result_2 = np.array([[4.80304887, 5.65210592, 4.10684432, 4.65689048],
                         [6.22918381, 5.07493322, 4.81205763, 4.81205763],
                         [4.64492396, 5.47453552, 4.71238898, 4.66243058],
                         [5.75655804, 3.87149231, 3.82100646, 4.31759786]])

    expected_results = np.round(result_1 * np.cos(270 - result_2), 2)

    assert_allclose(result_map_1, expected_results)
    assert_allclose(result_map_2, np.round(expected_results / (2 * 10), 2))


def test_omega_s_map_and_idx_consistent():
    m = MicroMet()

    array_test = np.array([[13, 24, 32, 50, 36, 28, 28],
                           [12, 11, 41, 51, 38, 10, 20],
                           [50, 13, 28, 48, 28, 5, 18],
                           [7, 32, 41, 53, 59, 28, 8],
                           [18, 19, 48, 20, 29, 24, 93],
                           [19, 29, 30, 28, 18, 49, 50]])

    result_map_1 = m.omega_s_map(array_test, 1, 270, verbose=False)
    result_map_1 = result_map_1[3, 4]
    result_idx_1 = m.omega_s_idx(array_test, 1, 270, [4], [3], verbose=False)

    result_map_2 = m.omega_s_map(array_test, 1, 270, scale=True, scaling_factor=10, verbose=False)
    result_map_2 = result_map_2[3, 4]
    result_idx_2 = m.omega_s_idx(array_test, 1, 270, [4], [3], scale=True, scaling_factor=10, verbose=False)

    assert_allclose(result_map_1, result_idx_1)
    assert_allclose(result_map_2, result_idx_2)


