import pytest
from numpy.testing import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from downscale.Operators.topo_utils import *


def test_std_slicing_numpy():
    tu = Sgp_helbig()

    array_test_1 = np.array([[1, 1, 1, 1],
                             [1, -2, 2, 1],
                             [1, -2, 2, 1],
                             [1, 1, 1, 1]])

    array_test_2 = np.array([[1, 1, 1, 1],
                             [1, 1, 2, 1],
                             [1, -2, 2, 1],
                             [1, 1, 1, 1]])

    array_test_3 = np.array([[1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [1, 1, -2, 2],
                             [1, 1, -2, 2]])

    y_left = np.array([1])
    y_right = np.array([2])
    x_left = np.array([1])
    x_right = np.array([2])

    result_1 = tu.std_slicing_numpy(array_test_1, y_left, y_right, x_left, x_right)

    y_left = np.array([0])
    y_right = np.array([1])
    x_left = np.array([0])
    x_right = np.array([1])

    result_2 = tu.std_slicing_numpy(array_test_2, y_left, y_right, x_left, x_right)

    y_left = np.array([0, 2])
    y_right = np.array([1, 3])
    x_left = np.array([0, 2])
    x_right = np.array([1, 3])

    result_3 = tu.std_slicing_numpy(array_test_3, y_left, y_right, x_left, x_right)

    assert result_1 == 2.0
    assert result_2 == 0.0
    assert (result_3[0] == 0.0 and result_3[1] == 2.0)


def test_normalize_topo():
    tu = Sgp_helbig()

    array_test_1 = np.array([[100, -100, 100, -100],
                             [100, -100, 100, -100],
                             [100, -100, 100, -100],
                             [100, -100, 100, -100]])

    expected_result_1 = np.array([[1, -1, 1, -1],
                                  [1, -1, 1, -1],
                                  [1, -1, 1, -1],
                                  [1, -1, 1, -1]])

    result_1 = tu.normalize_topo(array_test_1, np.array([0]), np.array([100]))
    assert_array_almost_equal(result_1, expected_result_1)


def test_normalize_topo_broadcast():
    tu = Sgp_helbig()

    array_test_1 = 100 * np.ones((10, 20, 30))

    expected_result_1 = 25 * np.ones((10, 20, 30))

    result_1 = tu.normalize_topo(array_test_1, np.array([50]), np.array([2]))
    assert_array_almost_equal(result_1, expected_result_1)


def test_mean_peak_valley():
    tu = Sgp_helbig()
    test_array_1 = np.array([1, np.nan, 2])
    test_array_2 = np.array([1, 2, 2])
    assert_array_almost_equal(tu.mean_peak_valley(test_array_1), np.float32(2*np.nanstd(test_array_1)))
    assert_array_almost_equal(tu.mean_peak_valley(test_array_2), np.float32(2*np.nanstd(test_array_2)))
    assert not np.array_equal(tu.mean_peak_valley(test_array_1), np.float32(2*np.std(test_array_1)))


def test_laplacian_classic():
    tu = Sgp_helbig()
    array_test = np.array([[1, 2], [3, 4]])
    expected_result = np.array([[3, 1], [-1, -3]])
    result = tu.laplacian_map(array_test, 1, librairie="numpy", helbig=False, verbose=False)
    assert_array_almost_equal(result, expected_result)


def test_laplacian_helbig():
    tu = Sgp_helbig()
    array_test = np.array([[1, 2], [3, 4]])
    expected_result = np.array([[3, 1], [-1, -3]])/4
    result = tu.laplacian_map(array_test, 1, librairie="numpy", helbig=True, verbose=False)
    assert_array_almost_equal(result, expected_result)


def test_mu_helbig_map():
    tu = Sgp_helbig()
    array_test = np.array([[1, 2], [3, 4]])
    expected_result = np.sqrt(np.array([[2.5, 2.5], [2.5, 2.5]]))
    result = tu.mu_helbig_map(array_test, 1, verbose=False)
    assert_array_almost_equal(result, expected_result)


def test_mu_helbig_map_idx():
    tu = Sgp_helbig()
    array_test = np.array([[1, 2], [3, 10]])
    expected_result = np.sqrt(np.array((2.5)))
    result = tu.mu_helbig_idx(array_test, 1, [0], [0], verbose=False)
    assert_almost_equal(result, expected_result)


def test_rolling_mean():

    tu = Sgp_helbig()

    array_test = np.array([[1,2,3,4],
                           [2,3,4,5],
                           [3,4,5,6],
                           [4,5,6,7]])

    result_1 = np.mean([[1,2], [2,3]])
    result_2 = np.mean([[1,2,3], [2,3,4]])
    result_3 = np.mean([[2,3,4], [3,4,5]])
    result_4 = np.mean([[3,4], [4,5]])

    result_5 = np.mean([[1,2], [2,3], [3,4]])
    result_6 = np.mean([[1,2,3], [2,3,4], [3,4,5]])
    result_7 = np.mean([[2,3,4], [3,4,5], [4,5,6]])
    result_8 = np.mean([[3,4], [4,5], [5,6]])

    result_9 = np.mean([[2,3], [3,4], [4,5]])
    result_10 = np.mean([[2,3,4], [3,4,5], [4,5,6]])
    result_11 = np.mean([[3,4,5], [4,5,6], [5,6,7]])
    result_12 = np.mean([[4,5], [5,6], [6,7]])

    result_13 = np.mean([[3,4], [4,5]])
    result_14 = np.mean([[3,4,5], [4,5,6]])
    result_15 = np.mean([[4,5,6], [5,6,7]])
    result_16 = np.mean([[5,6], [6,7]])

    expected_result = np.array([[result_1, result_2, result_3, result_4],
                               [result_5, result_6, result_7, result_8],
                               [result_9, result_10, result_11, result_12],
                               [result_13, result_14, result_15, result_16]])
    result = tu.rolling_window_mean_numpy(array_test, np.empty_like(array_test).astype(np.float32), 1, 1)
    assert_array_almost_equal(result, expected_result)


def test_rolling_tpi():

    tu = Sgp_helbig()

    array_test = np.array([[1,2,3,4],
                           [2,3,4,5],
                           [3,4,5,6],
                           [4,5,6,7]])

    result_1 = np.mean([[1,2], [2,3]])
    result_2 = np.mean([[1,2,3], [2,3,4]])
    result_3 = np.mean([[2,3,4], [3,4,5]])
    result_4 = np.mean([[3,4], [4,5]])

    result_5 = np.mean([[1,2], [2,3], [3,4]])
    result_6 = np.mean([[1,2,3], [2,3,4], [3,4,5]])
    result_7 = np.mean([[2,3,4], [3,4,5], [4,5,6]])
    result_8 = np.mean([[3,4], [4,5], [5,6]])

    result_9 = np.mean([[2,3], [3,4], [4,5]])
    result_10 = np.mean([[2,3,4], [3,4,5], [4,5,6]])
    result_11 = np.mean([[3,4,5], [4,5,6], [5,6,7]])
    result_12 = np.mean([[4,5], [5,6], [6,7]])

    result_13 = np.mean([[3,4], [4,5]])
    result_14 = np.mean([[3,4,5], [4,5,6]])
    result_15 = np.mean([[4,5,6], [5,6,7]])
    result_16 = np.mean([[5,6], [6,7]])

    expected_result = array_test - np.array([[result_1, result_2, result_3, result_4],
                                               [result_5, result_6, result_7, result_8],
                                               [result_9, result_10, result_11, result_12],
                                               [result_13, result_14, result_15, result_16]])
    result_1 = tu.tpi_map(array_test, 1, 1, librairie="numba")
    result_2 = tu.tpi_map(array_test, 1, 1, librairie="numpy")

    assert_array_almost_equal(result_1, expected_result)
    assert_array_almost_equal(result_2, expected_result)


def test_rolling_tpi_idx():
    tu = Sgp_helbig()

    array_test = np.array([[1, 2, 3, 4],
                           [2, 3, 4, 5],
                           [3, 4, 5, 6],
                           [4, 5, 6, 7]])

    result_1 = np.mean([[1, 2], [2, 3]])
    result_2 = np.mean([[1, 2, 3], [2, 3, 4]])
    result_3 = np.mean([[2, 3, 4], [3, 4, 5]])
    result_4 = np.mean([[3, 4], [4, 5]])

    result_5 = np.mean([[1, 2], [2, 3], [3, 4]])
    result_6 = np.mean([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    result_7 = np.mean([[2, 3, 4], [3, 4, 5], [4, 5, 6]])
    result_8 = np.mean([[3, 4], [4, 5], [5, 6]])

    result_9 = np.mean([[2, 3], [3, 4], [4, 5]])
    result_10 = np.mean([[2, 3, 4], [3, 4, 5], [4, 5, 6]])
    result_11 = np.mean([[3, 4, 5], [4, 5, 6], [5, 6, 7]])
    result_12 = np.mean([[4, 5], [5, 6], [6, 7]])

    result_13 = np.mean([[3, 4], [4, 5]])
    result_14 = np.mean([[3, 4, 5], [4, 5, 6]])
    result_15 = np.mean([[4, 5, 6], [5, 6, 7]])
    result_16 = np.mean([[5, 6], [6, 7]])

    mean_window = np.array([[result_1, result_2, result_3, result_4],
                                 [result_5, result_6, result_7, result_8],
                                 [result_9, result_10, result_11, result_12],
                                 [result_13, result_14, result_15, result_16]])

    expected_result = array_test - mean_window
    result_1 = tu.tpi_idx(array_test, [0, 1], [1, 2], 1, 1)

    assert_array_almost_equal(result_1, expected_result[[1, 2], [0, 1]])


def test_mu_helbig_average():

    tu = Sgp_helbig()

    array_test = np.array([[1, 2], [3, 4]])
    expected_result = np.sqrt(np.array([[2.5, 2.5], [2.5, 2.5]]))

    result_1 = tu.mu_helbig_average(array_test, 1, reduce_mnt=False, type="map", librairie="numba", verbose=False, x_win=1, y_win=1)
    result_2 = tu.mu_helbig_average(array_test, 1, reduce_mnt=False, type="map", librairie="numpy", verbose=False, x_win=1, y_win=1)
    result_3 = tu.mu_helbig_average(array_test, 1, idx_x=np.array([0, 1]), idx_y=np.array([1, 1]), reduce_mnt=False, type="indexes", verbose=False,  x_win=1, y_win=1)

    assert_array_almost_equal(result_1, expected_result)
    assert_array_almost_equal(result_2, expected_result)
    assert_array_almost_equal(result_3, expected_result[[1, 1], [0, 1]])


def test_xsi_helbig_map():

    tu = Sgp_helbig()

    array_test = np.array([[1, 2, 3, 4],
                           [2, 3, 4, 5],
                           [3, 4, 5, 6],
                           [4, 5, 6, 7]])

    result_1 = np.std([[1, 2], [2, 3]])
    result_2 = np.std([[1, 2, 3], [2, 3, 4]])
    result_3 = np.std([[2, 3, 4], [3, 4, 5]])
    result_4 = np.std([[3, 4], [4, 5]])

    result_5 = np.std([[1, 2], [2, 3], [3, 4]])
    result_6 = np.std([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    result_7 = np.std([[2, 3, 4], [3, 4, 5], [4, 5, 6]])
    result_8 = np.std([[3, 4], [4, 5], [5, 6]])

    result_9 = np.std([[2, 3], [3, 4], [4, 5]])
    result_10 = np.std([[2, 3, 4], [3, 4, 5], [4, 5, 6]])
    result_11 = np.std([[3, 4, 5], [4, 5, 6], [5, 6, 7]])
    result_12 = np.std([[4, 5], [5, 6], [6, 7]])

    result_13 = np.std([[3, 4], [4, 5]])
    result_14 = np.std([[3, 4, 5], [4, 5, 6]])
    result_15 = np.std([[4, 5, 6], [5, 6, 7]])
    result_16 = np.std([[5, 6], [6, 7]])

    std = np.array([[result_1, result_2, result_3, result_4],
                    [result_5, result_6, result_7, result_8],
                    [result_9, result_10, result_11, result_12],
                    [result_13, result_14, result_15, result_16]])

    mu = tu.mu_helbig_average(array_test, 1, reduce_mnt=False, type="map", librairie="numba", verbose=False)
    expected_result = np.sqrt(2) * std / mu
    result = tu.xsi_helbig_map(array_test, mu, reduce_mnt=False, x_win=1, y_win=1, librairie="numba", verbose=False)

    assert_array_almost_equal(result, expected_result)


def test_x_sgp_topo_helbig_idx_shape():

    tu = Sgp_helbig()

    array_test = np.array([[1, 2, 3, 4],
                           [2, 3, 8, 5],
                           [3, 90, 5, 6],
                           [4, 5, 6, 7]])

    result = tu.x_sgp_topo_helbig_idx(array_test, dx=1, L=4, type="map", x_win=1, y_win=1, reduce_mnt=False, verbose=False)
    assert array_test.shape == result.shape