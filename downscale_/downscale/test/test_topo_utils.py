import numpy as np
from numpy.testing import *
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from downscale.operators.topo_utils import *
from downscale.operators.helbig import *


def test_std_slicing_numpy():
    """test passes 30/11/2021"""
    tu = SgpHelbig()

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

    result_1 = tu.std_slicing_numpy_loop(array_test_1, y_left, y_right, x_left, x_right)

    y_left = np.array([0])
    y_right = np.array([1])
    x_left = np.array([0])
    x_right = np.array([1])

    result_2 = tu.std_slicing_numpy_loop(array_test_2, y_left, y_right, x_left, x_right)

    y_left = np.array([0, 2])
    y_right = np.array([1, 3])
    x_left = np.array([0, 2])
    x_right = np.array([1, 3])

    result_3 = tu.std_slicing_numpy_loop(array_test_3, y_left, y_right, x_left, x_right)

    assert result_1 == 2.0
    assert result_2 == 0.0
    assert (result_3[0] == 0.0 and result_3[1] == 2.0)


def test_normalize_topo():
    """test passes 30/11/2021"""
    tu = SgpHelbig()

    array_test_1 = np.array([[100, -100, 100, -100],
                             [100, -100, 100, -100],
                             [100, -100, 100, -100],
                             [100, -100, 100, -100]])

    expected_result_1 = np.array([[1, -1, 1, -1],
                                  [1, -1, 1, -1],
                                  [1, -1, 1, -1],
                                  [1, -1, 1, -1]])

    result_1 = tu.normalize_topo(array_test_1, np.array([0]), np.array([100]), verbose=False)
    assert_array_almost_equal(result_1, expected_result_1)


def test_normalize_topo_broadcast():
    """test passes 30/11/2021"""
    tu = SgpHelbig()

    array_test_1 = 100 * np.ones((10, 20, 30))

    expected_result_1 = 25 * np.ones((10, 20, 30))

    result_1 = tu.normalize_topo(array_test_1, np.array([50]), np.array([2]), verbose=False)
    assert_array_almost_equal(result_1, expected_result_1)


def test_mean_peak_valley():
    """test passes 30/11/2021"""

    tu = SgpHelbig()
    test_array_1 = np.array([1, np.nan, 2])
    test_array_2 = np.array([1, 2, 2])
    assert_array_almost_equal(tu.mean_peak_valley(test_array_1, axis=0, verbose=False), np.float32(2 * np.nanstd(test_array_1)))
    assert_array_almost_equal(tu.mean_peak_valley(test_array_2, axis=0, verbose=False), np.float32(2 * np.nanstd(test_array_2)))
    assert not np.array_equal(tu.mean_peak_valley(test_array_1, axis=0, verbose=False), np.float32(2 * np.std(test_array_1)))


def test_laplacian_classic():
    """test passes 30/11/2021"""
    tu = SgpHelbig()
    array_test_1 = np.array([[1, 2],
                             [3, 4]])

    expected_result_1 = np.array([[3, 1],
                                  [-1, -3]])

    array_test_2 = np.array([[12, 14, 28, 32],
                             [15, 27, 42, 53],
                             [41, 40, 21, 13],
                             [18, 12, 21, 42]])

    expected_result_2 = np.array([[5, 25, 4, 17],
                                  [35, 3, -39, -72],
                                  [-50, -59, 32, 77],
                                  [17, 43, 12, -50]])

    result_1 = tu.laplacian_map(array_test_1, 1, library="numpy", helbig=False, verbose=False)
    result_2 = tu.laplacian_map(array_test_2, 1, library="numpy", helbig=False, verbose=False)
    result_2 = tu.laplacian_map(array_test_2, 1, library="numba", helbig=False, verbose=False)
    result_2 = tu.laplacian_map(array_test_2, 1, library="numba", helbig=False, verbose=False)
    assert_array_almost_equal(result_1, expected_result_1)
    assert_array_almost_equal(result_2, expected_result_2)


def test_laplacian_helbig():
    """test passes 30/11/2021"""
    tu = SgpHelbig()

    array_test_1 = np.array([[1, 2],
                             [3, 4]])

    expected_result_1 = np.array([[3, 1],
                                  [-1, -3]]) / 4

    array_test_2 = np.array([[12, 14, 28, 32],
                             [15, 27, 42, 53],
                             [41, 40, 21, 13],
                             [18, 12, 21, 42]])

    expected_result_2 = np.array([[5, 25, 4, 17],
                                  [35, 3, -39, -72],
                                  [-50, -59, 32, 77],
                                  [17, 43, 12, -50]]) / 4

    result_1 = tu.laplacian_map(array_test_1, 1, library="numpy", helbig=True, verbose=False)
    result_2 = tu.laplacian_map(array_test_2, 1, library="numpy", helbig=True, verbose=False)
    result_2 = tu.laplacian_map(array_test_2, 1, library="numba", helbig=True, verbose=False)
    result_2 = tu.laplacian_map(array_test_2, 1, library="numba", helbig=True, verbose=False)
    assert_array_almost_equal(result_1, expected_result_1)
    assert_array_almost_equal(result_2, expected_result_2)


def test_laplacian_idx_and_map_give_same_result():
    """test passes 30/11/2021"""
    tu = SgpHelbig()

    array_test = np.array([[12, 14, 28, 32],
                           [15, 27, 42, 53],
                           [41, 40, 21, 13],
                           [18, 12, 21, 42]])

    result_idx_1 = tu.laplacian_idx(array_test, np.array([1]), np.array([2]), 1, library="numpy", helbig=False,
                                    verbose=False)
    result_map_1 = tu.laplacian_map(array_test, 1, library="numpy", helbig=False, verbose=False)[2, 1]

    result_idx_2 = tu.laplacian_idx(array_test, np.array([1]), np.array([2]), 1, library="numba", helbig=False,
                                    verbose=False)
    result_map_2 = tu.laplacian_map(array_test, 1, library="numba", helbig=False, verbose=False)[2, 1]

    assert_array_almost_equal(result_map_1, result_idx_1)
    assert_array_almost_equal(result_map_2, result_idx_2)


def test_laplacian_idx_result():
    """test passes 30/11/2021"""
    tu = SgpHelbig()
    array_test = np.array([[12, 14, 28, 32],
                           [15, 27, 42, 53],
                           [41, 40, 21, 13],
                           [18, 12, 21, 42]])
    expected_result_2 = np.array([[5, 25, 4, 17],
                                  [35, 3, -39, -72],
                                  [-50, -59, 32, 77],
                                  [17, 43, 12, -50]]) / 4

    result_idx_2 = tu.laplacian_idx(array_test, np.array([1]), np.array([2]), 1, library="numpy", helbig=True,
                                    verbose=False)
    result_idx_3 = tu.laplacian_idx(array_test, np.array([1]), np.array([2]), 1, library="numba", helbig=True,
                                    verbose=False)
    assert_array_almost_equal(result_idx_2, expected_result_2[2, 1])
    assert_array_almost_equal(result_idx_3, expected_result_2[2, 1])


def test_rolling_mean():
    """This test is not valid anymore"""
    tu = SgpHelbig()

    array_test = np.array([[1, 2, 3, 4],
                           [2, 3, 4, 5],
                           [3, 4, 5, 6],
                           [4, 5, 6, 7]])
    result_1 = np.mean([[1, 2],
                        [2, 3]])
    result_2 = np.mean([[1, 2, 3],
                        [2, 3, 4]])
    result_3 = np.mean([[2, 3, 4],
                        [3, 4, 5]])
    result_4 = np.mean([[3, 4],
                        [4, 5]])
    result_5 = np.mean([[1, 2],
                        [2, 3],
                        [3, 4]])
    result_6 = np.mean([[1, 2, 3],
                        [2, 3, 4],
                        [3, 4, 5]])
    result_7 = np.mean([[2, 3, 4],
                        [3, 4, 5],
                        [4, 5, 6]])
    result_8 = np.mean([[3, 4],
                        [4, 5],
                        [5, 6]])
    result_9 = np.mean([[2, 3],
                        [3, 4],
                        [4, 5]])
    result_10 = np.mean([[2, 3, 4],
                         [3, 4, 5],
                         [4, 5, 6]])
    result_11 = np.mean([[3, 4, 5],
                         [4, 5, 6],
                         [5, 6, 7]])
    result_12 = np.mean([[4, 5],
                         [5, 6],
                         [6, 7]])

    result_13 = np.mean([[3, 4],
                         [4, 5]])
    result_14 = np.mean([[3, 4, 5],
                         [4, 5, 6]])
    result_15 = np.mean([[4, 5, 6],
                         [5, 6, 7]])
    result_16 = np.mean([[5, 6],
                         [6, 7]])

    expected_result = np.array([[result_1, result_2, result_3, result_4],
                                [result_5, result_6, result_7, result_8],
                                [result_9, result_10, result_11, result_12],
                                [result_13, result_14, result_15, result_16]])
    result = tu.rolling_window_mean_numpy(array_test, np.empty_like(array_test).astype(np.float32), 1, 1)
    assert_array_almost_equal(result, expected_result)


def test_rolling_tpi():
    tu = SgpHelbig()

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

    expected_result = array_test - np.array([[result_1, result_2, result_3, result_4],
                                             [result_5, result_6, result_7, result_8],
                                             [result_9, result_10, result_11, result_12],
                                             [result_13, result_14, result_15, result_16]])
    result_1 = tu.tpi_map(array_test, 1, 1, library="numba")
    result_2 = tu.tpi_map(array_test, 1, 1, library="numpy")

    assert_array_almost_equal(result_1, expected_result)
    assert_array_almost_equal(result_2, expected_result)


def test_rolling_tpi_idx():
    tu = SgpHelbig()

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

