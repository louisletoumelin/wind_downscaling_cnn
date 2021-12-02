import numpy as np
from numpy.testing import *
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from downscale.operators.topo_utils import *
from downscale.operators.helbig import *


def test_mu_helbig_map():
    """test passes 30/11/2021"""

    tu = SgpHelbig()

    array_test_1 = np.array([[1, 2],
                             [3, 4]])

    expected_result_1 = np.sqrt(np.array([[2.5, 2.5],
                                          [2.5, 2.5]]))

    array_test_2 = np.array([[1, 3, 5, 7],
                             [2, 4, 6, 8],
                             [9, 11, 13, 15],
                             [10, 12, 14, 16]])

    expected_result_2 = np.array([[1.5811388, 1.5811388, 1.5811388, 1.5811388],
                                  [3.1622777, 3.1622777, 3.1622777, 3.1622777],
                                  [3.1622777, 3.1622777, 3.1622777, 3.1622777],
                                  [1.5811388, 1.5811388, 1.5811388, 1.5811388]], dtype=np.float32)

    result_1 = tu.mu_helbig_map(array_test_1, 1, verbose=False)
    result_2 = tu.mu_helbig_map(array_test_2, 1, verbose=False)

    assert_array_almost_equal(result_1, expected_result_1)
    assert_array_almost_equal(result_2, expected_result_2)


def test_mu_helbig_map_idx():
    """test passes 30/11/2021"""
    tu = SgpHelbig()
    array_test = np.array([[1, 3, 5, 7],
                           [2, 4, 6, 8],
                           [9, 11, 13, 15],
                           [10, 12, 14, 16]])
    expected_result = np.array([[1.5811388, 1.5811388, 1.5811388, 1.5811388],
                                [3.1622777, 3.1622777, 3.1622777, 3.1622777],
                                [3.1622777, 3.1622777, 3.1622777, 3.1622777],
                                [1.5811388, 1.5811388, 1.5811388, 1.5811388]], dtype=np.float32)
    result = tu.mu_helbig_idx(array_test, 1, np.array([1]), np.array([2]))
    assert_almost_equal(result, expected_result[2, 1])


def test_mu_helbig_average():
    """test passes 01/12/2021"""
    tu = SgpHelbig()

    array_test_2 = np.array([[1, 3, 5, 7],
                             [2, 4, 6, 8],
                             [9, 11, 13, 15],
                             [10, 12, 14, 16]])
    expected_result_2 = np.array([[2.37170825, 2.37170825, 2.37170825, 2.37170825],
                                  [2.63231, 2.63231, 2.63231, 2.63231],
                                  [2.63231, 2.63231, 2.63231, 2.63231],
                                  [2.37170825, 2.37170825, 2.37170825, 2.37170825]], dtype=np.float32)

    array_test_3 = np.random.randint(0, 20, (100, 200))

    result_3 = tu.mu_helbig_average(array_test_2, 1, x_win=1, y_win=1, library="numba", verbose=False)
    result_4 = tu.mu_helbig_average(array_test_2, 1, x_win=1, y_win=1, library="tensorflow", verbose=False)
    result_5 = tu.mu_helbig_average(array_test_2, 1, x_win=1, y_win=1, library="numpy", verbose=False)
    result_6 = tu.mu_helbig_average(array_test_3, 1, idx_x=50, idx_y=50, x_win=1, y_win=1, verbose=False)
    result_7 = tu.mu_helbig_average(array_test_3, 1, x_win=1, y_win=1, verbose=False)[50, 50]

    assert_allclose(result_3[1:-1, 1:-1], expected_result_2[1:-1, 1:-1], rtol=1e-03, atol=0.001)
    assert_allclose(result_4[1:-1, 1:-1], expected_result_2[1:-1, 1:-1], rtol=1e-03, atol=0.001)
    assert_allclose(result_5[1:-1, 1:-1], expected_result_2[1:-1, 1:-1], rtol=1e-03, atol=0.001)
    assert_allclose(result_5[1:-1, 1:-1], expected_result_2[1:-1, 1:-1], rtol=1e-03, atol=0.001)
    assert_allclose(result_6, result_7, rtol=1e-03, atol=0.001)


def test_xsi_helbig_map():
    """test passes 02/12/2021"""
    tu = SgpHelbig()

    array_test_2 = np.array([[1, 3, 5, 7],
                             [2, 4, 6, 8],
                             [9, 11, 13, 15],
                             [10, 12, 14, 16]])

    std_expected = np.array([[1.118033, 1.707825, 1.707825, 1.118033],
                             [3.696845, 3.915780, 3.915780, 3.696845],
                             [3.696845, 3.915780, 3.915780, 3.696845],
                             [1.118033, 1.707825, 1.707825, 1.118033]])

    mu_expected = np.array([[2.37170825, 2.37170825, 2.37170825, 2.37170825],
                            [2.63231, 2.63231, 2.63231, 2.63231],
                            [2.63231, 2.63231, 2.63231, 2.63231],
                            [2.37170825, 2.37170825, 2.37170825, 2.37170825]], dtype=np.float32)

    array_test_3 = np.random.randint(0, 20, (100, 200))
    mu_3 = tu.mu_helbig_average(array_test_3, 1, idx_x=50, idx_y=50, x_win=1, y_win=1)

    result_expected_2 = np.sqrt(2) * std_expected / mu_expected
    mu = tu.mu_helbig_average(array_test_2, 1, x_win=1, y_win=1, library="tensorflow")
    result_2 = tu.xsi_helbig_map(array_test_2, mu, x_win=1, y_win=1, library="numba")
    result_3 = tu.xsi_helbig_map(array_test_2, mu, x_win=1, y_win=1, library="tensorflow")
    result_4 = tu.xsi_helbig_map(array_test_2, mu, x_win=1, y_win=1, library="numpy")

    result_5 = tu.xsi_helbig_map(array_test_3, mu_3, x_win=1, y_win=1, library="tensorflow")[50, 50]
    result_6 = tu.xsi_helbig_map(array_test_3, mu_3, x_win=1, y_win=1, library="numba")[50, 50]
    result_7 = tu.xsi_helbig_map(array_test_3, mu_3, x_win=1, y_win=1, library="numpy")[50, 50]
    result_8 = tu.xsi_helbig_map(array_test_3, mu_3, idx_x=50, idx_y=50, x_win=1, y_win=1)

    assert_allclose(result_2[1:-1, 1:-1], result_expected_2[1:-1, 1:-1], rtol=1e-03, atol=0.001)
    assert_allclose(result_2[1:-1, 1:-1], result_3[1:-1, 1:-1], rtol=1e-03, atol=0.001)
    assert_allclose(result_2[1:-1, 1:-1], result_4[1:-1, 1:-1], rtol=1e-03, atol=0.001)

    assert_allclose(result_5, result_8, rtol=1e-03, atol=0.001)
    assert_allclose(result_6, result_8, rtol=1e-03, atol=0.001)
    assert_allclose(result_7, result_8, rtol=1e-03, atol=0.001)


def test_x_sgp_topo_helbig_idx_shape():
    """test passes 02/12/2021"""
    tu = SgpHelbig()

    array_test = np.array([[1, 2, 3, 4],
                           [2, 3, 8, 5],
                           [3, 90, 5, 6],
                           [4, 5, 6, 7]])

    result = tu.x_sgp_topo_helbig_idx(array_test, dx=1, L=4, library="tensorflow", x_win=1, y_win=1)
    result_1 = tu.x_sgp_topo_helbig_idx(array_test, dx=1, L=4, library="numpy", x_win=1, y_win=1)
    result_2 = tu.x_sgp_topo_helbig_idx(array_test, dx=1, L=4, library="numba", x_win=1, y_win=1)

    assert array_test.shape == result.shape
    assert array_test.shape == result_1.shape
    assert array_test.shape == result_2.shape


def test_x_sgp_topo_helbig_idx():
    tu = SgpHelbig()

    array_test = np.random.randint(0, 20, (100, 200))

    result_1 = tu.x_sgp_topo_helbig_idx(array_test, dx=1, L=4, x_win=1, y_win=1)
    result_2 = tu.x_sgp_topo_helbig_idx(array_test, dx=1, L=4, idx_x=50, idx_y=50, x_win=1, y_win=1)

    assert result_1.shape == array_test.shape

    assert_allclose(result_1[50, 50], result_2, rtol=1e-03, atol=0.001)


def test_downscaling():
    d = DwnscHelbig()
    array_test = np.random.randint(0, 20, (100, 200))

    result_1 = d.x_dsc_topo_helbig(array_test, dx=1, library="numba")[50, 50]
    result_2 = d.x_dsc_topo_helbig(array_test, dx=1, idx_x=50, idx_y=50, library="numba")

    assert result_1 == result_2
