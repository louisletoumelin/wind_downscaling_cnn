import numpy as np
from numpy.testing import *
import warnings
import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)

from downscale.Operators.Micro_Met import MicroMet


@pytest.fixture
def mnt():
    mnt = np.load("../Data_test/mnt_small.npy")
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
    result = m.terrain_slope_idx(mnt, 25, idx_x, idx_y)
    return result


@pytest.fixture
def result_terrain_slope_azimuth_map(mnt, m):
    result = m.terrain_slope_azimuth_map(mnt, 25)
    return result


@pytest.fixture
def result_terrain_slope_azimuth_idx(mnt, m, idx_x, idx_y):
    result = m.terrain_slope_azimuth_idx(mnt, 25, idx_x, idx_y)
    return result


@pytest.fixture
def result_curvature_map_scale_true(mnt, m):
    result = m.curvature_map(mnt, scale=True)
    return result


@pytest.fixture
def result_curvature_map_scale_false(mnt, m):
    result = m.curvature_map(mnt, scale=False)
    return result

@pytest.fixture
def result_curvature_idx_scale_true_safe(mnt, m, idx_x, idx_y):
    result = m.curvature_idx(mnt, idx_x, idx_y, scale=True, method="safe")
    return result

@pytest.fixture
def result_curvature_idx_scale_false_safe(mnt, m, idx_x, idx_y):
    result = m.curvature_idx(mnt, idx_x, idx_y, scale=False, method="safe")
    return result

@pytest.fixture
def result_curvature_idx_scale_true_other(mnt, m, idx_x, idx_y):
    result = m.curvature_idx(mnt, idx_x, idx_y, scale=True, method="other")
    return result

@pytest.fixture
def result_curvature_idx_scale_false_other(mnt, m, idx_x, idx_y):
    result = m.curvature_idx(mnt, idx_x, idx_y, scale=False, method="other")
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

