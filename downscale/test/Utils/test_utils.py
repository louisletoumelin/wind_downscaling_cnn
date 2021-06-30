import numpy as np

from downscale.Utils.Utils import *


def test_lists_to_arrays_if_required():
    a, b = lists_to_arrays_if_required([[1, 2, 3], np.array([1, 2, 4])])
    d = lists_to_arrays_if_required([1, 2, 3])
    e = lists_to_arrays_if_required(np.array([1, 2, 3]))
    assert isinstance(a, np.ndarray)
    assert isinstance(b, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(e, np.ndarray)
