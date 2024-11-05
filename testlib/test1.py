import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

import pclib
import pytest
import numpy as np


def test_sampling():

    """
    test the creation and functionality of the object
    SamplingModule
    """

    sm = pclib.SamplingModule("default", 10)

    assert len(sm) == 9, "Size of the sampling module " + \
        f"is not correct {len(sm)}"

    sm.reset()
    sm()
    y1 = sm.get_idx()
    sm()
    y2 = sm.get_idx()

    assert y1 != y2, "Two consecutive calls should " + \
        "yield different results"

    sm()
    assert sm.get_counter() == 3, "Counter isn't working correctly"



def test_leaky1D():

    """
    test the LeakyVariable1D class
    """

    eq = 0.
    tau = 2.

    # definition
    lv = pclib.LeakyVariable1D("TST", eq, tau)
    assert lv.get_v() == eq, f"Initial value is not " + \
        f"correct v={lv.get_v()}"

    # dynamics | v = v + (eq - v) / tau + x
    x = 1.
    lv(x)
    assert lv.get_v() == x, f"Value after first call is not " + \
        f"correct v={lv.get_v()}"

    lv()
    assert lv.get_v() == x + (eq - x) / tau, f"Value after " + \
        f"second call is not correct v={lv.get_v()}"


def test_leakyND():

    """
    test the LeakyVariableND class
    """

    eq = 0.
    tau = 2.
    ndim = 3

    # definition
    lv = pclib.LeakyVariableND("TST", eq, tau, ndim)

    # check dimensions
    assert len(lv) == ndim, f"Dimension of the leaky variable " + \
        f"is not correct {len(lv)}"

    # check initialization
    assert lv.get_v()[0] == eq, f"Initial value is not " + \
        f"correct v={lv.get_v()}"

    # dynamics | v = v + (eq - v) / tau + x
    x = np.array([1., 0.4, 0.5]).reshape(-1)
    lv(x)
    assert lv.get_v()[0] == x[0], f"Value after first call is " +\
        f"not correct v={lv.get_v()}"

    lv(np.zeros(ndim))
    assert lv.get_v()[0] == (x + (eq - x) / tau)[0], f"Value" + \
        f" after second call is not correct v={lv.get_v()}"




if __name__ == "__main__":
    test_leakyND()

