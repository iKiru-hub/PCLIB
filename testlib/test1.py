import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

import pclib
import pytest
import numpy as np


def test_sampling():

    """
    test the creation and functionality of the object
    ActionSampling2D
    """

    sm = pclib.ActionSampling2D("default", 10)

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


def test_densitymod():

    x, w = np.random.randn(2, 5)
    mod = pclib.DensityMod(weights=w,
                           theta=0.1)
    y = mod(x=x)

    assert isinstance(y, float), f"Output of the density " + \
        f"modulator is not correct {type(y)}"


def test_randlayer():

    """
    test the random input layer
    """

    N = 5

    # defintion
    layer = pclib.RandLayer(N)

    # check dimensions of the matrix
    m = layer.get_centers()
    assert (N, 2) == m.shape, f"Dimension of the layer is not " + \
        f"correct {m.shape} expected {(N, 2)}"

    # check call
    x = np.array([0.5, 0.5])
    y = layer(x)

    assert len(y) == N, f"Output of the layer is not" + \
        f" correct {len(y)}"


def test_pclayer():

    """
    test the layer of hard-coded place cells
    """

    n = 3
    sigma = 0.1
    bounds = np.array([0., 1., 0., 1.])

    # defintion
    layer = pclib.PCLayer(n, sigma, bounds)

    # check dimensions
    assert n**2 == len(layer), f"Dimension of the layer is not " + \
        f"correct {len(layer)}"

    # check call
    x = np.array([0.5, 0.5])
    y = layer(x)

    assert len(y) == n**2, f"Output of the layer is not" + \
        f" correct {len(y)}"


def test_gridlayer():

    n = 3
    sigma = 0.1
    speed = 0.1
    init_bounds = [-1, 1, -1, 1]

    # defintion
    layer = pclib.GridLayer(n**2, sigma, speed, init_bounds)

    # check dimensions
    assert n**2 == len(layer), f"Dimension of the layer" + \
        f"correct {len(layer)}"
    assert layer.get_centers().shape == (n**2, 2), \
        f"Dimension of the layer is not correct " + \
        f"{layer.get_centers().shape} expected {(n**2, 2)}"

    # check call
    v = np.array([0.5, 0.5])
    layer(v)


def test_pcnn_basics():

    """
    test the PCNN network model initialization and call
    """

    n = 3
    Ni = 10
    sigma = 0.1
    bounds = np.array([0., 1., 0., 1.])
    xfilter = pclib.PCLayer(n, sigma, bounds)
    # xfilter = pclib.RandLayer(int(n**2))

    # definition
    pcnn = pclib.PCNN(N=Ni, Nj=n**2, gain=0.1, offset=0.1,
                      clip_min=0.01, threshold=0.1,
                      rep_threshold=0.2,
                      rec_threshold=0.1,
                      num_neighbors=8, trace_tau=0.1,
                      xfilter=xfilter, name="2D")
    assert pcnn.get_size() == Ni, f"Number of neurons is not " + \
        f"correct {len(pcnn)}"

    # check call
    x = np.array([0.5, 0.5])
    y = pcnn(x)
    assert len(y) == Ni, f"Output of the network is not" + \
        f" correct {len(y)}"


def test_pcnn_plasticity():

    """
    test the PCNN network model learning
    """

    n = 12
    Ni = 10
    sigma = 0.04
    bounds = np.array([0., 1., 0., 1.])
    xfilter = pclib.PCLayer(n, sigma, bounds)
    # xfilter = pclib.RandLayer(int(n**2))

    # definition
    pcnn = pclib.PCNN(N=Ni, Nj=n**2, gain=3., offset=1.,
                      clip_min=0.09, threshold=0.5,
                      rep_threshold=0.7,
                      rec_threshold=0.0,
                      num_neighbors=8, trace_tau=0.1,
                      xfilter=xfilter, name="2D")

    # check learning
    x = np.array([0.5, 0.5])
    _ = pcnn(x)
    pcnn.update()
    assert len(pcnn) == 1, f"Wrong number of learned pc, " + \
        f"given {len(pcnn)} expected 1"

    x = np.array([0.45, 0.6])
    _ = pcnn(x)
    pcnn.update()

    assert len(pcnn) == 2, f"Wrong number of learned pc, " + \
        f"given {len(pcnn)} expected 2\n{pcnn.get_wff().sum(axis=1)}"

    # check recurrent connectivity
    connectivity = pcnn.get_connectivity()
    assert connectivity.sum() == 2, f"Recurrent connectivity" + \
        f" is not correct {connectivity.sum()}"


def test_two_layer_network():

    """
    test the two layer network model with
    input 5, hidden 2 and output 1
    """

    Wh = np.random.randn(5, 2).tolist()
    Wo = np.random.randn(2).tolist()

    model = pclib.TwoLayerNetwork(Wh, Wo)

    x = np.random.randn(5).tolist()
    y, h = model(x)

    assert type(y) == float, f"Output of the network is not " + \
        f"correct {type(y)}"
    assert len(h) == 2, f"Hidden layer output is not correct " + \
        f"{len(h)}"


def test_one_layer_network():

    """
    test the one layer network model with
    input 5, hidden 2 and output 1
    """

    Wh = np.random.randn(5).tolist()

    model = pclib.OneLayerNetwork(Wh)

    x = np.random.randn(5).tolist()
    y, h = model(x)
    wh = model.get_weights()

    assert type(y) == float, f"Output of the network is not " + \
        f"correct {type(y)}"
    assert len(h) == 5, f"Hidden layer output is not correct " + \
        f"{len(h)}"
    assert len(wh) == 5, f"Hidden layer weights are not " \
        f"correct {len(wh)}"


def test_hexagon():

    """
    test the hexagonal lattice
    """

    hex = pclib.Hexagon()



if __name__ == "__main__":
    test_leakyND()

