import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

import pclib
import pytest
import numpy as np
import matplotlib.pyplot as plt




def activity_over_grid(model):

    # make a list of points over a square 0,1 x 0,1
    n = 100
    x = np.linspace(0., 1., n)
    y = np.linspace(0., 1., n)
    grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    # compute the activity of the model over the grid
    activity = []
    for i in range(len(grid)):
        activity.append(model.fwd_ext(grid[i]).max())

    # plot the activity
    print(f"Len={len(pcnn)}")
    activity = np.array(activity).reshape(n, n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(activity.T, cmap='hot', interpolation='nearest',
               vmin=0., vmax=1.)
    ax1.set_title(f"Length {len(pcnn)}," + \
        f" max={activity.max():.3f}min={activity.min():.3f}")

    ax2.imshow(pcnn.get_wff(), cmap='hot', interpolation='nearest')
    ax2.set_title("Weights")

    plt.show()



if __name__ == "__main__":

    pclib.set_debug(False)

    n = 10
    Ni = 10
    sigma = 0.09
    bounds = np.array([0., 1., 0., 1.])
    xfilter = pclib.PCLayer(n, sigma, bounds)

    # definition
    pcnn = pclib.PCNN(N=Ni, Nj=n**2, gain=3., offset=1.5,
                      clip_min=0.09, threshold=0.5,
                      rep_threshold=0.5, rec_threshold=0.01,
                      num_neighbors=8, trace_tau=0.1,
                      xfilter=xfilter, name="2D")

    print(f"Len={len(pcnn)}")
    print(f"Size={pcnn.get_size()}")

    # learn position 1
    x = np.array([0.4, 0.5])
    _ = pcnn(x)

    # learn position 2
    x = np.array([0.6, 0.5])
    _ = pcnn(x)

    activity_over_grid(pcnn)

