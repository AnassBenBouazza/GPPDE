import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d(x, t, u, cmap="viridis", xlabel="x", ylabel="t", zlabel="u", angle=225) :
    """Plots u as a function of x and t

    Args:
        x (1D ndarray) : x_axis values
        t (1D ndarray) : y_axis values
        u (2D ndarray) : z_axis values for x and t    
    """
    x, t = np.meshgrid(x, t)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot_surface(x, t, u.T, cmap=cmap)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.view_init(30, angle)

    plt.show()