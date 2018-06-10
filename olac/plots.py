# plots
import matplotlib.pyplot as plt

from . import maths as mf
########################################################################################################################
#                                                    Learning at Cost                                                  #
########################################################################################################################


def plot_linear_ls(x, y, window_size=10, constant=True, colour='r', label=None):
    """Plot the linear fits for each period

    Parameters
    ----------
    x : ndarray
        The independent variables, features, over which to fit
    y : ndarray
        The dependent variable which we want approximate
    window_size : int (optional)
        The number of observations that will included in a period
    constant : boolean (optional)
        Whether a constant should be added, default is True
    colour : str
        The colour paramter to be passed to pyplot
    label : str
        The label parameter to be passed to pyplot

    Returns
    -------
    coefs : ndarray
        An numpy array containing the coefficients.
    ind : list
        The indexes of the periods
    """
    coefs, ind = mf.seq_linear_ls(x=x, y=y, window_size=window_size, constant=constant)
    for i in range(len(ind)):
        xi = ind[i]
        alpha, beta = coefs[i]
        plt.plot(xi, alpha + beta * xi, c=colour, label=label)
    return coefs, ind
