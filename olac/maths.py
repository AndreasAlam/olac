# maths
import numpy as np
import scipy.stats as stats

from . import vis as vf
from . import utils as uf
########################################################################################################################
#                                                    Learning at Cost                                                  #
########################################################################################################################


def linear_ls(x, y, constant=True):
    """Fast linear least squares

    Parameters
    ----------
    x : ndarray
        The independent variables, features, over which to fit
    y : ndarray
        The dependent variable which we want approximate
    constant : boolean (optional)
        Whether a constant should be added, default is True

    Returns:
    np.matrix
        The coefficients that minimise the square error
    """
    y = uf.dim_correct(y)
    if constant:
        x = uf.dim_correct(x)
        X = np.hstack((np.ones(shape=(x.shape[0], 1)), x))
    else:
        X = uf.dim_correct(x)
    return np.linalg.inv(X.T @ X) @ X.T @ y


def seq_linear_ls(x, y, window_size=10, constant=True, axis=0):
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
    axis : int
        The axis over which to split the input array

    Returns
    -------
    coefs : ndarray
        An numpy array containing the coefficients.
    ind : list
        The indexes of the periods
    """
    y = uf.dim_correct(y)
    if constant:
        x = uf.dim_correct(x)
        X = np.hstack((np.ones(shape=(x.shape[0], 1)), x))
    else:
        X = uf.dim_correct(x)

    n, m = X.shape
    periods = int(n / window_size)

    # Initialise the coefficients array
    coefs = np.ones((periods, m))
    # Create a list containing the index blocks for each period
    ind = np.array_split(np.arange(0, n, 1), periods, axis)
    for i, period in enumerate(ind):
        coefs[i, :] = linear_ls(X[period, :], y[period], constant=False).reshape(-1)
    return coefs, ind


def dist_coefs(coefs):
    """Describe the distribution of the coefficients

    Parameters
    ----------
    coefs : ndarray
        Numpy array containing the coefficients of the linear least squares fits

    Returns
    -------

    """
    n, m = coefs.shape
    mu = np.mean(coefs, axis=0)
    sigma = np.std(coefs, axis=0)
    median = np.median(coefs, axis=0)
    third_moment = stats.skew(coefs, axis=0)
    # Fisher's definition substracts 3 from the result to give 0.0 for a normal distribution
    fourth_moment = stats.kurtosis(coefs, axis=0, fisher=True)
    return (mu, sigma, third_moment, fourth_moment, median)


def dist_seq_lls():
    print("empty")


########################################################################################################################
#                                                   Data Generation                                                    #
########################################################################################################################


def rotation_matrix(theta: float):
    """
    Generates a rotation matrix to rotate a 2d vector by theta.

    Parameters
    ----------
    theta : float

    Returns
    -------
    The 2x2 rotation matrix

    """
    return np.array([[np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]])
