# maths
import numpy as np
from scipy import stats
from sklearn import metrics as skm
from numpy.lib import function_base as np_fb

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
    return (coefs, ind)


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


def auto_bin(arr):
    """Determine optimal number of bins for a given array

    Parameters
    ----------
    arr : ndarray
        The input array

    Returns
    -------
    int
        Approximation of the optimal number of bins
    """
    return int(np.round(np.ptp(arr) / np_fb._hist_bin_auto(arr), 0))


def mutual_info(arr1, arr2, normalized=1):
    """Determine similarity of two datasets

    The mutual information score estimates the difference between two distributions
    for a particular set of bins.
    The higher score the score the bigger the difference between the datasets

    Parameters
    ----------
    arr1 : array-like
        The first dataset

    arr2 : array-like
        The second dataset

    normalized : int [0, 2]
        0 -> no normalized mi score
        1 -> normalized mi score
        2 -> both mi scores

    Returns
    -------
    float
        The normalized mutual information score

    tuple
        The mutual information score and the normalized mutual info
    """
    def _mi_score():
        # determine the optimal number of bins
        c_arr = np.histogram2d(arr1, arr2, bins=auto_bin(arr1))[0]
        return skm.mutual_info_score(None, None, contingency=c_arr)

    if normalized == 1:
        return _mi_score()
    elif normalized == 0:
        return skm.normalized_mutual_info_score(arr1, arr2)
    elif normalized == 2:
        return (_mi_score(), skm.normalized_mutual_info_score(arr1, arr2))


def KL_div(arr1, arr2, eps=0.00001):
    """
    """
    def _normalize_t1(bins_arr, arr, eps=0):
        return (bins_arr / arr.size) + eps

    def _calc_kl(arr1, arr2):
        return np.sum(arr1*np.log(arr1 / arr2))

    def _bin_similarity(arr1, arr2):
        hist_arr1, bins_arr1 = np.histogram(arr1, bins=auto_bin(arr1))
        hist_arr2 = np.histogram(arr2, bins=bins_arr1)[0]
        return hist_arr1, hist_arr2

    hist_arr1, hist_arr2 = _bin_similarity(arr1, arr2)

    nrm_arr1 = _normalize_t1(hist_arr1, arr1, eps)
    nrm_arr2 = _normalize_t1(hist_arr2, arr2, eps)

    sc1 = _calc_kl(nrm_arr1, nrm_arr2)

    hist_arr2, hist_arr1 = _bin_similarity(arr2, arr1)

    nrm_arr1 = _normalize_t1(hist_arr1, arr1, eps)
    nrm_arr2 = _normalize_t1(hist_arr2, arr2, eps)

    sc2 = _calc_kl(nrm_arr2, nrm_arr1)
    # The mean of the two KL divergence scores
    return (sc1 + sc2) / 2

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
