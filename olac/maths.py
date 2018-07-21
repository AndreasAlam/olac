# maths
import numpy as np
from scipy import stats
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
    print('!!! NOT DONE !!!')
    return (mu, sigma, third_moment, fourth_moment, median)


def _bin_similarity(p, q):
    """Calculate the hist count for two arrays taken the bins chosen for the first array
       as the bins for the second array.

    Parameters
    ----------
    p : ndarray
        The first array to compare to the second
    q : ndarray
        The second array to compare to the first

    Returns
    -------
    (hcnt_p, hcnt_q) : tuple(ndarray, ndarray)
        The hist counts with optional bins for array 1
    """
    hcnt_p, bins_p = np.histogram(p, bins=auto_bin(p))
    hcnt_q = np.histogram(q, bins=bins_p)[0]
    return hcnt_p, hcnt_q


def _normalize_t1(hcnt_arr, arr_size, eps):
    """Normalise the array such that is sums to one. Epsilon is added to prevent zero values

    Parameters
    ----------
    hcnt_arr : ndarray
        the count of values for a bin, histogram count
    arr_size : int/float
        The number of data points in arr
    eps : float
        Correction factor to prevent null values

   Returns
   -------
   ndarray
       The normalized histogram count
    """
    return (hcnt_arr / arr_size) + eps


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
    len_uniq = np.unique(arr).size
    bin_cnt = int(np.round(np.ptp(arr) / np_fb._hist_bin_auto(arr), 0))
    if bin_cnt > len_uniq:
        return len_uniq
    else:
        return bin_cnt


def kl_div(p, q, eps=0.00001, full_outp=False):
    """ Calculate the Kullback–Leibler divergence score.

    The function calculates the average KL divergence score taking each array as base at a time.

    Parameters
    ----------
    p : ndarray
        The first array to compare to the second
    q : ndarray
        The second array to compare to the first
    eps : float (optional)
        Correction factor to prevent null values

    Returns
    -------
    float
        KL div score

    Reference
    ---------
    [1] https://en.wikipedia.org/wiki/Kullback–Leibler_divergence
    """
    # Bins that are optimal from perspective of the first array
    hcnt_p, hcnt_q = _bin_similarity(p, q)

    # KL expects counts that sum to one, i.e. for distribitions
    nrm_p = _normalize_t1(hcnt_p, p.size, eps)
    nrm_q = _normalize_t1(hcnt_q, q.size, eps)

    # The KL score
    return np.sum(nrm_p * np.log2(nrm_p / nrm_q))


def hellinger_dist(p, q, eps=0.00001):
    """ Calculate the hellinger_dist for concrete distributions

    Parameters
    ----------
    p : ndarray
        The first array to compare to the second
    q : ndarray
        The second array to compare to the first
    eps : float (optional)
        Correction factor to prevent null values

    Returns
    -------
    float
        hellinger_dist

    Reference
    ---------
    [1] https://en.wikipedia.org/wiki/Hellinger_distance
    """
    # Bins that are optimal from perspective of the first array
    hcnt_p, hcnt_q = _bin_similarity(p, q)

    # KL expects counts that sum to one, i.e. for distribitions
    nrm_p = _normalize_t1(hcnt_p, p.size, eps)
    nrm_q = _normalize_t1(hcnt_q, q.size, eps)

    return np.sqrt(np.sum((np.sqrt(nrm_p) - np.sqrt(nrm_q)) ** 2)) / np.sqrt(2.0)
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
