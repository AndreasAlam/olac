# general
import os
import sys
import pandas as pd

import scipy.stats
import numpy as np


def dim_correct(x, axis=1):
    """Check and dimensionality of the input array and add one if number of dimensions is zero.

    Parameters
    ----------
    x : ndarray
        The array to be validated and or corrected
    axis : int [0, 1] (Optional)
        The axis to which to add the dimension, 0 results in a row vector/array
        1 results in a columns vector/array

    Returns
    -------
    x : ndarray
        The original input array if the dimension were already greather than 0
        otherwise with a second dimension of 1 added.
    """
    if x.ndim == 1:
        if axis == 1:
            return x[:, None]
        else:
            return x[None, :]
    else:
        return x


def rotation_matrix(theta: float):
    """
    Generates a rotation matrix to rotate a 2d vector by theta.

    Parameters
    ----------
    theta : float

    Returns
    -------
    numpy.ndarray : The 2x2 rotation matrix
<<<<<<< HEAD

    """
    import warnings
    from . import maths as mf
    warnings.warn("The 'utils.rotation_matrix' method is deprecated, "
                  "use maths.rotation_matrix(theta: float) instead")
    return mf.rotation_matrix(theta)


def unit_circle_points(n):
    """
    Create `n` points evenly spaced around the unit circle.

    Parameters
    ----------
    n : int
        The number of points

    Returns
    -------
    numpy.ndarray : the generated points, shape (n, 2)
=======

    """
    points = np.zeros(shape=(n, 2))
    points[0, 0] = 1  # first point at (1, 0)
    rot = rotation_matrix(2 * np.pi / n)

    for i in range(1, n):
        # rotate by a fixed amount from the previous point
        points[i, :] = points[i-1, :] @ rot

    return points


def slide_probability_over_list(num_iterations, num_indices,
                                transition_rate=0.1, var=1):
    """
    Slide a gaussian probability density function over list indices for drawing
    numbers from a list.

    The function numpy.random.choice allows for the random selection of items
    from a list. Its `p` parameter allows to set a different probability for
    each item. This function will generate a series of `p` arrays, such that
    successive draws will see the probability for the elements differently
    weighted. The weighting is done using a gaussian probability density
    function which slides over the list positions. In successive iterations,
    the probability shifts along the list with the rate given by
    `transition_rate`.

    Parameters
    ----------
    num_iterations : int
        The number of probability distributions to generate. Set
        to zero to generate infinitely.

    num_indices : int
        Length of the list for choosing from

    transition_rate : float
        Speed of sliding the gaussian. A rate of 1 implies that
        the mean of the gaussian will shift by one index per iteration

    var : float
        The variance of the gaussian

    Returns
    -------
    generator : A sequence of probability arrays

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> # low variance to guarantee result
    >>> for p in slide_probability_over_list(6, 3, 1, 0.1):
    ...    np.random.choice([1,2,3], p=p)
    1
    2
    3
    1
    2
    3

    """
    loc = 0  # initial location of gaussian

    # make a long list to catch the tails of the distribution that would
    # otherwise go past the edges of the list
    base = np.arange(-num_indices, 2 * num_indices)

    # go forever if num_indices = 0
    def cond(i):
        if num_iterations:
            return i < num_iterations
        else:
            return True

    i = 0
    while cond(i):
        p = scipy.stats.norm.pdf(base, loc=loc % num_indices, scale=var)
        p = p / p.sum()  # normalize the distribution

        # wrap the tail of the distribution around the edges of the list
        p_wrapped = p[:num_indices] \
                    + p[num_indices:2 * num_indices] \
                    + p[2 * num_indices:]

        yield p_wrapped

        loc += transition_rate
        i += 1


def unit_circle_points(n):
    """
    Create `n` points evenly spaced around the unit circle.

    Parameters
    ----------
    n : int
        The number of points

    Returns
    -------
    numpy.ndarray : the generated points, shape (n, 2)

    """
    points = np.zeros(shape=(n, 2))
    points[0, 0] = 1  # first point at (1, 0)
    rot = rotation_matrix(2 * np.pi / n)

    for i in range(1, n):
        # rotate by a fixed amount from the previous point
        points[i, :] = points[i-1, :] @ rot

    return points


def slide_probability_over_list(num_iterations, num_indices,
                                transition_rate=0.1, var=1):
    """
    Slide a gaussian probability density function over list indices for drawing
    numbers from a list.

    The function numpy.random.choice allows for the random selection of items
    from a list. Its `p` parameter allows to set a different probability for
    each item. This function will generate a series of `p` arrays, such that
    successive draws will see the probability for the elements differently
    weighted. The weighting is done using a gaussian probability density
    function which slides over the list positions. In successive iterations,
    the probability shifts along the list with the rate given by
    `transition_rate`.

    Parameters
    ----------
    num_iterations : int
        The number of probability distributions to generate. Set
        to zero to generate infinitely.

    num_indices : int
        Length of the list for choosing from

    transition_rate : float
        Speed of sliding the gaussian. A rate of 1 implies that
        the mean of the gaussian will shift by one index per iteration

    var : float
        The variance of the gaussian

    Returns
    -------
    generator : A sequence of probability arrays

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> # low variance to guarantee result
    >>> for p in slide_probability_over_list(6, 3, 1, 0.1):
    ...    np.random.choice([1,2,3], p=p)
    1
    2
    3
    1
    2
    3

    """
    loc = 0  # initial location of gaussian

    # make a long list to catch the tails of the distribution that would
    # otherwise go past the edges of the list
    base = np.arange(-num_indices, 2 * num_indices)

    # go forever if num_indices = 0
    def cond(i):
        if num_iterations:
            return i < num_iterations
        else:
            return True

    i = 0
    while cond(i):
        p = scipy.stats.norm.pdf(base, loc=loc % num_indices, scale=var)
        p = p / p.sum()  # normalize the distribution

        # wrap the tail of the distribution around the edges of the list
        p_wrapped = p[:num_indices] \
                    + p[num_indices:2 * num_indices] \
                    + p[2 * num_indices:]

        yield p_wrapped

        loc += transition_rate
        i += 1


def set_path(level=1, change_path=True):
    """
    Set the path to the olac/olac directory or insert it in the path.

    Parameters
    ----------
    level : int
        The number of levels you are removed from the olac parent
        directory. Maximum of two levels are implemented.

    change_path : boolean
        True - Change the working directory
        False - Inset the path to the python path

    Returns
    -------
    None : nonetype
        The function does not return anything
    """
    if level == 1:
        base = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    elif level == 2:
        base = os.path.normpath(
            os.path.normpath(os.getcwd() + os.sep + os.pardir)
            + os.sep + os.pardir + '/olac'
        )
    else:
        raise Exception('Level {0} is not a valid value'.format(level))
    if change_path:
        os.chdir(base)
    else:
        sys.path.insert(0, base)
    return(None)


def data_prep(X):
    """
    Prepare data points for input in keras model. For now it just scales.

    Parameters
    ----------
    X : ndarray
        Input points
    """
    return X/np.max(np.abs(X))


def queue_point_list_to_df(qp_list):
    """
    Convert a list of QueuePoints to a pandas DataFrame

    Parameters
    ----------
    qp_list : list[QueuePoint]
        The list of QueuePoints to convert

    Returns
    -------
    pandas.DataFrame: A dataframe containing all the QueuePoint information

    """

    # pivot the list
    zipped = list(zip(*[p.to_tuple() for p in qp_list]))

    # initialize dataframe with columns x0...xn for datapoints

    df = pd.DataFrame(np.vstack(zipped[0]))
    df = df.rename(columns={i: f'x{i}' for i in df.columns})

    # add other columns
    df['index'] = zipped[1]
    df['y_pred'] = zipped[2]

    # prob can be either a number or an array
    prob = zipped[3]

    # if there are arrays in prob, then loose nans will cause problems. They
    # must also be put into an array of the right length before we can stack
    # the whole thing
    has_arrays = any([type(p) in [np.ndarray, list] for p in prob])
    if has_arrays:
        prob = [dim_correct(p) if type(p) is np.ndarray else p for p in prob]
        max_len = max([p.shape[1] for p in prob if type(p) is np.ndarray])

        prob = [
            np.array(max_len*[p]) if type(p) is float
            else p
            for p in prob
        ]

    prob = dim_correct(np.vstack(prob))

    for i in range(prob.shape[1]):
        df[f'prob{i}'] = prob[:, i]

    df['y_true'] = zipped[4]

    # use the index from the QueuePoints and sort
    df = df.set_index('index', drop=True)
    df = df.sort_index()

    return df
