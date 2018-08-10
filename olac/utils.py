# general
import os
import sys
import pandas as pd
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
    The 2x2 rotation matrix

    """
    import warnings
    from . import maths as mf
    warnings.warn("The 'utils.rotation_matrix' method is deprecated, use maths.rotation_matrix(theta: float) instead")
    return mf.rotation_matrix(theta)


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
