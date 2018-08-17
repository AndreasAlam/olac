# general
import os
import sys
import pandas as pd
import numpy as np
import warnings


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
    warnings.warn("The 'utils.rotation_matrix' method is deprecated, "
                  "use maths.rotation_matrix(theta: float) instead")
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


def get_params(cls, deep=True):
    """Get parameters for this estimator.

    Parameters
    ----------
    deep : boolean, optional
        If True, will return the parameters for this estimator and
        contained subobjects that are estimators.

    Returns
    -------
    params : mapping of string to any
        Parameter names mapped to their values.
    """

    def _get_param_names(cls):
        import sklearn
        """Get parameter names for the estimator
        Taken from sklearn
        """
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = sklearn.utils.fixes.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    out = dict()
    for key in _get_param_names(cls):
        # We need deprecation warnings to always be on in order to
        # catch deprecated param values.
        # This is set in utils/__init__.py but it gets overwritten
        # when running under python3 somehow.
        warnings.simplefilter("always", DeprecationWarning)
        try:
            with warnings.catch_warnings(record=True) as w:
                value = getattr(cls, key, None)
            if len(w) and w[0].category == DeprecationWarning:
                # if the parameter is deprecated, don't show it
                continue
        finally:
            warnings.filters.pop(0)

        # XXX: should we rather test if instance of estimator?
        if deep and hasattr(value, 'get_params'):
            deep_items = value.get_params().items()
            out.update((key + '__' + k, val) for k, val in deep_items)
        out[key] = value
    return out
