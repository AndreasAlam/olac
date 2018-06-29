import numpy as np
from .utils import rotation_matrix

########################################################################################################################
#                                                    Learning at Cost                                                  #
########################################################################################################################


def rand_walk(start=0, steps=1000, batch=False, rvs_func=np.random.normal, **kwargs):
    """ Generate a random time-series where the movement at time t is a sample from the given distribution.

    Parameters
    ----------
    start : int
        The y coordinate at time zero
    steps : int
        Number of steps to generate. Will generate forever if `steps==0`.
    batch : boolean [Optional]
        Whether the functions should return or yield
    rvs_func: function
        The random variate generator function, e.g. np.random.normal
    **kwargs
            parameters for the rvs_func

    Return
    -------
    ndarray
        The generated time series if batch
    int
        If not batch, yield the next value
    """
    # condition to go on forever if steps==0, else stop when i==steps.
    def cond(i):
        if not steps:
            return True
        else:
            return i < steps
    if batch:
        return start + np.cumsum(rvs_func(**kwargs, size=(steps, 1)))
    else:
        i = -1
        # generate the (infinite) stream
        alpha = start
        while cond(i):
            i += 1
            tmp = alpha + rvs_func(**kwargs, size=1)
            yield tmp
            alpha = tmp
########################################################################################################################


def roving_balls(balls=2, steps=1000, period=1000, radius=5, vars=1,
                 center=(0, 0,)):
    """
    Generator for the roving balls dataset.

    The generator will terminate after `steps` steps, or will go on forever if
    `steps==0`.

    See notebooks/jdp-data-roving-balls.ipynb for examples.

    Parameters
    ----------
    balls : int
        The number of balls to use
    steps : int
        Number of steps to generate. Will generate forever if `steps==0`.
    period : int
        Period of rotation of the balls
    radius : int
        Radius of the entire dataset
    vars : float or iterable
        The variances of the balls. If iterable, must have length equal to
        number of balls.
    center : tuple
        The center of the entire dataset

    Yields
    ------
    np.ndarray
        Data point of form [x1, x2, label]

    """

    i = -1

    try:
        scales = balls * [float(vars)]
    except TypeError:
        assert len(vars) == balls
        scales = vars

    # condition to go on forever if steps==0, else stop when i==steps.
    def cond(i):
        if not steps:
            return True
        else:
            return i < steps

    # initialize the cluster centers (evenly distributed around circle)
    base_center = np.array(center) + np.array([0, radius])
    cluster_locs = []
    for lab in range(balls):
        theta = lab * 2 * np.pi / balls
        cluster_locs.append(base_center @ rotation_matrix(theta))

    cluster_locs = np.array(cluster_locs)

    dtheta = 2 * np.pi / period
    step_rot = rotation_matrix(dtheta)
    clusters = list(range(balls))

    # generate the (infinite) stream
    while cond(i):
        i += 1
        cluster_locs = cluster_locs @ step_rot  # rotate cluster centers
        lab = np.random.choice(clusters)  # choose a ball to draw from
        x = np.random.normal(loc=cluster_locs[lab], scale=scales[lab])
        yield np.hstack([x, lab])
