import numpy as np
from .utils import rotation_matrix


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
        The variances of the balls. If float, each ball will have the same var
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
