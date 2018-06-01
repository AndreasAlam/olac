import numpy as np
from .utils import rotation_matrix


def roving_balls(steps=1000, period=1000, radius=5, var1=1, var2=1,
                   center=(0, 0)):
    """
    Generator for the roving balls dataset.

    The generator will terminate after `steps` steps, or will go on forever if
    `steps==0`.

    See notebooks/jdp-data-roving-balls.ipynb for examples.

    Parameters
    ----------
    steps : int
        Number of steps to generate. Will generate forever if `steps==0`.
    period : int
        Period of rotation of the balls
    radius : int
        Radius of the entire dataset
    var1 : float
        Variance of ball 1
    var2 : float
        Variance of ball 2
    center : tuple
        The center of the entire dataset

    Yields
    ------
    np.ndarray
        Data point of form [x1, x2, label]

    """

    i = -1

    # condition to go on forever if steps==0, else stop when i==steps.
    def cond(i):
        if not steps:
            return True
        else:
            return i < steps

    # initialize the cluster centers (one on either side of the global center)
    locs = np.array([center, center]).copy()
    locs[0, 0] += radius
    locs[1, 0] -= radius

    scales = [var1, var2]
    dtheta = 2 * np.pi / period
    rot = rotation_matrix(dtheta)

    while cond(i):
        i += 1
        locs = locs @ rot  # rotate cluster centers by dtheta
        lab = np.random.choice([0, 1])  # choose a ball to draw from
        x = np.random.normal(loc=locs[lab], scale=scales[lab])
        yield np.hstack([x, lab])
