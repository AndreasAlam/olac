import time
import numpy as np
import pandas as pd
from scipy.stats import poisson

from .maths import rotation_matrix

from . import utils, models

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
    `steps==0`.git st

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


def dynamify_data(X, y=None, transition_rate=0.1, cluster_simul=1,
                  dbshift_args=tuple(), dbshift_kwargs=dict()):

    X = np.array(X)

    if y is not None:
        y = np.array(y)
        assert y.shape[0] == X.shape[0], "y must have same number of rows as X"

    # do clustering
    dbs_clf = models.DBShift(*dbshift_args, **dbshift_kwargs)
    clusters = pd.Series(dbs_clf.fit_predict(X))

    # prep cluster indices
    n_clusters = clusters.nunique()
    n_points = X.shape[0]

    # deal with cluster_simul being a fraction
    if cluster_simul < 1:
        cluster_simul = min(1, np.round(n_clusters * cluster_simul))

    cluster_sizes = clusters.value_counts()
    cluster_labels = np.random.permutation(cluster_sizes.index.values)

    X_inds = np.arange(0, n_points)

    # shuffled indices per cluster
    cluster_inds = {
        c: list(np.random.permutation(X_inds[clusters == c]))
        for c in cluster_labels
    }

    # generator for cluster probability distributions
    cluster_probs = utils.slide_probability_over_list(
        len(X),
        n_clusters,
        transition_rate,
        var=0.63*cluster_simul,
    )

    for p in cluster_probs:
        # Rescale cluster probabilities using relative size
        for i, c in enumerate(cluster_labels):
            p[i] *= len(cluster_inds[c])

        # renormalize
        p /= np.sum(p)

        # choose a cluster to randomly draw from
        c = np.random.choice(cluster_labels, p=p)
        ind = cluster_inds[c].pop()

        if y is not None:
            output = np.hstack([X[ind], y[ind]])
        else:
            output = X[ind]

        yield output


def satellites(n_points=0, n_satellites=3, contamination=0.01,
               base_center=(0, 0,), base_std=1., satellite_std=0.3,
               satellite_radius=4., satellite_radius_std=0.,
               satellite_std_std=0., satellite_progress_rate=0.1,
               satellite_simul=1):
    """
    Generate a dataset with a big central cluster surrounded by smaller
    "satellite" clusters. The satellites are evenly spaced in a circle around
    the base with radius given by `satellite_radius`. They can be made to
    randomly vary from each other in spread and distance from the center
    by setting `satellite_radius_std` and `satellite_std_std`.

    As points are generated, the probability of an outlier (satellite point)
    shifts across the satellites. The speed of this shift can be controlled
    using `satellite_progress_rate`.

    Parameters
    ----------
    n_points : int, default 0
        Number of points to generate (0 for infinite)

    n_satellites : int, default 3
        Number of satellites

    contamination : float, default 0.1
        Fraction of points which are outliers (in satellites)

    base_center : iterable of length 2 (default tuple(0,0))
        Location of the base cluster

    base_std : float, default 1.
        Std. dev. of base cluster

    satellite_std : float, default 0.
        Std. dev. of satellites

    satellite_radius : float, default 4.
        Distance of satellite centers from base cluster center

    satellite_radius_std : float, default 0.
        Std. dev. between distances of different satellites from base center

    satellite_std_std : float, default 0.1
        Std. dev. between std. devs of different satellites

    satellite_progress_rate : float, default 0.1
        Rate at which probability density of an outlier coming from a
        particular satellite shifts over time

    satellite_simul: float, default 1.
        Amount of satellites that can be expected to be active at one time

    Returns
    -------
    np.ndarray : [x, y, label]
    """

    # generator for satellite probability distribution
    satellite_probs = utils.slide_probability_over_list(
        n_points,
        n_satellites,
        satellite_progress_rate,
        var=0.63*satellite_simul,
    )

    # variation between satellite radius
    rad_facts = np.random.normal(1, satellite_radius_std, (n_satellites, 1,))

    satellite_centers = (satellite_radius
                         * rad_facts
                         * utils.unit_circle_points(n_satellites)
                         ) + np.array(base_center)
    satellite_stds = satellite_std * np.ones((n_satellites,))\
                     * np.random.normal(1, satellite_std_std, (n_satellites,))

    satellite_inds = list(range(n_satellites))

    for p in satellite_probs:
        outlier = np.random.uniform(0, 1) < contamination
        label = int(outlier)

        if not outlier:
            x = np.random.normal(loc=base_center, scale=base_std)

        else:
            ind = np.random.choice(satellite_inds, p=p)
            x = np.random.normal(loc=satellite_centers[ind],
                                 scale=satellite_stds[ind])

        yield np.hstack([x, [label]])


def cluster_generator(n_clusters=5, n_points=1000, slider='poisson',
                      cluster_width=10.0, amount_of_labels=2):
    """
    Function that returns a dataset with timestamps/time dimension and positional coordinates.
    The function will generate x points where x is determined by (n_clusters + tail_time)/dt
    To determine the label of a cluster there is a randomly drawn from a pdf. The pdf is modular in time. Two options
    are used in the current model. One is a sum of squared sinuses and the other a moving poisson distribution.

    Parameters
    ----------
    n_clusters: int
        Default n_clusters = 5, amount clusters that will be generated in the simulation
    n_points: int
        Default n_points = 1000, amount of points that will be generated
    slider: basestring
        Either poisson or sinus, determines the way how the probabilities move along with the time
    cluster_width: float
        The maximum width of the cluster distributions
    amount_of_labels: int
        Default amount_of_labels = 2, because you have either good or bad

    Returns
    -------
    np.array
        points which will occur in time with a label and position
    """
    # --------------------------------- #
    # initializing the start parameters |
    # --------------------------------- #
    tail_time = n_clusters*0.8
    end_time = n_clusters + tail_time
    list_sliders = ['sinus', 'poisson']
    labels = np.arange(amount_of_labels)
    dt = end_time/n_points

    # checks for input variables
    # --------------------------
    assert slider in list_sliders, 'Wrong type of slider choose from the list: {}'.format(list_sliders)
    assert n_clusters >= 1, 'At least one cluster is needed for the simulation'
    assert type(n_clusters) is int, 'n_clusters is not an integer, please fill in an integer number'
    assert type(amount_of_labels) is int, 'fill an integer amount of labels'
    assert amount_of_labels <= n_clusters, 'There cannot be more labels then clusters'

    # =========================================================
    # FUNCTION DEFINITION
    # =========================================================

    # ---------------------------------
    # Time slider with sinus dependency
    # ---------------------------------
    def p_time_sin(n_clusters, centers, dt, end_time):
        """

        Parameters
        ----------
        n_clusters: int
            Number of clusters generated for simulation
        centers: np.array
            array of centers of the clusters with their std etc
        dt: float
            delta t, steps in time made during the cluster generation
        end_time: float or int
            Determines when the sliding probabilities will stop

        Yields
        ------
        np.array
            points which will occur in time with a label and position

        """
        p_t = []
        clusters_in_time = []
        for t in np.arange(0, end_time, dt):
            # determine probabilities for a certain moment in time
            chance = []
            for s in range(len(centers.T[0])):
                chance.append(np.sin((np.pi / n_clusters) * (t + s)) ** 2)
            p = chance / sum(chance)
            p_t.append(p)

            cluster_num = int(np.random.choice(centers.T[0], p=p))

            # recall the x and y for a certain cluster
            sigx = centers[cluster_num][3]
            sigy = centers[cluster_num][4]

            # generate random x and y for the cluster according to normal distributions
            x_c = np.random.normal(centers[cluster_num][1], sigx)
            y_c = np.random.normal(centers[cluster_num][2], sigy)

            # assign the right labels to the cluster
            label_point = centers[cluster_num][5]

            # keep track of the clusters
            clusters_in_time.append({'time': t, 'cluster_name': cluster_num, 'x': x_c, 'y': y_c, 'label': label_point})
            yield np.array([x_c, y_c, label_point])

    # -----------------------------------
    # Time slider with poisson dependency
    # -----------------------------------

    # keeps track of the
    def p_time_poisson(centers, dt, end_time):
        """
        This function generates the behavior of the probabilities over time. In this case a moving poission
        distribution is used. The mean of the poission moves along a linear axis in time.
        For each step in time a point of a cluster is generated.

        Parameters
        ----------
        centers: np.array
            array of centers of the clusters with their std etc
        dt: float
            delta t, steps in time made during the cluster generation
        end_time: float or int
            Determines when the sliding probabilities will stop

        Yields
        ------
        np.array
            points which will occur in time with a label and position

        """
        p_t = []  # list that stores the probabilities
        clusters_in_time = []  # list that keeps track of the occurring points
        for t in np.arange(0, end_time, dt):
            chances = poisson.pmf(centers.T[0], t)
            p = chances / sum(chances)  # list of probabilities in a certain moment in time
            p_t.append(p)

            # choose the cluster number randomly from the list with cluster centers
            # according to the probability distribution p
            cluster_num = int(np.random.choice(centers.T[0], p=p))

            # recall the sig x and sig y for a certain cluster
            sigx = centers[cluster_num][3]
            sigy = centers[cluster_num][4]

            # generate random x anc y points for the cluster point according to normal distributions
            x_c = np.random.normal(centers[cluster_num][1], sigx)
            y_c = np.random.normal(centers[cluster_num][2], sigy)

            # assign the right label to the clusters
            label_point = centers[cluster_num][5]

            # keep track of the clusters
            clusters_in_time.append({'time': t, 'cluster_name': cluster_num, 'x': x_c, 'y': y_c, 'label': label_point})
            yield np.array([x_c, y_c, label_point])

    # ========================================================
    # Data generation
    # ========================================================
    centers = []
    for i in range(n_clusters):
        # Generate cluster centers in the x and y plane
        x = np.random.uniform(0, 1000)
        y = np.random.uniform(0, 1000)
        # Generate the widths of the clusters and make sure that they are bigger then 0
        sig_x = abs(np.random.normal(cluster_width, cluster_width*0.2))
        sig_y = abs(np.random.normal(cluster_width, cluster_width*0.2))

        # Assign the label via random to a cluster
        label = np.random.choice(labels)

        # store everything in the list
        centers.append([int(i), x, y, sig_x, sig_y, label])
    centers = np.array(centers)

    # choose between the sliders and their complementary functions
    if slider == 'sinus':
        return p_time_sin(n_clusters, centers, dt, end_time)
    elif slider == 'poisson':
        return p_time_poisson(centers, dt, end_time)


def delayed_generator(data_generator, delay, precision=1e-3):
    """
    A wrapper function to delay the output of another generator.

    The delay parameter controls how much time should pass between iterations.
    Note that if the underlying data_generator is slower than the delay time,
    then the slower time will be observed.

    delay may also be callable, for generating random delays at each iteration.
    Then delay() will be called at each iteration, and the output number will
    be used as the delay for that iteration.


    Parameters
    ----------
    data_generator: iterable
        The generator whose output to delay

    delay: float or callable
        The delay per iteration. If callable, delay() should return a float. It
        will be called once per iteration. Can be used to create a random delay.

    precision: float (default 1e-3)
        How precise the delay should be. If precision=0.1, then the observed
        delay should be within 0.1 of the delay parameter. Note that a higher
        precision requires more CPU cycles.

        This parameter is an indication, not a guarantee, especially in cases
        of long running underlying data_generator processes.

    Yields
    ------
    The items of data_generator

    """
    start = time.time()

    for point in data_generator:
        if callable(delay):
            d = delay()
        else:
            d = delay

        # Don't just sleep for delay, in case the call to data_generator takes
        # time to complete
        while time.time() - start <= (1-precision)*d:
            time.sleep(precision*d)

        yield point
        start = time.time()


def scaling_generator(data_generator, x_min, x_max, dp0=0, dp1=1):
    """
    A wrapper to scale the output of the data generators. In pricnuiple they
    should be between -1 and -1 for the most optimal performance of the model.
    However, most of our data generators are on different scales, so this warpper
    takes that output and rescales it to 0 and 1.

    Parameters
    ----------
    data_generator: iterable
        The generator whose output to delay

    x_min: float
        Lowest value of the scale on which the data point should be

    x_max: float
        Highest value of the scale on which the data point should be
        """

    x_shift = x_min

    dp = np.array((dp0, dp1))
    x_min -= x_shift
    x_max -= x_shift
    for point in data_generator:
        point[:2] -= x_shift
        point[:2] = dp[0] + (point[:2] - x_min)*(dp[1] - dp[0])/(x_max - x_min)
        yield point


def generator_from_csv(path, data_columns, label_column, n_points=None, **args):
    """
    A wrapper around a dataset to output it as a generator.

    Parameters
    ----------
    path: str
        Path to the file

    data_columns: list
        List of stings of the columns containing the datapoint values

    label_columns: str
        name of the label columns

    n_points: int
        number of points to return from the dataset
    """
    data_columns.append(label_column)
    type_dict = {'xlsx': 'excel', 'csv': 'csv'}
    filetype = path.split('.')[-1]
    data = getattr(pd, 'read_' + type_dict[filetype])(path, **args)
    if n_points is None:
        n_points = len(data)
    data = data.loc[:n_points, data_columns]

    for point in data.values:
        yield point


def generator_from_df(data, data_columns=None, label_column=None, n_points=None):
    """
    A wrapper around a dataset to output it as a generator.

    Parameters
    ----------
    data: pandas DataFrame
        pandas data frame containing the data and label columns

    data_columns: list
        List of stings of the columns containing the datapoint values.
        If none it is assumed the first two columns are the values.

    label_columns: str
        name of the label columns. If none it is assumed the last column is the label

    n_points: int
        number of points to return from the dataset
    """
    try:
        data = data.loc[:n_points, data_columns]
    except TypeError:
        pass

    for point in data.values:
        yield point

