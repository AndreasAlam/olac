import numpy as np
from .utils import rotation_matrix
from scipy.stats import poisson


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


def cluster_generator(n_clusters=5, n_points=1000, slider='poisson',  cluster_width=10.0, amount_of_labels=2):
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
