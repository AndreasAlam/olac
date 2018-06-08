import numpy as np
from scipy.stats import poisson


def cluster_generator(n_clusters=5, slider='poisson', dt=0.01, tail_time=5, cluster_width=10):
    """
    Data set generator for clusters that pop up in time. The cluster points have have 4 features [ cluster_label, time
    of occurance, x coordinate, y coordinate]
    This is generated via random processes and the probability of a point occuring from a certain cluster moves over
    time. One can chose two options for the slider.
    :param n_clusters: amount of clusters that will occur over time in the 2D space
    :param slider: One can choose from: {'poisson', 'sinus'}
    slider describes how the probabilities will behave in time
    :param dt: the size of time steps
    :param tail_time: how much till will the distribution slide further than the index of the latest cluster
    :param cluster_width: Modulate the cluster width of multiple clusters the width of the clusters will determined
    according to a uniform distribution
    :return: np.array of dicts with the following format {cluster_name: , time: , x: , y: }

    """

    # initializing the start parameters
    # ---------------------------------
    end_time = n_clusters + tail_time
    list_sliders = ['sinus', 'poisson']

    # checks for input variables
    # --------------------------
    assert slider in list_sliders, 'Wrong type of slider choose from the list: {}'.format(list_sliders)
    assert n_clusters >= 1, 'At least one cluster is needed for the simulation'
    assert type(n_clusters) is int, 'n_clusters is not an integer, please fill in an integer number'

    # =========================================================
    # FUNCTION DEFINITION
    # =========================================================

    # ---------------------------------
    # Time slider with sinus dependency
    # ---------------------------------
    def p_time_sin(n_clusters, centers, dt, end_time):
        """
        :param n_clusters: number of clusters that are generated
        :param centers: list of center points of the clusters
        :param dt: delta t
        :param end_time: the time when the simulation ends
        :return: np.array of dicts with the following format {cluster_name: , time: , x: , y: }
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
            x = np.random.normal(centers[cluster_num][1], sigx)
            y = np.random.normal(centers[cluster_num][2], sigy)
            clusters_in_time.append({'time': t, 'cluster_name': cluster_num, 'x': x, 'y': y})
        return clusters_in_time

    # -----------------------------------
    # Time slider with poisson dependency
    # -----------------------------------

    # keeps track of the
    def p_time_poisson(centers, dt, end_time):
        """
        :param centers: list of centers of the points of the clusters
        :param dt: delta t
        :param end_time: the time when the simulation ends
        :return: np.array of dicts with the following format {cluster_name: , time: , x: , y: }
        """
        p_t = []  # list that stores the probabilities
        clusters_in_time = []  # list that keeps track of the occurring points
        for t in np.arange(0, end_time, dt):
            chances = poisson.pmf(centers.T[0], t)
            p = chances / sum(chances) # list of probabilities in a certain moment in time
            p_t.append(p)

            # choose the cluster number randomly from the list with cluster centers
            # according to the probability distribution p
            cluster_num = int(np.random.choice(centers.T[0], p=p))

            # recal the sig x and sig y for a certain cluster
            sigx = centers[cluster_num][3]
            sigy = centers[cluster_num][4]

            # generate random x anc y points for the cluster point according to normal distributions
            x = np.random.normal(centers[cluster_num][1], sigx)
            y = np.random.normal(centers[cluster_num][2], sigy)
            clusters_in_time.append({'time': t, 'cluster_name': cluster_num, 'x': x, 'y': y})
        return clusters_in_time

    # ========================================================
    # Data generation
    # ========================================================
    centers = []
    for i in range(n_clusters):
        # Generate cluster centers in the x and y plane
        x = np.random.uniform(0, 1000)
        y = np.random.uniform(0, 1000)
        # Generate the widths of the clusters and make sure that they are bigger then 0
        sig_x = abs(np.random.uniform(0, cluster_width))
        sig_y = abs(np.random.uniform(0, cluster_width))

        # store everything in the list
        centers.append([int(i), x, y, sig_x, sig_y])
    centers = np.array(centers)

    # choose between the sliders and their complementary functions
    if slider == 'sinus':
        return p_time_sin(n_clusters, centers, dt, end_time)
    elif slider == 'poisson':
        return p_time_poisson(centers, dt, end_time)
