from sklearn.cluster import MeanShift, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import NotFittedError


class DBShift(BaseEstimator, ClusterMixin):
    """Perform DBShift clustering on a vector array.

    DBShift is useful for splitting a feature space into regions for
    systematic exploration. Fitting occurs in two stages:
        1. DBSCAN is performed over the input to identify "main
           clusters" and outliers.
        2. Mean Shift clustering is performed over the DBSCAN
           outliers, to break these up into regions. These "outlier
           clusters" are labelled with negative integers.

    Prediction is done by kNN against the dataset used in fitting.

    Parameters
    ----------
    eps : float
        The `eps` parameter for DBSCAN. If None, `eps` is chosen to
        be the mean of the ranges of the input data dimensions,
        divided by 33.

    min_samples : int
        The `min_samples` parameter for DBSCAN. If None,
        `min_samples` is taken to be 1% of the input data.

    n_neighbors : int
        The `n_neighbors` (k) parameter for kNN. If None,
        `n_neighbors` is taken to be the same as `min_samples`.

    Attributes
    ----------
    labels_ : The cluster labels identified during fitting
    components_ : The vector array input used in fitting

    _dbscan : The internal DBSCAN classifier
    _meanshift : The internal Mean Shift classifier
    _knn : The internal KNN classifier
    """
    def __init__(self, eps=None, min_samples=None, n_neighbors=None):
        self.eps = eps
        self.min_samples = min_samples
        self.n_neighbors = n_neighbors

        self._dbscan = None
        self._meanshift = None
        self._knn = None

        self.labels_ = None
        self.components_ = None

    def fit(self, X, y=None):
        """Perform clustering.

        Parameters
        -----------
        X : array-like, shape=[n_samples, n_features]
            Samples to cluster.

        y : Ignored

        """
        # DBSCAN parameters
        if self.eps is not None:
            eps = self.eps
        else:
            eps = (X.max(axis=0) - X.min(axis=0)).mean() / 33

        if self.min_samples is not None:
            m = self.min_samples
        else:
            m = X.shape[0] // 100

        # Do dbscan
        self._dbscan = DBSCAN(eps=eps, min_samples=m)
        labels = self._dbscan.fit_predict(X)

        # Do mean shift if there are outliers (default parameters)
        outliers = X[labels == -1]
        self._meanshift = MeanShift()

        if outliers.shape[0]:
            outlier_clusters = self._meanshift.fit_predict(outliers)
            labels[labels == -1] = -1 - outlier_clusters

        # Fit KNN
        if self.n_neighbors is not None:
            k = self.n_neighbors
        else:
            k = self._dbscan.min_samples

        self._knn = KNeighborsClassifier(n_neighbors=k).fit(X, labels)

        # save output
        self.components_ = X
        self.labels_ = labels

        return self

    def predict(self, X):
        """Predict the cluster labels for the provided data using KNN

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)
            Test samples.

        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Cluster labels for each data sample.
        """
        if self._knn is None:
            raise NotFittedError

        return self._knn.predict(X)

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)
            Test samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of
            n_outputs of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are
            ordered by lexicographic order.
        """
        if self._knn is None:
            raise NotFittedError

        return self._knn.predict_proba(X)
