import numpy as np
import pandas as pd

from olac.clusterers import DBShift


class LaC(object):
    def __init__(
            self,
            windower,
            mab,
            clusterer=DBShift,
            n_obs=50000,
            prior_alpha=0.5,
            prior_beta=0.5,
            windower_kwargs={},
            clusterer_kwargs={},
            mab_kwargs={}
            ):
        """

        """
        self.n = 0
        self.n_obs = n_obs
        self.prior_beta = prior_beta
        self.prior_alpha = prior_alpha
        self.cluster_index = {}

        self.windower = windower
        self.clusterer = clusterer(clusterer_kwargs)
        self.mab = mab

    def init_store(self):
        """Create observation store

        | 0 | 1 |    2    |   3   |   4   |     5    |
        ------------------------------------------------
        | x | y | cluster | label | fraud | estimate |
        """
        self.store = pd.DataFrame(
                np.zeros((self.n_obs, 6)),
                columns=['x', 'y', 'cluster',
                         'label', 'fraud', 'estimate']
                )

    def init_cluster_array(self):
        """Create cluster data array

        |   0   |   1  |   2   |    3    |   4    |   5    |   6   |
        ------------------------------------------------------------
        | alpha | beta | theta | entropy | fraud | label | count |

        """
        # init the array
        self.clusters = pd.DataFrame(
                np.zeros((self.k, 7)),
                columns=['alpha', 'beta', 'theta', 'entropy',
                         'fraud', 'label', 'count']
                )

        # s
        self.clusters.loc[:, 'alpha'] = self.prior_alpha
        self.clusters.loc[:, 'beta'] = self.prior_beta
        self.clusters.loc[:, 'theta'] = self.prior_alpha / (self.prior_alpha + self.prior_beta)

        self._update_cluster_entropy()

    def update_cluster_index(self):
        """
        """
        for cluster in self.store['cluster'].unique():
            self.cluster_index[cluster] = self.store['cluster'] == cluster

    def _update_cluster_counts(self):
        """Aggregate the counts on cluster level
        """
        for cluster, ind in self.cluster_index.items():
            subset = self.store.loc[ind, ['label', 'fraud']]
            # update count
            self.clusters.loc[cluster, 'count'] = subset.loc[:, 'x'].size
            # update frauds and labels
            self.clusters.loc[cluster, ['label', 'fraud']] = subset.loc[:, ['label', 'fraud']].sum(axis=1)

    def _update_cluster_beta_params(self):
        """Determine the alpha and beta value of the conjugate prior
           Beta distribution.
        """
        self.clusters.loc[:, 'alpha'] = self.prior_alpha + self.clusters.loc[:, 'fraud']
        self.clusters.loc[:, 'beta'] = self.prior_beta + self.clusters.loc[:, 'label'] - self.clusters.loc[:, 'fraud']

    def _update_theta(self):
        """Estimate the probability of k = 1 for the bernoulli dist.

        The expected value of a Beta dist can be used to estimate the
        probability of k = 1.
        """
        self.clusters.loc[:, 'theta'] = (
                self.clusters.loc[:, 'alpha'] /
                (self.clusters.loc[:, 'alpha'] + self.clusters.loc[:, 'beta'])
                )

    def _update_cluster_entropy(self):
        """Update entropy score for each cluster

        :note: log2 is used over ln to obtain a maximum entropy of 1
        """
        self.clusters.loc[:, 'entropy'] = ((
                self.clusters.loc[:, 'theta'] - 1) * np.log2(1 - self.clusters.loc[:, 'theta']) -
                (self.clusters.loc[:, 'theta'] * np.log2(self.clusters.loc[:, 'theta']))
                )

    def fit_clusters(self):
        """
        """
        self.store.loc[:, 'cluster'] = self.clusterer.fit_predict(self.store.loc[:, ['x', 'y']])
        self.update_cluster_index()
        self._update_cluster_counts()

    def fit_observations(self, obs, estimates):
        """
        """
        lb = self.cnt + 1
        ub = lb + obs.shape[0]

        knnclusters = self.clusters.fit_predict(obs)

        self.store.iloc[lb:ub, :3] = np.hstack((obs, knnclusters))
        self.store.loc[lb:ub, 'estimate'] = estimates

        self.update_cluster_index()

    def update_clusters():
        pass
