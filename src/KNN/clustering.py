"""Clustering Tools
"""
import numpy as np
import numpy.random as npr

class KMeans:

    def __init__(self, num_means: int):
        self.num_means = num_means
        self.rng = npr.default_rng(42)

    def _assign_iter(self,
                     means: np.ndarray,
                     dat: np.ndarray):
        """Single KMeans assignment iteration

        Args:
            means (np.ndarray): num_means x N
                array representing means
            dat (np.ndarray): num_sample x N
                array representing data samples
        """
        # make num_means x num_sample distance matrix
        # use mean to prevent potential overflow
        dist_mat = np.mean((dat[None,:] - means[:,None])**2.0, axis=2)
        # assign data to means --> num_sample array of indices
        clust_assigns = np.argmin(dist_mat, axis=0)
        return clust_assigns
    
    def _handle_cluster_loss(self,
                             new_means: np.ndarray,
                             grp_size: np.ndarray):
        """Cluster loss = when some mean has become useless
        Greedy strat:
        > Assign to largest group + eta (tiny)
        > count that group as split in half
        > repeat till enough

        Args:
            new_means (np.ndarray): num_means x N
                means of current update
            grp_size (np.ndarray): number of elems
                assigned to each cluster
                DESTRUCTIVE

        """
        add_means = []
        num_miss = self.num_means - np.shape(new_means)[0]
        mu_shape = np.shape(new_means[1])
        for i in num_miss:
            # largest group:
            maxg = np.argmax(grp_size)
            # remake group size
            v = grp_size[maxg] / 2.
            grp_size[maxg] = v
            grp_size = np.hstack((grp_size, [v]))
            # add mean:
            add_means.append(new_means[maxg] + self.rng.random(mu_shape))
        return add_means

    
    def _update_iter(self,
                     dat: np.ndarray,
                     clust_assigns: np.ndarray):
        """Single KMeans update iteration

        Args:
            dat (np.ndarray): num_sample x N array 
                representing data samples
            clust_assigns (np.ndarray): len num_sample
                array of sample indices
        """
        # sort in order of cluster assignments
        sinds = np.argsort(clust_assigns)
        clust_assigns = clust_assigns[sinds]
        dat = dat[sinds]
        # split across clusters
        _, uniqinds = np.unique(clust_assigns, return_index=True)
        grps = np.split(clust_assigns, uniqinds)[1:]
        means = [], grp_size = []
        for grp in grps:
            means.append(np.mean(dat[grp], axis=0))
            grp_size.append(len(grp))
        # deal with cluster loss
        if len(means) < self.num_means:
            means = means + self._handle_cluster_loss(np.array(means),
                                                      grp_size)
        return np.array(means)
