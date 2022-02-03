"""Clustering Tools
"""
from dis import dis
import numpy as np
import numpy.random as npr

class KMeans:

    def __init__(self, num_means: int):
        self.num_means = num_means
        self.rng = npr.default_rng(42)

    def assign_iter(self,
                     means: np.ndarray,
                     dat: np.ndarray):
        """Single KMeans assignment iteration

        Args:
            means (np.ndarray): num_means x N
                array representing means
            dat (np.ndarray): num_sample x N
                array representing data samples
        
        Returns:
            np.ndarray: cluster assignments
            np.ndarray: distance matrix
                num_means x num_sample

        """
        # make num_means x num_sample distance matrix
        # use mean to prevent potential overflow
        dist_mat = np.mean((dat[None,:] - means[:,None])**2.0, axis=2)
        # assign data to means --> num_sample array of indices
        clust_assigns = np.argmin(dist_mat, axis=0)
        return clust_assigns, dist_mat
    
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
        dgrps = np.split(dat, uniqinds)[1:]
        means, grp_size = [], []
        for dgrp in dgrps:
            means.append(np.mean(dgrp, axis=0))
            grp_size.append(len(dgrp))
        # deal with cluster loss
        if len(means) < self.num_means:
            means = means + self._handle_cluster_loss(np.array(means),
                                                      grp_size)
        return np.array(means)
    
    def _init_means(self, dat: np.ndarray):
        """Get initial means
        > randomly choose data points from dat

        Args:
            dat (np.ndarray): N x M raw data
        
        Returns:
            np.ndarray: num_means x M means
        """
        assert(np.shape(dat)[0] >= self.num_means), "not enough data"
        inds = np.arange(np.shape(dat)[0])
        self.rng.shuffle(inds)
        return dat[inds[:self.num_means]]
    
    def run(self, dat: np.ndarray, num_iter: int = 5):
        """Single KMean runs

        Args:
            dat (np.ndarray): raw data
            num_iter (int): number of iterations
        
        Returns:
        """
        means = self._init_means(dat)
        for _ in range(num_iter):
            clust_assigns, _ = self.assign_iter(means, dat)
            means = self._update_iter(dat, clust_assigns)
        _, dist_mat = self.assign_iter(means, dat)
        return means, dist_mat
    
    def multi_run(self,
                  dat: np.ndarray,
                  num_iter: int = 5,
                  num_run: int = 3):
        """Run KMeans from different starting points

        Args:
            dat (np.ndarray): raw data
            num_iter (int, optional):
            num_run (int, optional):
        """
        min_dist, means = None, None
        for _ in range(num_run):
            cmeans, dists = self.run(dat, num_iter)
            cdist = np.mean(np.argmin(dists, axis=1))
            if min_dist is None or cdist < min_dist:
                min_dist = cdist
                means = cmeans
        return means


if __name__ == '__main__':
    # kmeans testing
    import pylab as plt
    km = KMeans(2)
    dat = np.array([[1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [100, 100, 100, 100],
                    [101, 100, 101, 99]])
    means, dist_mat = km.run(dat)
    print(means)
    print(dist_mat)
    means = km.multi_run(dat)
    print(means)

    # more kmeans testing
    rng = npr.default_rng(0)
    v1 = npr.rand(10, 2)
    v2 = npr.rand(15, 2) + .5 * np.ones((15, 2))
    dat = np.vstack((v1, v2))
    means = km.multi_run(dat)
    print(means)
    print(km.assign_iter(means, dat))
    plt.figure()
    plt.scatter(dat[:,0], dat[:,1])
    plt.scatter(means[:,0], means[:,1], c='r')
    plt.show()