"""Clustering Tools
"""
from dis import dis
from ntpath import join
import numpy as np
import numpy.random as npr
import numpy.linalg as nplg

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

class wGMM:
    """Weighted Gaussian Mixture Modeling
    Weights = priors    
    """

    def __init__(self, num_means: int, tolerance: float = 1e-5):
        """Initialize weighted Gaussian Mixture Model

        Args:
            num_means (int): number of clusters/means
            tolerance (float): tolerance for decomposition
                prevents vanishing variance
        """
        self.num_means = num_means
        self.tolerance = tolerance
        self.rng = npr.default_rng(66)
        self.km = KMeans(num_means)

    def _calc_covar(self,
                    dat: np.ndarray,
                    means: np.ndarray):
        """Calculate covariance matrices

        Args:
            dat (np.ndarray): num_sample x N
                array of raw data samples
            means (np.ndarray): num_mean x N
                set of means
        
        Returns:
            np.ndarray: num_means x N x N
                covariance matrices for each mean
                order-matched to input
            np.ndarray: num_mean x num_sample x N
                differences (x - mu)
                where x = data
                mu = means
        """
        # subtract means
        # --> num_mean x num_sample x N
        di = dat[None] - means[:, None]
        # calc each covariance matrix
        # --> num_mean x N x N
        covars = np.mean(di[:,:,None] * di[:,:,:,None],
                         axis=1)
        return covars, di

    def _decompose_covar(self, covar_mat: np.ndarray):
        """SVD decomposition of covariance matrix
        --> get inverse and determinant of inverse

        Args:
            covar_mat (np.ndarray): N x N covariance matrix

        Returns:
            np.array: N x N prevision array
                inverse of covariance matrix
            float: | determinant | of the precision
        """
        # covariance matrix is real and symmetric
        # --> hermetian
        # decompose into U S V^T
        # NOTE: symmetric matrix --> singular
        # values = absolute values of eigenvalues
        u, s, vt = nplg.svd(covar_mat, hermitian=True)

        # apply tolerance to singular values:
        s[s < self.tolerance] = self.tolerance

        # determinant = product of singular values
        det = np.prod(s)

        # invert s
        # since diagonal --> elem-wise inverse
        sinv = 1. / s

        # inverse calculation
        precision = vt.T @ np.diag(sinv) @ u.T

        return precision, det
    
    def _mmult(self,
               A: np.ndarray,
               B: np.ndarray,
               C: np.ndarray):
        """Helper function A mult B mult C
        A = num_sample x N
        B = N x M
        C = num_sample x M"""
        # --> num_sample x M
        m1 = np.sum(A[:, :, None] * B[None], axis=1)
        # --> num_sample
        m2 = np.sum(m1 * C, axis=1)
        return m2
    
    def _apply_gaussian(self,
                        di: np.ndarray,
                        precision: np.ndarray,
                        cov_det: float):
        """Calculate multivariate gaussian
        probabilities of data for single gaussian
        NOTE: assumes x - mu differences already
        computed

        Args:
            di (np.ndarray): x - mu differences
                num_sample x N
            precision (np.ndarray): single precision
                matrix
                N x N
            cov_det (float): determinant of covariance
                matrix = prevision ^ -1
        
        Returns:
            np.ndarray: forward probs = len num_sample
        """
        dim = np.shape(di)[1]
        m = self._mmult(di, precision, di)
        # numerator --> num_sample
        num = np.exp(-.5 * m)
        # denom --> float
        denom = np.sqrt(((2. * np.pi)**dim)
                         * cov_det)
        return num / denom

    def probs(self,
              means: np.ndarray,
              precisions: np.ndarray,
              cov_dets: np.ndarray,
              dat: np.ndarray,
              priors: np.ndarray):
        """Single GMM probability iteration
        == assignment
        P(cluster | sample) = P(sample | cluster) P(cluster) /
                                    sum(P(sample | cluster))

        Args:
            means (np.ndarray): num_means x N
                array representing means
            cov_dets (np.ndarray): covariance
                determinants = len num_mean
            precisions (np.ndarray): inverse of covariance
                matrices = num_means x N x N
            dat (np.ndarray): num_sample x N
                array representing data samples
            priors (np.ndarray): prior probs
                = len num_sample
        
        Returns:
        """
        # subtract means
        # --> num_mean x num_sample x N
        di = dat[None] - means[:, None]
        # forward probs
        fprobs = []
        for i, cov_det in enumerate(cov_dets):
            fprobs.append(self._apply_gaussian(di[i], precisions[i], cov_det))
        # --> num_mean x num_sample
        fprobs = np.array(fprobs)
        # TODO: posterior probs
        joint_probs = fprobs * priors[None]
        post_probs = joint_probs / np.sum(joint_probs,
                                          axis=0, keepdims=True)
        return post_probs, joint_probs, fprobs


# TESTING

def test_kmeans():
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

    return dat, means

def test_gmm():
    # NOTE: need scipy to run test
    # more specific testing:
    print('GMM testing')
    G = wGMM(2)

    # mmult ~ guarantees right order calc
    print('mmult test')
    G._mmult(npr.rand(20, 5),
             npr.rand(5, 8),
             npr.rand(20, 8))

    from scipy.stats import multivariate_normal
    for _ in range(3):
        mu = npr.rand(4)
        cov_raw = npr.rand(4, 4)
        cov = 5 * cov_raw @ cov_raw.T
        print(cov)
        print(nplg.eig(cov))
        dat = npr.rand(4)
        var = multivariate_normal(mean=mu, cov=cov)
        # compare against ours:
        precision, cov_det = G._decompose_covar(np.array(cov))
        print('precision check: should be identity')
        print(np.array(cov) @ precision)
        print('determinant check')
        print(nplg.det(np.array(cov)))
        print(cov_det)
        pr = G.probs(np.array(mu)[None], np.array(precision)[None], np.array(cov_det)[None],
                    np.array(dat)[None], np.array([[1.]]))
        print('iter; probs')
        print(var.pdf(dat))
        print(pr)
        input('cont?')



if __name__ == '__main__':
    import pylab as plt

    # kmeans testing
    # dat, means = test_kmeans()

    # wGMM testing
    test_gmm()
