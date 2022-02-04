"""Clustering Tools
"""
import numpy as np
import numpy.random as npr
import numpy.linalg as nplg

class wKMeans:
    """Weighted Kmeans"""

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
                     clust_assigns: np.ndarray,
                     priors: np.ndarray):
        """Single KMeans update iteration

        Args:
            dat (np.ndarray): num_sample x N array 
                representing data samples
            clust_assigns (np.ndarray): len num_sample
                array of sample indices
            priors (np.ndarray): prior probabilities
                = len num_sample array
        """
        # sort in order of cluster assignments
        sinds = np.argsort(clust_assigns)
        # split across clusters
        _, uniqinds = np.unique(clust_assigns[sinds], return_index=True)
        dgrps = np.split(sinds, uniqinds)[1:]
        means, grp_size = [], []
        for dgrp in dgrps:
            # normalize priors within cluster
            cprior = priors[dgrp]
            weights = cprior / np.sum(cprior)
            means.append(np.sum(dat[dgrp] * weights[:,None], axis=0))
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
    
    def run(self,
            dat: np.ndarray,
            priors: np.ndarray,
            num_iter: int = 5):
        """Single KMean runs

        Args:
            dat (np.ndarray): raw data
            priors(np.ndarray): prior probabilities
                = len num_sample array
            num_iter (int): number of iterations
        
        Returns:
            np.ndarray: means
                num_mean x N
            np.ndarray: distance matrix
                num_means x num_sample
        """
        means = self._init_means(dat)
        for _ in range(num_iter):
            clust_assigns, _ = self.assign_iter(means, dat)
            means = self._update_iter(dat, clust_assigns, priors)
        _, dist_mat = self.assign_iter(means, dat)
        return means, dist_mat
    
    def multi_run(self,
                  dat: np.ndarray,
                  priors: np.ndarray,
                  num_iter: int = 5,
                  num_run: int = 3):
        """Run KMeans from different starting points

        Args:
            dat (np.ndarray): raw data
            priors (np.ndarray): prior probabilities
                on raw data = len num_sample array
            num_iter (int, optional):
            num_run (int, optional):
        
        Returns:
            np.ndarray: means
                num_mean x N
            np.ndarray: distance matrix
                num_mean x num_sample
        """
        assert(np.sum(priors) == 1), "priors must sum to 1"
        min_dist, means, dmz = None, None, None
        for _ in range(num_run):
            cmeans, dist_mat = self.run(dat, priors, num_iter)
            cdist = np.mean(np.argmin(dist_mat, axis=1))
            if min_dist is None or cdist < min_dist:
                min_dist = cdist
                means = cmeans
                dmz = dist_mat
        return means, dmz

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
        self.km = wKMeans(num_means)

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

    # TODO: there's a confusion as to
    # what's meant by prior
    # Priors i'm referring to
    # = weights for datapoints
    # ... not assignment priors!
    # TODO: where do pt_weights come in?

    def probs(self,
              raw_diff: np.ndarray,
              precisions: np.ndarray,
              cov_dets: np.ndarray,
              mixing_coeffs: np.ndarray):
        """Single GMM probability iteration
        == assignment

        Args:
            raw_diff (np.ndarray): x - mu
                num_mean x num_sample x N
            precisions (np.ndarray): inverse of covariance
                matrices = num_means x N x N
            cov_dets (np.ndarray): covariance
                determinants = len num_mean
            mixing_coeffs (np.ndarray): coefficients
                for each guassian (height of gauss)
                = len num_mean array
        
        Returns:
            np.ndarray: posterior probabilities
                num_mean x num_sample array
            np.ndarray: joint probabilities
                num_mean x num_sample array
            np.ndarray: forward probabilities
                num_mean x num_sample array
        """
        # forward probs
        fprobs = []
        for i, cov_det in enumerate(cov_dets):
            fprobs.append(self._apply_gaussian(raw_diff[i], precisions[i], cov_det))
        # --> num_mean x num_sample
        fprobs = np.array(fprobs)

        # multiply forward probs by mixing_coeffs
        # --> mixing_probs
        # --> num_mean x num_sample
        mixing_probs = mixing_coeffs[:, None] * fprobs

        # posterior probs
        # normalizes mixing probs within cluster:
        post_probs = mixing_probs / np.sum(mixing_probs,
                                           axis=0, keepdims=True)
        return post_probs, mixing_probs, fprobs

    def update(self, dat: np.ndarray, post_probs: np.ndarray,
                raw_diff: np.ndarray):
        """Update Step of WGMM
        Calculate new means, covariances

        Args:
            dat (np.ndarray): raw data
                num_samples x N array
            post_probs (np.ndarray): posterior
                probabilities
                num_mean x num_sample array
            raw_diff (np.ndarray): raw difference
                from previous step... will 
                be updated here
        
        Returns:
            np.ndarray: updated mean
                num_mean x N
            np.ndarray: updated covariance
                num_mean x N x N
            np.ndarray: updated mixing coeffs
                len = num_mena
        """
        # M-Step:
        # convert posteriors to weights:
        weights = post_probs / np.sum(post_probs, axis=1,
                                      keepdims=True)

        # update mixing_coeffs:
        # --> num_mean
        mix_coeffs = np.sum(post_probs, axis=1)

        # update means:
        # --> num_mean x N
        means = np.sum(weights[:,:,None] * dat[None],
                        axis=1)
        
        # update covariance:
        # outp = num_mean x num_sample x N x N
        outp = raw_diff[:,:,None] * raw_diff[:,:,:,None]
        # contract along samples:
        # --> num_mean x N x N
        cov = np.sum(weights[:,:,None,None] * outp,
                      axis = 1)
    
        return means, cov, mix_coeffs
    
    def run(self,
            dat: np.ndarray,
            priors: np.ndarray,
            num_iter: int = 10):
        """Run GMM

        Args:
            dat (np.ndarray): raw data
                num_sample x N
            priors (np.ndarray): prior probabilities
                len num_sample array
            num_iter (int, optional): number of update
                iterations

        Returns:
            np.ndarray: means
                num_mean x N array
            np.ndarray: covariance matrices
                num_mean x N x N array
            np.ndarray: posterior probabilities
                num_mean x num_sample array
        """

        # init with kmeans:
        # --> means = num_mean x N
        # --> dist_mat = num_mean x num_sample
        means, dist_mat = self.km.multi_run(dat, priors, num_iter, 2)
        # posterior estimate based on distance matrix:
        edist = np.exp(-2. * dist_mat) * priors[None]
        posts = edist / np.sum(edist, axis=0, keepdims=True)

        # update raw difference
        # calculate raw difference:
        # --> num_mean x num_sample x N
        raw_diff = dat[None] - means[:, None]

        # repeated iterations of 
        # > means / covariance updates
        # > posterior probability calc
        plt.figure()

        means, covars = None, None
        for _ in range(num_iter):
            means, covars, mix_coeffs = self.update(dat, posts, raw_diff)

            print('mixing coeffs')
            print(mix_coeffs)

            # update raw difference
            # calculate raw difference:
            # --> num_mean x num_sample x N
            raw_diff = dat[None] - means[:, None]

            plt.scatter(means[:, 0], means[:, 1])

            # decompose each covariance matrix
            # --> precision and determinant simul calc
            precisions, cov_dets = [], []
            for i in range(np.shape(covars)[0]):
                prec, covd = self._decompose_covar(covars[i])
                precisions.append(prec)
                cov_dets.append(covd)

            posts, _, _ = self.probs(raw_diff, np.array(precisions), np.array(cov_dets),
                                     mix_coeffs)
        
        plt.show()
        
        return means, covars, posts


        # TODO: how do you factor in priors?
        # > don't matter for forward probabilities
        # > do they matter for gamma / posteriors?
        # > > normalize across means
        # > > they do NOT matter (unlike mixing coeffs)
        # > > cuz will just be a scalar that pops out of sums
        # > > TODO: do this math out!
        # > They definitely factor in to updates!


# TESTING

def test_kmeans():
    km = wKMeans(2)
    dat = np.array([[1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [100, 100, 100, 100],
                    [101, 100, 101, 99]])
    priors = np.ones((4,)) / 4.
    means, dist_mat = km.run(dat, priors)
    print(means)
    print(dist_mat)
    means, _ = km.multi_run(dat, priors)
    print(means)

    # more kmeans testing
    v1 = npr.rand(10, 2)
    v2 = npr.rand(15, 2) + .9 * np.ones((15, 2))
    priors = np.ones((25,)) / 25.
    dat = np.vstack((v1, v2))
    means, _ = km.multi_run(dat, priors)
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
        raw_diff = np.array(dat)[None,None] - np.array(mu)[None, None]
        pr = G.probs(raw_diff, np.array(precision)[None], np.array(cov_det)[None],
                     np.array([[1.]]))
        print('iter; probs')
        print(var.pdf(dat))
        print(pr)
        input('cont?')
    
    # more testing
    v1 = npr.rand(10, 2)
    v2 = npr.rand(15, 2) + .9 * np.ones((15, 2))
    priors = np.ones((25,)) / 25.
    dat = np.vstack((v1, v2))
    means, covars, posts = G.run(dat, priors)
    print('means')
    print(means)
    print('covars')
    print(covars)
    print('posteriors')
    print(posts)
    plt.figure()
    plt.scatter(dat[:,0], dat[:,1])
    plt.scatter(means[:,0], means[:,1])
    plt.show()



if __name__ == '__main__':
    import pylab as plt

    # kmeans testing
    #test_kmeans()

    # wGMM testing
    test_gmm()
