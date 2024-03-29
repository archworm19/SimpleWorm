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
        dist_mat = np.nanmean((dat[None,:] - means[:,None])**2.0, axis=2)
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
        
        Returns:
            np.ndarray: num_mean x N array of all means
        """

        num_miss = self.num_means - np.shape(new_means)[0]
        mu_shape = np.shape(new_means[0])
        # total assigned --> mean indices
        mu_inds = [[z] for z in range(len(grp_size))]
        for i in range(num_miss):
            assign_density = [grp_size[z] / len(mu_inds[z])
                                for z in range(len(grp_size))]
            # assign to highest number of pts per means
            maxg = np.argmax(assign_density)
            mu_inds[maxg].append(i + len(new_means))
        # make the final means:
        muz = []
        for i, mui in enumerate(mu_inds):
            for _ in mui:
                muz.append(new_means[i] + 1e-8 * self.rng.random(mu_shape))
        return np.array(muz)
    
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
            means = self._handle_cluster_loss(np.array(means),
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
        assert(np.fabs(np.sum(priors) - 1) < 1e-4), "priors must sum to 1"
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

    def __init__(self, num_means: int,
                 tolerance: float = 1e-4,
                 tolerance_scale: float = 1e-32,
                 gauss_tolerance: float = 1e-10):
        """Initialize weighted Gaussian Mixture Model

        Args:
            num_means (int): number of clusters/means
            tolerance (float): tolerance for each each singular
                value of the covariance matrix. Singular
                values cannot go below this value.
            tolerance_scale (float): if determinant of covariance
                matrix drops below this value --> singular
                values are scaled so that determinant = tolerance_scale
            gauss_tolerance (float): if value in gaussian
                exponential is < gauss_tolerance -->
                resulting probability treated as 0
        """
        self.num_means = num_means
        self.tolerance = tolerance
        self.tolerance_scale = tolerance_scale
        self.gauss_tolerance = gauss_tolerance
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

        # tolerance scaling:
        # det = s0 * s1 * s2 * ...
        # target det = tolerance_scale if det below
        # -->
        # tolerance_scale = s0*k * s1*k * ...
        # k = tolerance factor
        # k^dim = tolerance_scale / det

        if det < self.tolerance_scale:
            tol_scale_factor = (self.tolerance_scale / det) ** (1. / len(s))
            s = s * tol_scale_factor
            det = self.tolerance_scale

        # invert s
        # since diagonal --> elem-wise inverse
        sinv = 1. / s

        # inverse calculation
        precision = vt.T @ np.diag(sinv) @ u.T

        return precision, det
    
    def _decompose_all_covars(self, covars: np.ndarray):
        """Decompose all covariance matrices

        Args:
            covar_mats (np.ndarray): covariances matrices
                num_mean x N x N array
        
        Returns:
            np.ndarray: precision matrices (covar inverses)
                num_mean x N x N array
            np.ndarray: covariance determinants
                len num_mean array
        """
        # decompose each covariance matrix
        # --> precision and determinant simul calc
        precisions, cov_dets = [], []
        for i in range(np.shape(covars)[0]):
            prec, covd = self._decompose_covar(covars[i])
            precisions.append(prec)
            cov_dets.append(covd)
        return np.array(precisions), np.array(cov_dets)
    
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
    
        # underflow protection:
        tol_mask = m < self.gauss_tolerance
        m[tol_mask] = 0.

        # numerator --> num_sample
        num = np.exp(-.5 * m)

        # bring back tolerance
        num[tol_mask] = 0.

        # denom --> float
        denom = np.sqrt(((2. * np.pi)**dim)
                         * cov_det)

        return num / denom

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
            np.ndarray: mixing probabilities
                forward probabilities scaled by
                mixing coefficients (gaussian mags)
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
                raw_diff: np.ndarray, priors: np.ndarray):
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
            priors (np.ndarray): prior probabilities
                on datapoints = weights for datapoints
                = len num_sample array
        
        Returns:
            np.ndarray: updated mean
                num_mean x N
            np.ndarray: updated covariance
                num_mean x N x N
            np.ndarray: updated mixing coeffs
                len = num_mena
        """
        # M-Step:
        # weight creation:
        # p(X,Z) = p(Z | X)p(X)
        # = posteriors * priors
        # > raw_weights: mult posteriors by priors (num_mean x num_sample)
        # > weights: normalize
        raw_weights = post_probs * priors[None]
        weights = raw_weights / np.sum(raw_weights, axis=1,
                                       keepdims=True)
        # update mixing_coeffs:
        # sum_x [ p(X,Z) ] = p(Z)
        # = sum raw_weigths across samples
        # --> num_mean
        mix_coeffs = np.sum(raw_weights, axis=1)

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
            np.ndarray: mixing coeffs
                len num_mean array
            np.ndarray: posterior probabilities
                num_mean x num_sample array
        """
        # init with kmeans:
        # --> means = num_mean x N
        # --> dist_mat = num_mean x num_sample
        means, _dist_mat = self.km.multi_run(dat, priors, num_iter, 2)

        # update raw difference
        # calculate raw difference:
        # --> num_mean x num_sample x N
        raw_diff = dat[None] - means[:, None]

        # initial precision estimate:
        # outp = num_mean x num_sample x N x N
        outp = raw_diff[:,:,None] * raw_diff[:,:,:,None]
        # contract along samples:
        # --> num_mean x N x N
        covars = np.sum(priors[None,:,None,None] * outp,
                      axis = 1)
        
        precisions, cov_dets = self._decompose_all_covars(covars)

        # repeated iterations of 
        # > posterior probability calc
        # > means / covariance updates
        mix_coeffs = np.ones((self.num_means)) * (1. / self.num_means)
        for _ in range(num_iter):

            # posterior calc:
            posts, _, _ = self.probs(raw_diff, precisions, cov_dets,
                            mix_coeffs)

            # update --> new means/covars
            means, covars, mix_coeffs = self.update(dat, posts, raw_diff,
                                                    priors)
            # update raw difference
            # calculate raw difference:
            # --> num_mean x num_sample x N
            raw_diff = dat[None] - means[:, None]

            # decompose each covariance matrix
            # --> precision and determinant simul calc
            precisions, cov_dets = self._decompose_all_covars(covars)
        
        return means, covars, mix_coeffs, posts
    
    def log_like(self,
                 means: np.ndarray,
                 covars: np.ndarray,
                 mixing_coeffs: np.ndarray,
                 priors: np.ndarray,
                 dat: np.ndarray):
        """Weighted Log-Likelihood calculation
        Standard log-likelihood, weighted by priors

        Args:
            means (np.ndarray): means for each cluster
                num_mean x N
            covars (np.ndarray): covariance matrices
                for each cluster
                num_mean x N x N
            mixing_coeffs (np.ndarray): mixing coefficients
                = scaling coeffs for each gaussian
                len num_mean
            priors (np.ndarray): sample priors
                = sample weights
                len num_sample
            dat (np.ndarray): raw data
                num_sample x N

        Returns:
            float: log-likelihood
        """
        raw_diff = dat[None] - means[:, None]
        precisions, cov_dets = self._decompose_all_covars(covars)
        # mixing probs = num_mean x num_sample array
        _posts, mixing_probs, _fors = self.probs(raw_diff, precisions, cov_dets, mixing_coeffs)
        # log sum within each cluster:
        # --> len num_sample
        logsum = np.log(np.sum(mixing_probs, axis=0))
        # scale by sample weights for final calc:
        return np.sum(priors * logsum)
