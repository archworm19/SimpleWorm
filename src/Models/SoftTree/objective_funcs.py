""" 
    Objective Functions
    
    Components?
    > (Forward) Probability Interface
    > > Different Gaussian reps
    > > > Ex: different factorization / constraints for covariance
    > > Q? Other forward probability funcs?
    > > Q? takes in a layer? controls linear vs. offset only
    > Mixture Function (class needed?)
    > > Takes in M x N probability tensor
    > > Takes in M x N individual log-likelihoods
    > > M = input states
    > > N = 'noise' states (uncertainty within subspace)
    > > Handles Log-Likelihood calc

"""
import abc
import tensorflow as tf
import numpy as np
import numpy.random as npr
from Models.SoftTree.layers import var_construct


class log_prob_iface(abc.ABC):
    def calc_log_prob(self, x):
        """Calculate log probability

        Args:
            x (_type_): input
                batch_size x d

        Returns:
            _type_: log probabilities for each batch sample, model, and state
                batch_size x num_model x num_state
        """
        pass


def calc_mixture_loglike(x, weights, log_prob_cls: log_prob_iface):
    """Calculate mixture log-likelihood

    Weight-based Mixture Construction
    > N total states + states are independent
    > Calculate log probability within each of the states
    > Return weighted sum across log probabilities

    Args:
        x (_type_): input
            batch_size x d
        weights (_type_): weights
            batch_size x num_model x num_state
        log_prob_cls (log_prob_iface): log probability class.
            implements log_prob_iface interface

    Returns:
        _type_: mixture log-likelihoods ~ scalar

    """
    return tf.reduce_sum(weights * log_prob_cls.calc_log_prob(x))


class GaussFull(log_prob_iface):
    """Full Variance Gaussian
    
    Construction:
    > Cholesky factorization background
    > > If A is symmetric, positive-definite, it can be written as A = LL.T
    where L = lower triangular matrix
    > > We use LDL.T cholesky factorization
    > > D = diagonal matrix
    > > L = lower triangular matrix with 1s on diagonal
    > > elems of D = singular values (and eigenvalues for pos, def) of A
    > Inversion of symmetric matrix background
    > > If A is positive, definite, symmetric (and invertible) --> A^-1 is positive, definite, symmetric
    > LDL Factorization of Precision
    > > Since covar is symmetric, invertible (assumed), positive, definie --> precision is symmetric, positive, definite
    > > Thus, we factorize precision as LDL.T
    > Determinant of Covariance matrix
    > > 1. / det(precision matrix)
    > > det(precision matrix) = prod(D elems)
    
    """

    def __init__(self, num_model: int, num_state: int, dim: int, rng: npr.Generator):
        """Build structures

        Args:
            num_model (int): number of parallel models
            num_state (int): number of states for each model
            dim (int): dimensionality of Gaussian domain
        """
        self.dim = dim
        self.mu = var_construct(rng, [num_model, num_state, dim])
        # precision construction
        diag_mask = np.diag(np.ones((dim,)))
        base_tril_mask = np.tril(np.ones((dim,)))
        tril_mask = base_tril_mask * (1 - diag_mask)
        L_base = var_construct(rng, [num_model, num_state, dim, dim])
        D_base = var_construct(rng, [num_model, num_state, dim, dim])
        self.tf_diag_mask = tf.constant(diag_mask[None, None].astype(np.float32))
        # --> num_model x num_state x dim x dim; and each dim x dim matrix is LD constructed
        self.L = L_base * tf.constant(tril_mask[None, None].astype(np.float32)) + self.tf_diag_mask
        self.D = D_base
        self.Dexp = tf.exp(self.D) * self.tf_diag_mask  # constrain positive and set off diags 0
        # LD mult
        ld = self._matmul_modstate(self.L, self.Dexp)
        # LDL.T mult
        L_trans = tf.transpose(self.L, perm=[0, 1, 3, 2])
        # --> 1 x num_model x num_state x d x d
        self.LDL = tf.expand_dims(self._matmul_modstate(ld, L_trans), 0)
    
    def _matmul_modstate(self, x1, x2):
        # matmul where x1, x2 = [num_model, num_state, d, d]
        return tf.reduce_sum(tf.expand_dims(x1, 4) * tf.expand_dims(x2, 2), axis=3)

    def calc_log_prob(self, x):
        """Calculate log probability

        Args:
            x (_type_): input
                batch_size x d (dim of gaussian domain)

        Returns:
            _type_: log probabilities for each batch sample, model, and state
                batch_size x num_model x num_state
        """
        # double expand x --> batch_size x 1 x 1 x d
        x = tf.expand_dims(x, 1)
        x = tf.expand_dims(x, 1)
        # diff from mean --> batch_size x num_model x num_state x d
        di = x - tf.expand_dims(self.mu, 0)
        # right mult: prec * di --> batch_size x num_model x num_state x d
        rmul = tf.reduce_sum(self.LDL * tf.expand_dims(di, 4), axis=3)
        # exponential term:
        exp_term = -.5 * tf.reduce_sum(rmul * di, axis=3)

        # denom term:
        # log sqrt[ (2pi) ^ k * cov_det]
        # = log sqrt[ (2pi) ^ k * 1. / det(precision) ]
        # = .5 * log [ (2pi) ^ k ] + .5 * log [1 / det(precision)]
        # = .5k * log 2pi - .5 * log det(precision)
        # = .5k * log 2pi - .5 * log prod Dexp (diagonal)
        # = .5k * log 2pi - .5 * sum_i log exp(D_i)
        # = .5k * log 2pi - .5 * sum_i D_i
        denom_term = .5 * (self.dim * np.log(2. * np.pi) -
                           tf.reduce_sum(self.D * self.tf_diag_mask, axis=[2,3]))

        return exp_term - tf.expand_dims(denom_term, 0)
