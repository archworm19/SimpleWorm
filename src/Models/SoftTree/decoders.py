"""Decoding layers"""
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

    def get_num_states(self):
        """Get the number of states

        Returns:
            int: number of states
        """
        pass

    def get_trainable_weights(self):
        """Get the trainable weights for optimization
        
        Returns:
            List[tf.tensor]
        """
        pass


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
        self.num_state = num_state
        self.dim = dim
        self.mu = var_construct(rng, [num_model, num_state, dim])
        # precision construction
        diag_mask = np.diag(np.ones((dim,)))
        base_tril_mask = np.tril(np.ones((dim,)))
        tril_mask = base_tril_mask * (1 - diag_mask)
        self.L_base = var_construct(rng, [num_model, num_state, dim, dim])
        self.D_base = var_construct(rng, [num_model, num_state, dim, dim])
        self.tf_diag_mask = tf.constant(diag_mask[None, None].astype(np.float32))
        self.tril_mask = tf.constant(tril_mask[None, None].astype(np.float32))
    
    def _matmul_modstate(self, x1, x2):
        # matmul where x1, x2 = [num_model, num_state, d, d]
        return tf.reduce_sum(tf.expand_dims(x1, 4) * tf.expand_dims(x2, 2), axis=3)

    def _get_LDL_comps(self):
        # --> num_model x num_state x dim x dim; and each dim x dim matrix is LD constructed
        L = self.L_base * self.tril_mask + self.tf_diag_mask
        Dexp = tf.exp(self.D_base) * self.tf_diag_mask  # constrain positive and set off diags 0
        # LD mult
        ld = self._matmul_modstate(L, Dexp)
        # LDL.T mult
        L_trans = tf.transpose(L, perm=[0, 1, 3, 2])
        return L, Dexp, ld, L_trans

    def _get_LDL(self):
        """Have to get LDL matrix on demand to work with tensorflow eager execution"""
        L, Dexp, ld, L_trans = self._get_LDL_comps()
        # --> 1 x num_model x num_state x d x d
        LDL = tf.expand_dims(self._matmul_modstate(ld, L_trans), 0)
        return LDL

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
        rmul = tf.reduce_sum(self._get_LDL() * tf.expand_dims(di, 4), axis=3)
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
                           tf.reduce_sum(self.D_base * self.tf_diag_mask, axis=[2,3]))

        return exp_term - tf.expand_dims(denom_term, 0)

    def get_num_states(self):
        return self.num_state

    # TODO: need test case for this
    def get_trainable_weights(self):
        """Get the trainable weights for optimization
        
        Returns:
            List[tf.tensor]
        """
        # means + the two covariance components
        return [self.mu, self.L_base, self.D_base]
