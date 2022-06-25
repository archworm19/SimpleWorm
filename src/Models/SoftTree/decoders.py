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


class BayesianGauss(log_prob_iface):
    # TODO: docstring
    # implement derivations here: https://towardsdatascience.com/variational-inference-in-bayesian-multivariate-gaussian-mixture-model-41c8cc4d82d7
    # Simpler version = only means will be generated in Bayesian style
    # Overall Design: 
    # > use q distro for mean generating distro and categorical distros
    # > > assume factorizable --> q(c), q(mu)
    # > > Generating Dsitros
    # > > > mu_k ~ N(alpha, sigma^2 I)
    # > > > c_i ~ Categorical(1 / K, ..., 1 / K); K = number of gaussians
    # > > > x_i | c_i, mu ~ N(c_i.T mu, lambda^2 I); c_i = one-hot
    # Variational inference approach
    # > ideally, minimize KL divergence between q and p
    # > since this is not tractable --> maximize ELBO
    # > ELBO = L(x | m, s^2, phi)
    #        = E_q [ log p(x, mu, c) ] - E_q[ log q(mu, x)]
    # Interpretation of Bayesian
    # = q distro approximates posterior
    # --> get splitting by reinforcing 'good guesses' of q

    # I think:
    # p = dim of gaussians
    # K = number of gaussians

    # Q? can I use tfp?
    # https://medium.com/analytics-vidhya/gaussian-mixture-models-with-tensorflow-probability-125315891c22
    # https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MixtureSameFamily
    # use probflow? https://probflow.readthedocs.io/en/latest/examples/gmm.html
    # raw tensorflow: https://brendanhasz.github.io/2019/06/12/tfp-gmm.html#model
    # > these are just sampling-based approaches...


    def _elbo_term1():
        """term1 of ELBO computation
        sum_k [ E_q [ log p(mu_k)]]
        derive terms =
        > sub gaussian distro for p
        > apply log rules to gaussian (gets rid of exponential)
        > factor terms not dependent on mu out of sum (mult by K = num gauss)
        > within the sum --> split the terms into 2 integrals and distribute q(mu_k)
        > > split strategy is key to making the definite integrals tractable
        > integrate the gaussians over their domain
        > > strat?
        > > try to get terms to be approx derivative of gaussian
        > > Ex: if can make term = -2ax exp (-ax^2), indef igral = exp(-ax^2)
        > > one of the grals integrates to exp(-alpha * mu^2)
        > > > integrating over whole domain? (-inf, inf for all dims)
        > > > = umm, isn't this term just zero?
        > > first gral over mu_k.T mu_k q(mu_k) dmu_k
        > > > 1d approach: factor x^2 exp(-ax^2) into (-1 / 2a) x (-2ax exp(-ax^2))
        > > > then: integrate by parts --> sqrt(pi) / (2 a ^ (3/2))
        > > > bit different here
        > > > 1d  here = [mu^2 / Z] exp((mu - m)sigma^2 ^ -1 (mu - m))
        > > > end up with (m^2 + sigma^2) erf[over domain]
        > > > end up with (m^2 + sigma^2) sqrt(2 pi sigma^2) as definite integral 
        > > > pretty much the same for 2d

        Returns:
            _type_: _description_
        """



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
