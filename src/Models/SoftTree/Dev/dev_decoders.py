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

    # also useful:
    # chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://cs229.stanford.edu/section/more_on_gaussians.pdf
    # > 

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
