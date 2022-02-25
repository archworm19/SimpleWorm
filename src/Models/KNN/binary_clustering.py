"""Binary Clustering

    Example Use Case:
    > have some signal u(t)
    > u(t) is in {0, 1}
    > BUT: u(t:t+tau) can have a bunch of different patterns
    > > Differ in the timing of their different binary values

    Approach:
    > Mixture Modeling
    > Difference from GMM
    > > Forward Likelihood = p( u(t) | z(t) ) * …
    > > = prod [ p( u(t), mu_i ) ^ z_i(t) ]
    > > p( u(t), mu_i) = mu_i,tau ^ u(t) * (1 - mu_i,tau) ^ (1 - u(t))

    Independent assumption ~ also made in GMM
    > u(t+tau) is statistically independent from u(t) given the model/means

    Mean construction assumption:
    > means assumed to be probability that sample
    > is 1 at given timepoint = p ( u(t) == 1)    
"""
import numpy as np
import numpy.random as npr

class wBMM:
    """Weighted Binary Mixture Model"""
    def __init__(self, num_means: int, rand_seed: int = 42):
        self.num_means = num_means
        self.rng = npr.default_rng(rand_seed)
    
    def probs(self,
              x: np.ndarray,
              means: np.ndarray,
              mixing_coeffs: np.ndarray):
        """Calculate all important probabilities

        Args:
            x (np.ndarray): sample data
                num_sample x N
            means (np.ndarray): current estimate of means
                means assumed to be probability that sample
                is 1 at given timepoint = p ( u(t) == 1)
                num_mean x N
            mixing_coeffs (np.ndarray): current estimate of
                mixing coeffs 
                len num_mean array

        Returns:
            np.ndarray: posterior probabilities
                normalized across means (within sample)
                num_mean x num_sample
            np.ndarray: mixing probabilities
                forward probabilities scaled by mixing coeffs
                num_mean x num_sample
            np.ndarray: forward probabilities
                num_mean x num_sample
        """
        # forward probabilities 
        # = binary log-likelihood of each timepoint
        # Independent assumption: sum across timepoints within each mean
        x2 = x[None]  # --> 1 x num_sample x N
        mu2 = means[:,None]  # --> num_mean x 1 x N
        full_ll = x2 * np.log(mu2) + (1. - x2) * np.log(1. - mu2)
        # reduce across N --> num_mean x num_sample
        forward_probs = np.sum(full_ll, axis=2)

        # scale by mixing coefficients
        mixing_probs = mixing_coeffs[:,None] * forward_probs

        # calculate posteriors
        # normalize across means
        post_probs = mixing_probs / np.sum(mixing_probs, axis=0,
                                           keepdims=True)
        return post_probs, mixing_probs, forward_probs

    def update(self,
               x: np.ndarray,
               post_probs: np.ndarray,
               priors: np.ndarray):
        """Update means and mixing coefficients

        Args:
            x (np.ndarray): sample data
                num_sample x N
            post_probs (np.ndarray): posterior probabilities
                num_mean x num_sample
            priors (np.ndarray): prior data probabilities
                weights on each datapoint 
                used to couple to KNN
                len num_sample array
        
        Returns:
            np.ndarray: updated means
                num_mean x N
            np.ndarray: updated mixing coeffs
                len num_mean
        """
        # gamma calculation
        # scale posterior probabilties by the data priors
        # --> num_mean x num_sample
        gamma = post_probs * priors[None]

        # weights = normalize gamma values across samples
        weights = gamma / np.sum(gamma, axis=1, keepdims=True)

        # update mixing_coeffs:
        # sum_x [ p(X,Z) ] = p(Z)
        # = sum gamma across samples
        # --> num_mean
        mix_coeffs = np.sum(gamma, axis=1)

        # update means:
        # --> num_mean x N
        means = np.sum(weights[:,:,None] * x[None],
                        axis=1)
        
        return means, mix_coeffs

    # TODO: random initialization

    # TODO: run 
    # > initialize
    # > repeated loop of 1. probs; 2. update