"""Binary Clustering

    TODO: this needs a stronger mathematical basis
    --> stick to GMM for now

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
            np.ndarray: mixing log-likelihood
                forward log-likelihood scaled by mixing coeffs
                num_mean x num_sample
            np.ndarray: forward log-likelihood
                num_mean x num_sample
        """
        # forward probabilities 
        # = binary log-likelihood of each timepoint
        # Independent assumption: sum across timepoints within each mean
        x2 = x[None]  # --> 1 x num_sample x N
        mu2 = means[:,None]  # --> num_mean x 1 x N

        # full likelihood
        # num_mean x num_sample x N
        full_like = (mu2**x2) * (1. - mu2)**(1. - x2)

        # correct math
        # > prod reduce mixing_like --> num_mean x num_sample
        # > normalize across means for posteriors
        # ... Potential issue: could get underflow from reduction
        # TODO: let's do correct first --> figure out how to address underflow
        forward_like = np.prod(full_like, axis=2)

        # scale by mixing coeffs
        # --> num_mean x num_sample
        mixing_like = mixing_coeffs[:,None] * forward_like

        # calculate posteriors
        # normalize across means
        post_probs = mixing_like / np.sum(mixing_like, axis=0,
                                           keepdims=True)
        return post_probs, mixing_like, forward_like


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


    def rand_init(self, x: np.ndarray,
                  priors: np.ndarray,
                  num_mean: int):
        """Randomly initialize means and mixing coeffs
        Strategy:
        > shuffle x (indices of x ~ x remains unchanged)
        > break up shuffled x into num_mean groups
        > weighted means

        Args:
            x (np.ndarray): sample data
                num_sample x N
            priors (np.ndarray): data prior probabilities
                len num_sample
            num_mean (int): number of means
        
        Returns:
            np.ndarray: initial estimate of means
                num_mean x N
        """
        # shuffle:
        inds = [i for i in range(np.shape(x)[0])]
        self.rng.shuffle(inds)
        block_size = int(np.floor(len(inds) / num_mean))
        means = []
        for z in num_mean:
            zinds = inds[z*block_size:(z+1)*block_size]
            pz = priors[zinds]
            xz = x[zinds]
            mu = np.sum(pz[:,None] * xz, axis=0)
            means.append(mu)
        mixing_coeffs = np.ones((num_mean)) * (1. / num_mean)
        return np.array(means), mixing_coeffs


    def run(self, x: np.ndarray,
                  priors: np.ndarray,
                  num_mean: int,
                  num_iter: int = 5):
        """Single run
        > random init
        > repeated iters of
        > > probs
        > > updates 

        Args:
            x (np.ndarray): sample data
                num_sample x N
            priors (np.ndarray): data prior probabilities
                len num_sample
            num_mean (int): number of means
        
        Returns:
            TODO
        """
        means, mixing_coeffs = self.rand_init(x, priors, self.num_mean)
        post_probs, forward_probs = None, None
        for i in range(num_iter):
            post_probs, _, forward_probs = self.probs(x,
                                                      means,
                                                      mixing_coeffs)
            means, mixing_coeffs = self.update(x, post_probs, priors)
        # TODO: is this return signature right?
        return means, mixing_coeffs, post_probs, forward_probs
  