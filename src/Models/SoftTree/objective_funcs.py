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
import tensorflow as tf

# TODO: I'm not totally sure about this...
# TODO: I think this is probs right... but gotta dub check
# REMEMBER: ELBO
# we want to maximize the evidence lower bound (spiritually similar)
# = L(Q) in the wiki page
# L(Q) = -E_Q [log Q(Z) - log p(Z,X)]
# = sum_Z Q(Z) [log Q(Z) - log p(Z,X)]
# Q(Z) = the state generation distro = M x N probs (soft tree and mixture probs)
# log P(Z,X) = log p(X | Z) p(Z) = log p(X | Z) + log p(Z)
# We'll probably assume the every Z is equally probable --> wipe out that term
# Q? do we get a posterior like thing?
# Don't need posterior
# Just need the Q(Z) term mult by forward probs --> gets us the positive feedback loop required
# = Q(Z) will learn to favor the states that maximize p(X | Z) !!!
# WELL: Q(Z) is approximating posterior ~ so, yeah, we're effectively doing posterior thing

# Assuming we nix p(Z) term (just for simplification now)
# -->
# max: -sum_Z [ Q(Z) log Q(Z) - Q(Z) log p(X | Z)
# max: sum_z [ Q(Z) log p(X | Z) - Q(Z) log Q(Z) ]
# left term is the posterior-like term that makes everything work (reinforcing states that do well!)
# right term? entropy term == have to consider p(Z) here....


def compute_elbo(forward_probs, q_probs, prior_probs):
    """Calculate mixture log likelihoods
    > M = number of input states
    > > Ex: dividing neural space into M neighborhoods
    > N = 'noise' states
    > > Ex: in each of these spaces, there is decoding uncertainty
    > > > decoding uncertainty can be capture by N clusters

    Args:
        forward_probs (_type_): forward probabilities for each of
            the batch_size x num_models x M x N states
            obtained by compare predictions with truths
        q_probs (_type_): Q(Z) = approximation of posterior p(Z | X)
            the batch_size x num_models x M x N states
            will typically include probabilities that each
                of the samples is in a given input state
                and the mixing probabilities (outer product, typically)
        prior_probs (_type_): p(Z) = prior probability on all Z
            batch_size x num_models x M x N
    
    Returns:
        (_type_): evidence lower bound each model / sample
            = log (forward probabilities) scaled by the posterior
            batch_size x num_models
    """
