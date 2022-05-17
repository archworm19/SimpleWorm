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
from Models.SoftTree.decoders import log_prob_iface


def mixture_log_like(x, log_prob: log_prob_iface,
                     data_weights, state_weights, mixture_weights):
    """Loss function

    Args:
        x (tf.tensor): target / truth
            batch_size x dim
            dim should match space of log_prob class
        log_prob (log_prob_iface): class that generates log probabilities
            have different params for each model, state, mixture
        data_weights (tf.tensor): weights for each
            model, sample combination
            batch_size x num_model
        state_weights (tf.tensor): weights for each model state
                batch_size x num_models x num_states
        mixture_weights (tf.tensor): weights for each mixture within each state
                batch_size x num_models x num_states x num_mix
    """
    # --> batch_size x num_model x num_state x 1
    state_weights = tf.expand_dims(state_weights, 3)
    # --> batch_size x num_model x (num_state * num_mix)
    raw_probs = log_prob.calc_log_prob(x)
    # --> batch_size x num_model x num_state x num_mix
    raw_probs = tf.reshape(raw_probs, tf.shape(mixture_weights))

    # --> batch_size x num_model x 1 x 1
    data_weights = tf.expand_dims(data_weights, 2)
    data_weights = tf.expand_dims(data_weights, 2)

    # scale, reduce + negate (negate turns to loss)
    return -1 * tf.reduce_sum(data_weights * mixture_weights * 
                                state_weights * raw_probs,
                                axis = [2,3])
