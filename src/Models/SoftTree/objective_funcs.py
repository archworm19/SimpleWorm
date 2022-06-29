""" 
    Useful objective functions
    > those defined here will be used by assembled_models

"""
from typing import List
import tensorflow as tf
from Models.SoftTree.layers import LayerIface
from Models.SoftTree.decoders import log_prob_iface


# TODO: should maybe have additional helper function that calcs forest state
def forest_loss(forest_eval, data_weights):
    """Forest evaluation Loss
    Designed to force the forest outputs to occupy all states
    equally on average

    > Each model outputs N states = N leaves
    > > for M batch_size
    > weighted average across batch, within model
    > > weights = data weights
    > > Gets us N-len vector for each model (vals gauranteed between 0, 1)
    > Calc entropy for each model
    > Return neg-entropy as loss --> maximize entropy

    Args:
        forest_eval (tf.tensor): output of forest
            batch_size x num_model x num_leaves
        data_weights (tf.tensor): weights on the data points
            batch_size x num_model

    Returns:
        tf.tensor: prediction losses for given batch + each model
            num_model
    """
    # reshape for legal mult:
    dw = tf.expand_dims(data_weights, 2)
    # weighted average across batch:
    # --> num_model x num_leaves
    wave = tf.math.divide(tf.reduce_sum(forest_eval * dw, axis=0),
                            tf.reduce_sum(dw, axis=0))
    # negentropy across state/leaves
    # --> num_model
    negent = tf.reduce_sum(wave * tf.math.log(wave), axis=1)
    # scale by batch_size to make it batch_size dependent
    return negent * tf.cast(tf.shape(forest_eval)[0], negent.dtype)


def spread_loss(ref_layers: List[LayerIface]):
    """Spread penalty for model layers

    Returns:
        tf.tensor scalar: spread error summed across all layers

    """
    sp_errs = [l.spread_error() for l in ref_layers]
    return tf.add_n(sp_errs)