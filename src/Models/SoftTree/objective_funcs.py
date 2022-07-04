""" 
    Useful objective functions
    > those defined here will be used by assembled_models

"""
import abc
from typing import List
import tensorflow as tf
from Models.SoftTree.layers import LayerIface
from Models.SoftTree.decoders import log_prob_iface


class ObjFunc(abc.ABC):
    # rely on specific model types to reduce across samples
    def loss_sample(self, predictions, y):
        """loss for each model, sample combination

        Args:
            predictions (tf.tensor): predictions
            y (tf.tensor): target

        Returns:
            tf.tensor: batch_size x num_model
        """
        pass


# TODO: binary classification loss

class MultinomialLoss(ObjFunc):

    def __init__(self, num_class: int):
        self.num_class = num_class

    def loss_sample(self, predictions, y):
        """multinomial loss for each sample, model combination
        Multinomial loss = 
            sum_i [ y_i * log pred_i]
            where y = one-hot vector
        Softmax construction:
            pred_i_prob = exp(pred_i) / sum[ exp(pred_j) ]
            log pred_i_prob = pred_i - log sum[ exp(pred_j) ]
             = pred_i - k(x)
            ... sub into multinomial -->
            sum_i [ y_i * [pred_i - k(x)]]
            = sum_i[ y_i * pred_i] - k(x)
            ... assuming y_i is one-hot vector

        Args:
            predictions (tf.tensor): predictions logits
                batch_size x num_model x 
                TODO: should we include num_state here? probs
            y (tf.tensor): truths
                programmed as one-hot vector

        Returns:
            tf.tensor: batch_size x num_model
        """
        # TODO: finish


class QuantileLoss(ObjFunc):

    def __init__(self, taus):
        """
        taus (tf.tensor): taus define the quantile
            num_quantile
        """
        self.taus = taus

    def loss_sample(self, predictions, y):
        """Quantile loss = pinball loss
        = [ (tau - 1) sum_[yi < q] (yi - q)
            + tau sum_[yi >= q] (yi - q)]
        where q = prediction
        and tau defines the quantile
            Ex: tau = .5 = 50% quantile = median

        Args:
            predictions (tf.tensor): predictions
                batch_size x num_model x num_quantile
            y (tf.tensor): truths
                batch_size
                NOTE: quantile_loss is only defined for
                    scalar truths/predictions

        Returns:
            tf.tensor: batch_size x num_model
        """
        # dt = (y_i - q)
        # --> batch_size x num_model x num_quantile
        dt = tf.reshape(y, [-1, 1, 1]) - predictions

        # reshape taus:
        re_taus = tf.reshape(self.taus, [1, 1, -1])

        # left side = (tau - 1) terms
        left = ((re_taus - 1) *
                tf.stop_gradient(dt < 0) *
                dt)

        # right side = tau terms
        right = (re_taus *
                tf.stop_gradient(dt >= 0) *
                dt)

        return tf.reduce_sum(left + right, axis=-1)


# TODO: move this to SoftForest --> make it fulfill model interfaces
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
