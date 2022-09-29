"""loss funcs

NOTE: this will typically not reduce within the batch
"""

import tensorflow as tf


def binary_loss(predictions: tf.Tensor,
                y: tf.sparse.SparseTensor,
                mask: tf.Tensor = None):
    """binary cross-entropy loss using sigmoid function
    p = exp(pred) / (exp(pred) + 1)
    x-entropy = y log p + (1 - y) log (1 - p)
    > log p = pred - log(exp(pred) + 1)
    > log(1-p) = log[ exp(pred) + 1 - exp(pred) / (exp(pred) + 1)]
    = log [1 / (exp(pred) + 1)] = -log(exp(pred) + 1)
    in sum:
    y [pred - log(exp(pred) + 1)] - (1 - y)log(exp(pred) + 1)
    K = log exp(pred) + 1
    --> y pred - y K - K + yK = y pred - K
    = y pred - log(exp(pred) + 1)
    ... same as multinomial but 1 for off logit

    Args:
        predictions (tf.Tensor): prediction logits
            batch_size x [parallel_dimensions]
        y (tf.Tensor): binary/boolean tensor; truth
            shape = batch_size
        mask (tf.Tensor): scale on each datapoint
            batch_size x [parallel_dimensions]

    Returns:
        tf.Tensor: batch_size x [parallel_dimensions]
    """
    # reshape y to match shape of predictions
    yshape = tf.concat([tf.shape(predictions)[:1],
                        1 + 0 * tf.shape(predictions)[1:]],
                       axis=0)
    y = tf.reshape(y, yshape)
    # norm term = logsumexp
    K = tf.math.log(tf.math.exp(predictions) + 1)
    # primary term
    yp = tf.cast(y, predictions.dtype) * predictions
    log_like = yp - K
    if mask is None:
        return -1 * log_like
    return -1 * log_like * tf.cast(mask, log_like.dtype)


def quantile_loss(predictions: tf.Tensor,
                  y: tf.Tensor,
                  taus: tf.Tensor):
    """Quantile loss = pinball loss
    = [ (tau - 1) sum_[yi < q] (yi - q)
        + tau sum_[yi >= q] (yi - q)]
    where q = prediction
    and tau defines the quantile
        Ex: tau = .5 = 50% quantile = median

    Args:
        predictions (tf.Tensor): predictions
            batch_size x [parallel_dims] x num_quantile
        y (tf.Tensor): truths
            shape = batch_size
            NOTE: quantile_loss is only defined for
                scalar truths/predictions
        taus (tf.Tensor): quantiles
            shape = num_quantile

    Returns:
        tf.tensor: quantile loss
            batch_size x [parallel_dims] x num_taus
                only reduces across last dim
    """
    # reshape y to match shape of predictions
    yshape = tf.concat([tf.shape(predictions)[:1],
                        1 + 0 * tf.shape(predictions)[1:]],
                       axis=0)
    y = tf.reshape(y, yshape)

    # reshape taus to also match prediction shape
    taushape = tf.concat([1 + 0 * tf.shape(predictions)[:-1],
                          tf.shape(predictions)[-1:]],
                          axis=0)
    taus = tf.reshape(taus, taushape)

    # dt = (y_i - q)
    # --> batch_size x [parallel_dims] x num_quantile
    dt = y - predictions
    # left side = (tau - 1) terms
    left = ((taus - 1) *
            tf.stop_gradient(tf.cast(dt < 0, predictions.dtype)) *
            dt)
    # right side = tau terms
    right = (taus *
            tf.stop_gradient(tf.cast(dt >= 0, predictions.dtype)) *
            dt)
    return left + right
