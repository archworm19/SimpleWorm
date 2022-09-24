"""loss funcs

NOTE: this will typically not reduce within the batch
"""

import tensorflow as tf


def binary_loss(predictions: tf.Tensor,
                y: tf.sparse.SparseTensor,
                mask: tf.Tensor):
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
        y (tf.Tensor): binary/boolean tensor
            batch_size x [parallel_dimensions]
            or batch_size x [1] * number of parallel dims
        mask (tf.Tensor): scale on each datapoint
            batch_size x [parallel_dimensions]

    Returns:
        tf.Tensor: batch_size x [parallel_dimensions]
    """
    # norm term = logsumexp
    K = tf.math.log(tf.math.exp(predictions) + 1)
    # primary term
    yp = y * predictions
    log_like = yp - K
    return -1 * log_like * mask
