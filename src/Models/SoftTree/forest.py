"""SoftForest"""
import tensorflow as tf
from typing import List
from klayers import MultiDense


def _build_forest_node(width: int,
                       inps: List[tf.keras.Input]):
    # build layers for each input --> add the result
    # with only 0th dim parallel --> batch_size x width
    mv = [MultiDense([0], width)(inp) for inp in inps]
    return tf.math.add_n(mv)


def _build_forest(weight: tf.Tensor,
                  depth: int,
                  width: int,
                  inps: List[tf.keras.Input]):
    # recursive helper:
    # ASSUMES: weight = batch_size
    # Returns: weights from end layers
    if depth == 0:
        return [weight]
    # make the next layer --> child call for each
    # --> batch_size x width
    v = _build_forest_node(width, inps)
    v_norm = tf.nn.softmax(v, axis=-1)
    res = []
    for i in range(len(width)):
        res.extend(_build_forest(v_norm[:,i] * weight, depth-1, width, inps))
    return res


def build_forest(depth: int,
                 width: int,
                 inps: List[tf.keras.Input]):
    """build forest network

    Args:
        depth (int): forest depth
            depth = 1 --> just the root node
        width (int): width of each node of the tree
            = number of outputs from each node
        inps (List[tf.keras.Input]): all inputs

    Returns:
        tf.Tensor: batch_size x M
            where M = sum of bottom layer widths
    """
    assert(depth > 0)
    assert(width > 1)
    v = _build_forest(tf.constant(1.0, dtype=inps[0].dtype),
                      depth, width, inps)
    return tf.concat(v, axis=1)


# TODO: we need the following network aggregators:
# ... don't think we need aggregation funcs
# > just do these ops in the model wirings

# 1. sum/boosting: add model logits together
# ... why? allows you to perform alignment analysis
#
# 2. concatenation: concat tensor of multiple models together
# ... why? will be necessary for parallel model fits
# 
# 3. averaging predictions 