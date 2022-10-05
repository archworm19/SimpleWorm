"""Softtree"""
import abc
import tensorflow as tf
from typing import List
from tensorflow.keras.backend import int_shape
from Models.SoftTree.klayers import MultiDense


class LayerFactory(abc.ABC):

    def func_build(self, x: tf.Tensor) -> tf.Tensor:
        """build via keras functional api

        Args:
            x (tf.keras.Input): batch_size x num_tree x ...

        Returns:
            tf.Tensor: shape = batch_size x num_tree x tree_width
            tf.keras.layers.Layer
        """
        pass

    def get_width(self) -> int:
        pass

    def get_num_trees(self) -> int:
        pass


class StandardTreeLayerFactory(LayerFactory):
    def __init__(self, width: int, num_tree: int):
        self.width = width
        self.num_tree = num_tree
        self.tree_dim = 0

    def func_build(self, x: tf.Tensor):
        # expects batch_size x 1 x d1 x ... input
        # returns: batch_size x num_tree x width
        x = tf.repeat(x, self.num_tree, 1)
        M = MultiDense([self.tree_dim], self.width)
        return M(x), M

    def get_width(self) -> int:
        return self.width

    def get_num_trees(self) -> int:
        return self.num_tree


def _build_forest_node(inps: List[tf.keras.Input],
                       layer_factories: List[LayerFactory]):
    # build layers for each input --> add the result
    # with only 0th dim parallel --> batch_size x num_tree x width
    # input = batch_size x d1 x ...
    yz = []
    for lf, inpi in zip(layer_factories, inps):
        # parallelize inp across trees:
        y, _ = lf.func_build(tf.expand_dims(inpi, 1))
        assert len(int_shape(y)) == 3
        yz.append(y)
    return tf.math.add_n(yz)


def _build_forest(weight: tf.Tensor,
                  width: int,
                  depth: int,
                  inps: List[tf.keras.Input],
                  layer_factories: List[LayerFactory]):
    # recursive helper:
    # ASSUMES: weight = batch_size x forest
    # Returns: weights from end layers
    if depth == 0:
        return [weight]
    # make the next layer --> child call for each
    # --> batch_size x num_tree x width
    v = _build_forest_node(inps, layer_factories)
    v_norm = tf.nn.softmax(v, axis=-1)
    res = []
    for i in range(width):
        res.extend(_build_forest(v_norm[:, :, i] * weight, width, depth-1, inps,
                                    layer_factories))
    return res


def build_forest(width: int,
                 depth: int,
                 inps: List[tf.keras.Input],
                 layer_factories: List[LayerFactory]):
    """build tree network

    Args:
        width (int): width of each node of the tree
            = number of outputs from each node
        depth (int): tree depth
            depth = 1 --> just the root node
        inps (List[tf.keras.Input]): all inputs

    Returns:
        tf.Tensor: batch_size x num_forest x M
            where M = sum of bottom layer widths
    """
    assert(depth > 0)
    assert(width > 1)
    for lf in layer_factories:
        assert lf.get_width() == width
    num_trees = layer_factories[0].get_num_trees()
    for lf in layer_factories[1:]:
        assert num_trees == lf.get_num_trees()
    v = _build_forest(tf.constant(1.0, dtype=inps[0].dtype),
                      width, depth, inps, layer_factories)
    return tf.stack(v, axis=2)


# forest specific loss functions


def forest_reg_loss(forest_output: tf.Tensor,
                    reg_strength: tf.Tensor):
    """Forest regularization loss
    Recommended use: make this penalty strong initially
        --> relax it in later training iterations

    Args:
        forest_output (tf.Tensor): forest output states
            assumed to be batch_size x num_forest x num_state
            but will work for ... x num_state
        reg_strength (tf.Tensor): regularization strength
            assumed to have shape batch_size

    Returns:
        tf.Tensor: scalar tensor loss
    """
    # goal = maximize entropy across states
    # --> minimize negentropy
    negent = tf.math.reduce_sum(forest_output *
                                tf.math.log(forest_output),
                                axis=-1)
    rs = tf.expand_dims(reg_strength, 1)
    # average across trees and batches
    return tf.math.reduce_mean(negent * tf.stop_gradient(rs))
