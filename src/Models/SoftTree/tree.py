"""Softtree"""
import abc
import tensorflow as tf
from typing import List
from Models.SoftTree.klayers import MultiDense


class LayerFactory(abc.ABC):

    def func_build(self, x: tf.Tensor) -> tf.Tensor:
        """build via keras functional api

        Args:
            x (tf.keras.Input):

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
        # expects batch_size x d1 x ... input
        # tile to add trees in 1st dim
        # --> batch_size x num_tree x d1 x ...
        # TODO: tiling should maybe be in the tree itself
        M = MultiDense([self.tree_dim], self.width)
        xbig = tf.repeat(tf.expand_dims(x, 1), self.num_tree)
        y = M(xbig)
        return y, M

    def get_width(self) -> int:
        return self.width

    def get_num_trees(self) -> int:
        return self.num_tree


def _build_forest_node(inps: List[tf.keras.Input],
                       layer_factories: List[LayerFactory]):
    # build layers for each input --> add the result
    # with only 0th dim parallel --> batch_size x num_tree x width
    yz = []
    for lf, inpi in zip(layer_factories, inps):
        y, _ = lf.func_build(inpi)
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

    print(v)
    input('cont?')

    v_norm = tf.nn.softmax(v, axis=-1)
    res = []
    for i in range(width):
        res.extend(_build_forest(v_norm[:, :, i] * weight, depth-1, width, inps,
                                    layer_factories))
    return res


def build_forest(depth: int,
                 width: int,
                 inps: List[tf.keras.Input],
                 layer_factories: List[LayerFactory]):
    """build tree network

    Args:
        depth (int): tree depth
            depth = 1 --> just the root node
        width (int): width of each node of the tree
            = number of outputs from each node
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
                      depth, width, inps, layer_factories)
    return tf.stack(v, axis=2)
