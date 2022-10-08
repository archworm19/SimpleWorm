"""Softtree"""
import abc
import tensorflow as tf
import keras
import numpy as np
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


# TODO: design for making forest into custom keras layer
# > idea 1: takes in tensors ~ don't have to bother with factory
# > > what shape? batch_size x num_tree x total_nodes x width
# ... total_nodes = width^depth --> depth = log(total_node) / log(width)
class Forest(keras.layers.Layer):

    def __init__(self):
        super(Forest, self).__init__()

    def _depth_calc(self, input_shape):
        [batch_size, num_tree, total_nodes, width] = input_shape
        depth_num = np.log(1 - total_nodes * (1 - width))
        depth_denom = np.log(width)
        depth = int(depth_num / depth_denom - 1)
        tot_recon = sum([width**z for z in range(depth + 1)])
        assert tot_recon == total_nodes
        return depth

    def build(self, input_shape):
        # TODO: docstring
        # assumes shape = batch_size x num_tree x total_nodes x width
        # total_nodes = geometric series
        #   = sum_{k=0}^{n}r^k = (1 - r^{n+1}) / (1 - r)
        #   [total_nodes] * (1 - r) = (1 - r^{n+1})
        #   r^{n+1} = 1 - [total_nodes] * (1 - r)
        #   n+1 = log(1 - [total_nodes] * (1 - r)) / log(r)
        # ... r = width
        depth = self._depth_calc(input_shape)
        self.depth = depth
        self.width = input_shape[-1]
        self.init_weight = tf.constant(1.)

    def _eval_forest(self,
                     inp_tensor: tf.Tensor,
                     weight: tf.Tensor,
                     depth: int,
                     width: int,
                     inp_ind: int):
        # TODO: docstring
        # weight = batch_size x num_tree
        # inp_tensor = batch_size x num_tree x total_nodes x width

        if depth == 0:
            return [weight], inp_ind
        print("inp ind " + str(inp_ind))
        res = []
        # pull and softmax current layer
        # --> batch_size x num_tree x width
        v_norm = tf.nn.softmax(inp_tensor[:, :, inp_ind, :], axis=-1)
        res = []
        next_ind = inp_ind + 1
        for i in range(width):
            r, next_ind = self._eval_forest(inp_tensor,
                                            v_norm[:, :, i] * weight,
                                            depth - 1, width,
                                            next_ind)
            res.extend(r)
        return res, next_ind

    def call(self, inputs):
        """Get tree probabilities ~ softmaxed across states

        Args:
            inputs (tf.Tensor):
                batch_size x num_tree x total_nodes x width

        Returns:
            tf.Tensor:
                batch_size x num_tree x output_states
                output_states = width**depth
        """
        v, _ = self._eval_forest(inputs,
                                 tf.cast(self.init_weight, dtype=inputs.dtype),
                                 self.depth + 1, self.width, 0)
        return tf.stack(v, axis=-1)


if __name__ == "__main__":
    # test out layer
    v = tf.ones([8, 5, 3, 2])
    vout = Forest()(v)
    print(vout)
    print(tf.math.reduce_sum(vout, axis=-1))
