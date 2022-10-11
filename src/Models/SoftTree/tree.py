"""Softtree"""
import tensorflow as tf
import keras
import numpy as np
from typing import List
from Models.SoftTree.klayers import MultiDense


# forest specific loss functions


# TODO: move this somewhere...
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


class Forest(keras.layers.Layer):
    # NOTE: depth of 0 = 1 layer ~ 0 indexed

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
        # TODO: docstring;
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
        """Get tree probabilities
            > softmaxes across states

        Args:
            inputs (tf.Tensor): unnormalized
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


class ForestLinear(keras.layers.Layer):
    # Forest with linear features
    # NOTE: depth of 0 = 1 layer ~ 0 indexed

    def __init__(self, width: int = 0, depth: int = 0, num_tree: int = 0):
        super(ForestLinear, self).__init__()
        self.width = width
        self.depth = depth
        self.num_tree = num_tree
        # (1 - r^{n+1}) / (1 - r)
        num = 1 - self.width**(self.depth + 1)
        self.total_nodes = int(num / (1 - self.width))
        print(self.total_nodes)

    def build(self, input_shape):
        # TODO: how do multiple inputs get handled?
        # target output shape: batch_size x num_tree x total_nodes x width
        # input shapes: [batch_size x ...], ...
        #   num_tree, total_nodes = parallel dims
        self.lin_layers = [MultiDense([0, 1], self.width) for _ in input_shape]
        self.forest = Forest()

    def call(self, inputs: List[tf.Tensor]):
        # input shapes: [batch_size x ...], ...
        # TODO: docstring

        # eval each input indepdendenly
        v = []
        for i, inp in enumerate(inputs):
            # reshape to batch_size x num_tree x total_nodes x ...
            inp_shape = tf.shape(inp)
            new_shape = tf.concat([inp_shape[:1], tf.ones([2], dtype=inp_shape.dtype), inp_shape[1:]], axis=0)
            inp = tf.reshape(inp, new_shape)
            tile_shape = tf.concat([tf.ones([1], inp_shape.dtype),
                                    tf.constant([self.num_tree, self.total_nodes], inp_shape.dtype),
                                    tf.ones(tf.shape(inp_shape[1:]), inp_shape.dtype)],
                                   axis=0)
            inp = tf.tile(inp, tile_shape)
            v.append(self.lin_layers[i](inp))
        return self.forest(tf.math.add_n(v))


if __name__ == "__main__":
    # test out layer
    # depth = 1
    v = tf.ones([8, 5, 3, 2])
    vout = Forest()(v)
    print(vout)
    print(tf.math.reduce_sum(vout, axis=-1))
    # test with more complex size:
    # depth = 2; width = 3
    v = tf.ones([8, 5, 1 + 3 + 9, 3])
    F = Forest()
    vout = F(v)
    print(vout)
    print(tf.math.reduce_sum(vout, axis=-1))
    # super important test:
    _, next_ind = F._eval_forest(v, tf.constant(1.), 3, 3, 0)
    print(next_ind)
    assert next_ind == tf.shape(v)[2]

    # what if I use a keras tensor:
    v2 = tf.keras.Input((5, 1 + 3 + 9, 3))
    vout = Forest()(v2)
    model = tf.keras.Model(inputs=v2, outputs=vout)
    model(v)

    # testing forest linear
    v1 = tf.ones([8, 3])
    v2 = tf.ones([8, 6, 2])
    Flin = ForestLinear(3, 2, 11)
    vout = Flin([v1, v2])
    print(tf.shape(vout))
    print(tf.math.reduce_sum(vout, axis=-1))