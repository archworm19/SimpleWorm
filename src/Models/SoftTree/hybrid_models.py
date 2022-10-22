"""Hybrid Models
    > Combine >1 Models"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from typing import Callable, List
from Models.SoftTree.klayers import MultiDense
from Models.SoftTree.tree import ForestLinear


def _expand_and_tile(v: tf.Tensor,
                     expand_dim: int,
                     num_tile: int):
    v_shape = tf.shape(v)
    base_tile = tf.ones(tf.shape(v_shape), dtype=v_shape.dtype)
    v2_tile = tf.concat([base_tile[:expand_dim],
                         tf.constant([num_tile], dtype=v_shape.dtype),
                         base_tile[expand_dim:]], axis=0)
    v2 = tf.expand_dims(v, expand_dim)
    return tf.tile(v2, v2_tile)


class LinearScalarForest(Layer):
    # Scalar = outputs a scalar for each batch, tree

    def __init__(self,
                 num_tree: int,
                 depth: int,
                 width: int,
                 transform_func: Callable = tf.math.sigmoid):
        """
        Args:
            num_tree (int):
            depth (int):
            width (int):
            transform_func (Callable, optional): how to transform preds
                before averaging across trees. Defaults to tf.math.sigmoid.
        """
        super(LinearScalarForest, self).__init__()
        self.num_tree = num_tree
        self.depth = depth
        self.width = width
        self.num_state = int(width**(depth+1))
        self.transform_func = transform_func

        # build components:
        self.FL = ForestLinear(width, depth, num_tree)

    def build(self, input: List[tf.Tensor]):
        self.LWs = [MultiDense([0, 1], 1) for _ in range(len(input))]

    def call(self, input: List[tf.Tensor], x2: tf.Tensor = None):
        """
        NOTE: boosting is done in logit space while averaging
            is done in probability space
        NOTE: inps, target, sample_mask, forest_reg_penalty
            can be tensors or keras inputs

        input (List[tf.Tensor]): model inputs
            each input should be batch_size x d1 x ...
        x2 (tf.Tensor): boosting value = added to all trees
            in logit space
            shape = batch_size

        Returns:
            tf.Tensor: parallel predictions in logit space
                for each tree, output state combo
                batch_size x num_tree x num_state
            tf.Tensor: tree/state averaged prediction
                shape = batch_size
        """
        # --> batch_size x num_tree x num_state
        x_states = self.FL(input)
        all_pred = []
        for v, lay in zip(input, self.LWs):
            # v transformation
            # --> batch_size x num_state x ...
            v = _expand_and_tile(v, 1, self.num_state)
            # --> batch_size x num_tree x num_state x ...
            v = _expand_and_tile(v, 1, self.num_tree)
            # --> batch_size x num_tree x num_state x 1 (cuz binary)
            all_pred.append(lay(v))
        # --> batch_size x num_tree x num_state
        x_pred = tf.add_n(all_pred)[:, :, :, 0]
        # loss: fit each tree and each state individually:
        if x2 is not None:
            # BOOST ~ in logit space
            y_pred_logit_parallel = tf.reshape(x2, [-1, 1, 1]) + x_pred
        else:
            y_pred_logit_parallel = x_pred
        # mean prediction: batch_size
        # average ~ in probability space
        # reduce sum across states
        ave_pred_tree = tf.math.reduce_sum(x_states *
                                           self.transform_func(y_pred_logit_parallel),
                                           axis=2)
        ave_pred = tf.math.reduce_mean(ave_pred_tree, axis=1)
        return y_pred_logit_parallel, ave_pred
