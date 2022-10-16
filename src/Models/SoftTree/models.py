"""Combines Models and Losses ~ is this really necessary?
    OR: should assembly be done here as well? probably"""
import tensorflow as tf
from typing import List
from Models.SoftTree.tree import (ForestLinear, forest_reg_loss)
from Models.SoftTree.klayers import MultiDense
from Models.SoftTree.loss_funcs import binary_loss


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


def build_linear_binary_forest(inps: List[tf.Tensor],
                               target: tf.Tensor,
                               sample_mask: tf.Tensor,
                               forest_reg_penalty: tf.Tensor,
                               num_tree: int,
                               depth: int,
                               width: int,
                               x2: tf.Tensor = None):
    """forest weighted linear predictor
    NOTE: boosting is done in logit space while averaging
        is done in probability space
    NOTE: inps, target, sample_mask, forest_reg_penalty
        can be tensors or keras inputs

    Args:
        inps (List[tf.Tensor]): model inputs
            each input should be batch_size x d1 x ...
        target (tf.Tensor): model target
            binary tensor
            shape = batch_size
        sample_mask (tf.Tensor): mask on sample-tree combos
            = enforces random forest = training diff trees on diff samples
            = batch_size x num_tree
        forest_reg_penalty (tf.Tensor): forest regularization penalty
            Recommendation: make large early in training --> relax
            == discourages early bad solns
        num_tree (int): number of tree
        depth (int): tree depth
        width (int): tree_width
            number of states = width**depth
        x2 (tf.Tensor): if supplied --> tensor is added to model
            logits = implements boosting
            shape = batch_size

    Returns:
        useful outputs
    """
    num_state = int(width**(depth+1))
    # --> batch_size x num_tree x num_state
    x_states = ForestLinear(width, depth, num_tree)(inps)
    all_pred = []
    for v in inps:
        # v transformation
        # --> batch_size x num_state x ...
        v = _expand_and_tile(v, 1, num_state)
        # --> batch_size x num_tree x num_state x ...
        v = _expand_and_tile(v, 1, num_tree)
        # --> batch_size x num_tree x num_state x 1 (cuz binary)
        all_pred.append(MultiDense([0, 1], 1)(v))
    # --> batch_size x num_tree x num_state
    x_pred = tf.add_n(all_pred)[:, :, :, 0]
    # loss: fit each tree and each state individually:
    if x2 is not None:
        # BOOST ~ in logit space
        y_pred_logit_parallel = tf.reshape(x2, [-1, 1, 1]) + x_pred
    else:
        y_pred_logit_parallel = x_pred
    bin_loss = binary_loss(y_pred_logit_parallel, target,
                           x_states * tf.expand_dims(sample_mask, 2))
    # mean prediction: batch_size
    # average ~ in probability space
    # reduce sum across states
    ave_pred_tree = tf.math.reduce_sum(x_states * tf.expand_dims(sample_mask, 2) *
                                       tf.math.sigmoid(y_pred_logit_parallel),
                                       axis=2)
    ave_pred = tf.math.reduce_mean(ave_pred_tree, axis=1)

    print("TESTING: shape")
    print(tf.shape(x_pred))
    print(tf.shape(y_pred_logit_parallel))
    print(tf.shape(ave_pred_tree))
    print(tf.shape(ave_pred))

    return bin_loss, ave_pred, y_pred_logit_parallel


if __name__ == "__main__":
    batch_size = 12
    num_tree = 8
    depth = 1
    width = 2
    inps = [tf.ones([batch_size, 5, 3])]
    target = tf.ones(batch_size)
    sample_mask = tf.ones([batch_size, num_tree])
    forest_reg_penalty = 0.0
    build_linear_binary_forest(inps,
                               target,
                               sample_mask,
                               forest_reg_penalty,
                               num_tree,
                               depth,
                               width,
                               x2=None)