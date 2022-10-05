"""Assembled keras model"""
import tensorflow as tf
from typing import List
from Models.SoftTree.tree import (build_forest, StandardTreeLayerFactory,
                                  forest_reg_loss)
from Models.SoftTree.klayers import MultiDense
from Models.SoftTree.loss_funcs import binary_loss


def build_parallel_binary_preds(inps: List[tf.keras.Input],
                                num_tree: int,
                                num_state: int):
    # inps = batch_size x d1 x ...
    # returns: logits ~ batch_size x num_tree x num_state
    yz = []
    for inpi in inps:
        x = tf.expand_dims(inpi, 1)
        x = tf.expand_dims(x, 1)
        x = tf.repeat(x, num_tree, 1)
        x = tf.repeat(x, num_state, 2)
        yz.append(MultiDense([0, 1], 1)(x))
    return tf.math.add_n(yz)[:, :, :, 0]


# TODO: what about regularization penalty???
def build_fweighted_linear_pred(inps: List[tf.Tensor],
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
        Dict[str, tf.Tensor]: named model outputs --> package into keras model
    """
    layer_factories = [StandardTreeLayerFactory(width, num_tree) for _ in inps]
    # --> batch_size x num_tree x num_state
    states = build_forest(width, depth, inps, layer_factories)
    # build parallel predictors:
    preds = build_parallel_binary_preds(inps, num_tree, width**depth)

    # if x2 supplied (as logits) --> add to preds (BOOST)
    if x2 is not None:
        preds = preds + tf.reshape(x2, [-1, 1, 1])

    # parallel loss
    # > loss computed for each tree separately
    parallel_loss = binary_loss(preds, tf.reshape(target, [-1, 1, 1]),
                                states * tf.expand_dims(sample_mask, 2))
    red_parallel_loss = tf.math.reduce_mean(tf.math.reduce_sum(parallel_loss, axis=2))
    f_loss = forest_reg_loss(states, forest_reg_penalty)

    # Taverage prediction (RANDOM FOREST)
    # NOTE: no mask
    # > weighted sum across states > average across trees
    prob_preds = tf.math.sigmoid(preds)
    w_ave = tf.math.reduce_sum(states * prob_preds, axis=2)
    pred_mu = tf.math.reduce_mean(w_ave, axis=1)

    # reduce sum error across states --> average across trees and batch
    # ... cuz scaled by vector that is normalized across states
    return {"loss": red_parallel_loss + f_loss,
            "logit_predictions": preds,
            "predictions": prob_preds,
            "average_predictions": pred_mu}
