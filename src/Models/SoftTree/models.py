"""Assembled keras model"""
import tensorflow as tf
from typing import Tuple, List
from dataclasses import dataclass
from tree import build_forest, StandardLayerFactory
from Models.SoftTree.klayers import MultiDense


def build_binary_tree_predictor(inps: List[tf.keras.Input]):
    # linear model logits --> batch_size x num_tree
    tree_dim = 0
    mv = [MultiDense([tree_dim], 1)(inpi)[:,:,0]
            for inpi in inps]
    return tf.math.add_n(mv)


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



def build_fweighted_linear_pred(inps: List[tf.keras.Input],
                                num_tree: int,
                                depth: int,
                                width: int):
    # forest weighted linear predictor

    # TODO: missing mask for loss
    # mask = batch_size x num_tree
    # ... each tree gets a subset of the input sample!!!

    layer_factories = [StandardLayerFactory(width, num_tree) for _ in inps]
    # --> batch_size x num_tree x num_state
    states = build_forest(depth, width, inps, layer_factories)
    # build parallel predictors:
    preds = build_parallel_binary_preds(inps, num_tree, depth**width)

    # TODO: if x2 supplied (as logits) --> add to preds (BOOST)

    # TODO: parallel loss

    # TODO: average prediction (RANDOM FOREST)
    # NOTE: no mask
    # > weighted sum across states > average across trees
    w_ave = tf.math.reduce_sum(states * preds, axis=2)
    pred_mu = tf.math.reduce_mean(w_ave, axis=1)
