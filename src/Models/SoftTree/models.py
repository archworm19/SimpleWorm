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


def build_fweighted_linear_pred(inps: List[tf.keras.Input],
                                num_tree: int,
                                depth: int,
                                width: int):
    # forest weighted linear predictor

    # TODO: add in optional x2 tensor argument?

    layer_factories = [StandardLayerFactory(width) for _ in inps]
    # --> batch_size x num_tree x num_state
    states = build_forest(depth, width, inps, layer_factories)
    # build a linear predictor for each state/tree
    losses, wpred = [], []
    for z in int(depth**width):
        # TODO: can we use binary cross entropy loss?

        # TODO: averaging in the logit domain... does that make sense?
        # add state-weighted prediction
        # shape = batch_size x num_tree
        wpred.append(states[:,:,z] * build_binary_tree_predictor(inps))

    # mean prediction across trees (random forest) --> batch_size
    # NOTE: in logit space TODO: DOCSTRING
    sum_pred = tf.math.add_n(wpred)
    mu_pred = tf.math.divide(sum_pred,
                             tf.constant(depth**width, dtype=sum_pred.dtype))

    # TODO: still need loss
