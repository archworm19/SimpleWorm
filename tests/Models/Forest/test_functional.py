"""Functional Tests"""
import tensorflow as tf
from keras.backend import int_shape
from Models.SoftTree.kmodel import CustomModel
from Models.SoftTree.models import build_fweighted_linear_pred


def test_fweighted_linear_pred():
    # params:
    num_tree = 11
    depth = 2
    width = 3

    inp0 = tf.keras.Input(shape=(10, 3), dtype=tf.float32)
    inp1 = tf.keras.Input(shape=(5), dtype=tf.float32)
    # binary target
    target = tf.keras.Input(shape=(), dtype=tf.int32)
    # mask = batch_size x num_tree
    # TODO: test different types?... float allows for graded masking
    sample_mask = tf.keras.Input(shape=(num_tree), dtype=tf.float32)
    
    outputs = build_fweighted_linear_pred([inp0, inp1],
                                          target,
                                          sample_mask,
                                          num_tree,
                                          depth,
                                          width,
                                          x2=None)
    assert int_shape(outputs["loss"]) == ()
    assert int_shape(outputs["logit_predictions"]) == (None, 11, 9)
    assert int_shape(outputs["average_predictions"]) == (None, )


if __name__ == "__main__":
    test_fweighted_linear_pred()