"""Functional Tests"""
import tensorflow as tf
import numpy as np
from keras.backend import int_shape
from Models.SoftTree.kmodel import CustomModel
from Models.SoftTree.models import build_fweighted_linear_pred


def test_fweighted_linear_pred():
    # params:
    num_tree = 11
    depth = 2
    width = 3

    inp0 = tf.keras.Input(shape=(10, 3), dtype=tf.float32,
                          name="A")
    inp1 = tf.keras.Input(shape=(5), dtype=tf.float32,
                          name="B")
    # binary target
    target = tf.keras.Input(shape=(), dtype=tf.int32,
                            name="target")
    # mask = batch_size x num_tree
    # TODO: test different types?... float allows for graded masking
    sample_mask = tf.keras.Input(shape=(num_tree), dtype=tf.float32,
                                  name="sample_mask")
    
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

    # fake dataset:
    d = {"A": np.concatenate([np.ones((50, 10, 3)),
                              -1 * np.ones((50, 10, 3))], axis=0),
         "B": np.ones((100, 5)),
         "sample_mask": np.ones((100, num_tree), dtype=np.int32),
         "target": np.hstack([-1*np.ones((50,)), np.ones((50,))])}
    dset = tf.data.Dataset.from_tensor_slices(d).shuffle(100).batch(8)
    # compile the model
    model = CustomModel("loss", inputs=[inp0, inp1, target, sample_mask],
                            outputs=outputs)
    model.compile(optimizer='rmsprop')
    model.fit(dset, epochs=5)
    for v in dset:
        print(v["target"])
        print(model(v)["average_predictions"])
        break


if __name__ == "__main__":
    test_fweighted_linear_pred()