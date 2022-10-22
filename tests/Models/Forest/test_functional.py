"""Functional Tests"""
import tensorflow as tf
import numpy as np
from keras.backend import int_shape
from Models.SoftTree.kmodel import CustomModel
from Models.SoftTree.hybrid_models import LinearScalarForest
from Models.SoftTree.tree import forest_reg_loss
from Models.SoftTree.loss_funcs import binary_loss


def test_linear_binary_forest():
    # single forest ~ no boosting
    num_tree = 11
    depth = 1
    width = 2
    LSF = LinearScalarForest(num_tree, depth, width,
                             transform_func=tf.math.sigmoid)

    inp0 = tf.keras.Input(shape=(10, 3), dtype=tf.float32,
                          name="A")
    inp1 = tf.keras.Input(shape=(5), dtype=tf.float32,
                          name="B")
    # binary target
    target = tf.keras.Input(shape=(), dtype=tf.int32,
                            name="target")
    # forest regularization penalty
    fpen = tf.keras.Input(shape=(), dtype=tf.float32,
                          name="forest_pen")
    # mask = batch_size x num_tree
    # TODO: test different types?... float allows for graded masking
    sample_mask = tf.keras.Input(shape=(num_tree,), dtype=tf.float32,
                                  name="sample_mask")

    # LSF eval:
    y_pred, y_pred_ave = LSF([inp0, inp1])

    # binary loss
    loss = binary_loss(y_pred, target, sample_mask)

    # TODO: left off here

    # fake dataset:
    d = {"A": np.concatenate([np.ones((50, 10, 3)),
                              -1 * np.ones((50, 10, 3))], axis=0),
         "B": np.ones((100, 5)),
         "sample_mask": np.ones((100, num_tree), dtype=np.int32),
         "target": np.hstack([np.zeros((50,)), np.ones((50,))]),
         "forest_pen": np.zeros((100,))}
    dset = tf.data.Dataset.from_tensor_slices(d).shuffle(100).batch(8)
    # compile the model
    model = CustomModel("loss", inputs=[inp0, inp1, target, sample_mask,
                                        fpen],
                            outputs=outputs)
    model.compile(optimizer='rmsprop')
    model.fit(dset, epochs=5)
    for v in dset:
        ypred = model(v)["average_predictions"]
        primary = tf.math.reduce_sum(ypred * tf.cast(v["target"], ypred.dtype))
        off_primary = tf.math.reduce_sum(ypred * tf.cast(1.0 - v["target"], ypred.dtype))
        assert (primary - off_primary).numpy() > 0


"""
def test_fweighted_linear_pred_XOR():
    # can it learn XOR pattern? yep
    # params:
    num_tree = 11
    depth = 1  # should be good enough
    width = 2

    inp0 = tf.keras.Input(shape=(2,), dtype=tf.float32,
                          name="A")
    # binary target
    target = tf.keras.Input(shape=(), dtype=tf.int32,
                            name="target")
    # mask = batch_size x num_tree
    sample_mask = tf.keras.Input(shape=(num_tree,), dtype=tf.float32,
                                  name="sample_mask")
    # forest regularization penalty
    fpen = tf.keras.Input(shape=(), dtype=tf.float32,
                          name="forest_pen")
    
    outputs = build_fweighted_linear_pred([inp0],
                                          target,
                                          sample_mask,
                                          fpen,
                                          num_tree,
                                          depth,
                                          width,
                                          x2=None)

    # fake XOR dataset:
    d = {"A": np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]] * 500),
         "target": np.array([0, 1, 1, 0] * 500),
         "sample_mask": np.ones((4 * 500, num_tree)),
         "forest_pen": np.zeros((4 * 500,))}

    dset = tf.data.Dataset.from_tensor_slices(d).shuffle(100).batch(8)
    # compile the model
    model = CustomModel("loss", inputs=[inp0, target, sample_mask,
                                        fpen],
                            outputs=outputs)
    model.compile(optimizer='rmsprop')
    model.fit(dset, epochs=20)
    one_vals, non_one_vals = [], []
    for v in dset:
        ypred = model(v)["average_predictions"]
        one_inds = tf.math.reduce_sum(v["A"], axis=1).numpy() == 1
        one_vals.append(ypred.numpy()[one_inds])
        non_one_vals.append(ypred.numpy()[np.logical_not(one_inds)])
    one_vals = np.concatenate(one_vals, axis=-1)
    non_one_vals = np.concatenate(non_one_vals, axis=-1)
    assert len(one_vals) == len(non_one_vals)
    assert np.all(one_vals > 0.5)
    assert np.all(non_one_vals < 0.5)
"""


if __name__ == "__main__":
    test_linear_binary_forest()
