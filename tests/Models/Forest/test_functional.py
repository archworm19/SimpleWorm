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
    y_pred, y_pred_state_weights, y_pred_ave = LSF([inp0, inp1])

    # binary loss
    pred_mask = y_pred_state_weights * tf.expand_dims(sample_mask, 2)
    loss = binary_loss(y_pred, target, pred_mask)

    # fake dataset:
    # deterministic from A
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
                        outputs={"loss": loss,
                                 "average_predictions": y_pred_ave})
    model.compile(optimizer='rmsprop')
    model.fit(dset, epochs=10)
    for v in dset:
        ypred = model(v)["average_predictions"].numpy()
        targz = v["target"].numpy()
        yp1 = ypred[targz > 0.5]
        yp0 = ypred[targz <= 0.5]
        if len(yp1) > 0:
            assert np.amin(yp1) > 0.5
        if len(yp0) > 0:
            assert np.amax(yp0) < 0.5


def test_linear_binary_forest_XOR():
    # XOR testing = tests whether we can learn nonlinear functions
    # A xor B --> target

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
    y_pred, y_pred_state_weights, y_pred_ave = LSF([inp0, inp1])

    # binary loss
    pred_mask = y_pred_state_weights * tf.expand_dims(sample_mask, 2)
    loss = binary_loss(y_pred, target, pred_mask)

    # fake dataset:
    # deterministic from A
    d = {"A": np.concatenate([np.ones((50, 10, 3)),
                              -1 * np.ones((50, 10, 3))], axis=0),
         "B": np.concatenate([np.zeros((25, 5)),
                              np.ones((50, 5)),
                              np.zeros((25, 5))], axis=0),
         "sample_mask": np.ones((100, num_tree), dtype=np.int32),
         "target": np.concatenate([np.ones((25,)),
                                   np.zeros((25,)),
                                   np.ones((25,)),
                                   np.zeros((25,))]),
         "forest_pen": np.zeros((100,))}
    dset = tf.data.Dataset.from_tensor_slices(d).shuffle(100).batch(8)
    # compile the model
    model = CustomModel("loss", inputs=[inp0, inp1, target, sample_mask,
                                        fpen],
                        outputs={"loss": loss,
                                 "average_predictions": y_pred_ave})
    model.compile(optimizer=tf.keras.optimizers.Adam(.01))
    model.fit(dset, epochs=10)
    for v in dset:
        ypred = model(v)["average_predictions"].numpy()
        targz = v["target"].numpy()
        yp1 = ypred[targz > 0.5]
        yp0 = ypred[targz <= 0.5]
        if len(yp1) > 0:
            assert np.amin(yp1) > 0.5
        if len(yp0) > 0:
            assert np.amax(yp0) < 0.5


if __name__ == "__main__":
    test_linear_binary_forest()
    test_linear_binary_forest_XOR()
