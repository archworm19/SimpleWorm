"""Test the loss functions"""
import numpy as np
import tensorflow as tf
from Models.SoftTree.loss_funcs import binary_loss, quantile_loss


def test_binaryloss():
    tol = 1e-4
    # binary loss
    ar_np = np.ones((4, 2)) * -5.
    # class predictions = 0, 0, 1, 1 for both models
    ar_np[2:,:] = 5.
    # truths = 0, 1, 0, 1 (ends correct --> low error end, high error mids)
    truths_np = np.array([0, 1, 0, 1])
    preds = tf.constant(ar_np, tf.float32)
    truths = tf.constant(truths_np, tf.float32)
    bl_loss = binary_loss(preds, truths)
    assert(np.shape(bl_loss.numpy()) == (4, 2))
    targ = np.array([[.0067, .0067], [5.0067, 5.0067], [5.0067, 5.0067], [.0067, .0067]])
    assert(np.all((bl_loss.numpy() - targ) < tol))

    # compare to non-expanded formulation
    pred_prob = tf.nn.sigmoid(preds)
    truths_re = tf.reshape(truths, [-1,1])
    loss_ref = -1 * (truths_re * tf.math.log(pred_prob) + (1 - truths_re) * tf.math.log(1 - pred_prob))
    assert(np.all((bl_loss - loss_ref).numpy() < tol))


def test_quantileloss():
    # quantile loss testing
    preds_np = np.ones((5, 2, 3))
    preds = tf.constant(preds_np, tf.float32)
    truths = tf.constant(np.linspace(0.0, 2.0, 5), tf.float32)
    taus = tf.constant([.1, .5, .9])
    ql_loss = quantile_loss(preds, truths, taus)
    assert(tf.math.reduce_all(tf.shape(ql_loss) == tf.shape(preds)))

    # more easily interpretable test?
    # ensure loss is minimized for correct quantile!
    truths = np.arange(0, 100)  # quantile numbs = 10, 50, 90
    best_preds = np.array([[[10, 50, 90]]])
    best_preds = np.tile(best_preds, (100, 1, 1))
    for scale_direction in [2, -2]:
        lozz = []
        for i in range(5):
            mut_preds = best_preds + scale_direction * i * np.ones((100, 1, 1))  # shifting preds test
            mut_loss = tf.reduce_sum(quantile_loss(tf.constant(mut_preds, tf.float32),
                                                   tf.constant(truths, tf.float32),
                                                   taus))
            lozz.append(mut_loss.numpy())
        assert(np.all(np.diff(np.array(lozz)) >= 0))


if __name__ == "__main__":
    test_binaryloss()
    test_quantileloss()
