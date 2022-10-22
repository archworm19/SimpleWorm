"""Test Hybrid Models"""
import tensorflow as tf
from Models.SoftTree.hybrid_models import LinearScalarForest


def test_linear_scalar():
    batch_size = 16
    depth = 1
    width = 2
    num_tree = 11
    num_state = int(width**(depth+1))
    LSF = LinearScalarForest(num_tree, depth, width)
    inps = [tf.ones([batch_size, 5, 3])]
    y_pred, y_pred_ave = LSF(inps)
    assert tf.math.reduce_all(tf.shape(y_pred) == tf.constant([batch_size, num_tree, num_state]))
    assert tf.shape(y_pred_ave).numpy() == (batch_size,)
    # trees are different?
    tree_ave = tf.math.reduce_mean(y_pred, axis=[0, 2])
    assert tf.math.reduce_all(tf.math.abs(tree_ave[1:] - tree_ave[:-1]) > 1e-6)

    # test boosting:
    LSF2 = LinearScalarForest(2, 0, 3)
    y_pred2, y_pred_ave2 = LSF2(inps * 1000, x2=y_pred_ave)
    assert tf.math.reduce_all(tf.shape(y_pred2) == tf.constant([batch_size, 2, 3]))
    assert tf.shape(y_pred_ave2).numpy() == (batch_size,)
    # testing that average prediction is range restricted
    assert tf.math.reduce_all(tf.math.logical_and(y_pred_ave2 <= 1., y_pred_ave2 >= 0.))


if __name__ == "__main__":
    test_linear_scalar()
