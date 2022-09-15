"""Forest Builder"""
import tensorflow as tf
from keras.backend import int_shape
from Models.SoftTree.klayers import MultiDense
from Models.SoftTree.tree import build_tree


def test_multi_dense():
    # test with 1st dim parallel
    inp1 = tf.keras.Input(shape=(4, 10),
                          dtype=tf.float32)
    v = MultiDense([1], 3)(inp1)
    assert(int_shape(v) == (None, 10, 3))


def test_build():
    for depth, width in zip([2, 3], [2, 3, 4]):
        inp1 = tf.keras.Input(shape=(4, 10),
                            dtype=tf.float32)
        inp2 = tf.keras.Input(shape=(5),
                            dtype=tf.float32)
        x = build_tree(depth, width, [inp1, inp2])
        assert(int_shape(x) == (None, width**depth))


def test_fnorm():
    x = 10. * tf.ones([5, 3, 4])
    depth = 2
    width = 3
    x = build_tree(depth, width, [x])
    assert(tf.math.reduce_all(tf.math.abs(tf.math.reduce_sum(x, axis=1) - 1.0) < .001))


if __name__ == "__main__":
    test_multi_dense()
    test_build()
    test_fnorm()
