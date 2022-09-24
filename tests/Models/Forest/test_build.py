"""Forest Builder"""
import tensorflow as tf
from keras.backend import int_shape
from Models.SoftTree.klayers import MultiDense
from Models.SoftTree.tree import build_forest, StandardTreeLayerFactory


def test_multi_dense():
    # test with 1st dim parallel
    inp1 = tf.keras.Input(shape=(4, 10),
                          dtype=tf.float32)
    v = MultiDense([1], 3)(inp1)
    assert(int_shape(v) == (None, 10, 3))


def test_multi_dense_parallel():
    # ensure that parallel dims in multi dense 
    # are actually parallel

    # 1st dim is parallel
    v = tf.ones([100, 10, 4])
    M = MultiDense([0], 2)
    vout = M(v)
    assert(all(tf.shape(vout).numpy() == (100, 10, 2)))
    # vary even dims --> ensure odd dims are unchanged:
    vmask = tf.reshape(tf.constant([0, 1] * 5), [1, -1, 1])
    vmask = tf.tile(vmask, [100, 1, 4])
    v_inv_mask = 1 - vmask[:, :, :2]
    vout_const_mask = vout * tf.cast(v_inv_mask, vout.dtype)
    vout_dynamic_mask = vout * tf.cast(vmask[:, :, :2], vout.dtype)
    for z in range(0, 10):
        v2 = v + tf.cast(vmask * z, v.dtype)
        vout2 = M(v2)
        vout2_const_mask = vout2 * tf.cast(v_inv_mask, vout2.dtype)
        vout2_dynamic_mask = vout2 * tf.cast(vmask[:, :, :2], vout2.dtype)
        assert(tf.math.reduce_all(vout2_const_mask == vout_const_mask).numpy())
        if z == 0:
            assert(tf.math.reduce_all(vout2_dynamic_mask == vout_dynamic_mask).numpy())
        else:
            assert(not tf.math.reduce_all(vout2_dynamic_mask == vout_dynamic_mask).numpy())


def test_build():
    for depth, width in zip([2, 3], [2, 3, 4]):
        tfacts = [StandardTreeLayerFactory(width, 11),
                  StandardTreeLayerFactory(width, 11)]
        inp1 = tf.keras.Input(shape=(4, 10),
                            dtype=tf.float32)
        inp2 = tf.keras.Input(shape=(5),
                            dtype=tf.float32)
        x = build_forest(width, depth, [inp1, inp2],
                         tfacts)
        assert(int_shape(x) == (None, tfacts[0].get_num_trees(), width**depth))


def test_fnorm():
    x = 10. * tf.ones([5, 3, 4])
    depth = 2
    width = 3
    x = build_forest(width, depth, [x],
                     [StandardTreeLayerFactory(width, 1)])
    assert(tf.math.reduce_all(tf.math.abs(tf.math.reduce_sum(x, axis=-1) - 1.0) < .001))


if __name__ == "__main__":
    test_multi_dense()
    test_multi_dense_parallel()
    test_build()
    test_fnorm()
