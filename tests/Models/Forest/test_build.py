"""Forest Builder"""
import tensorflow as tf
from keras.backend import int_shape
from Models.SoftTree.klayers import MultiDense
from Models.SoftTree.tree import Forest, ForestLinear


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


def test_forest():
    # test out layer
    # depth = 1, width = 2
    batch_size = 8
    d1 = 5
    depth = 1  # implied
    width = 2
    total_nodes = 3
    out_state = 4
    v = tf.ones([batch_size, d1, total_nodes, width])
    vout = Forest()(v)
    assert tf.math.reduce_all(tf.shape(vout) == [batch_size, d1, out_state])
    assert tf.math.reduce_all((tf.math.reduce_sum(vout, axis=-1) - 1) < 1e-3)

    # test with more complex size:
    # depth = 2; width = 3
    width = 3
    total_nodes = 1 + 3 + 9
    out_state = 27
    v = tf.ones([batch_size, d1, total_nodes, width])
    F = Forest()
    vout = F(v)
    assert tf.math.reduce_all(tf.shape(vout) == [batch_size, d1, out_state])
    # super important test ~ tests that forest evaluates all inputs
    _, next_ind = F._eval_forest(v, tf.constant(1.), 3, 3, 0)
    assert next_ind == tf.shape(v)[2]

    # what if I use a keras tensor:
    v2 = tf.keras.Input((5, 1 + 3 + 9, 3))
    vout = Forest()(v2)
    model = tf.keras.Model(inputs=v2, outputs=vout)
    model(v)

def test_linforest():
    # testing forest linear
    batch_size = 8
    width = 3
    depth = 2
    num_tree = 11
    v1 = tf.ones([batch_size, 3])
    v2 = tf.ones([batch_size, 6, 2])
    Flin = ForestLinear(width, depth, num_tree)
    vout = Flin([v1, v2])
    assert tf.math.reduce_all(tf.shape(vout) == [batch_size, num_tree, width**(depth+1)])

def test_forest_grad():
    # let's look at forest gradients to make sure stuff looks ok
    # TODO: think about how to do this test properly
    batch_size = 8
    width = 2
    depth = 1
    num_tree = 3
    v1 = tf.ones([batch_size, 3])
    v2 = tf.ones([batch_size, 5, 3])
    Flin = ForestLinear(width, depth, num_tree)
    with tf.GradientTape(persistent=True) as g:
        y = Flin([v1, v2])
    grad_eval = g.gradient(y, Flin.lin_layers[0].w)
    print(grad_eval[0])
    assert tf.math.reduce_min(tf.math.abs(grad_eval)).numpy() != 0
    grad_eval = g.gradient(y, Flin.lin_layers[1].w)
    assert tf.math.reduce_min(tf.math.abs(grad_eval)).numpy() != 0


if __name__ == "__main__":
    test_multi_dense()
    test_multi_dense_parallel()
    test_forest()
    test_linforest()
    test_forest_grad()
