"""Testing forest components"""
from Models.SoftTree import layers
import tensorflow as tf
import numpy as np
import numpy.random as npr

def test_layers():
    rng = npr.default_rng(42)

    num_models = 12
    batch_size = 2
    xshape = [4, 5]
    width = 3
    x = tf.ones([batch_size] + xshape)
    LFB = layers.LayerFactoryBasic(num_models, xshape, width, rng)
    layer1 = LFB.build_layer()
    layer2 = LFB.build_layer()

    # ensure layers are not identical == factory (not clone)
    assert(np.sum(layer1.eval(x).numpy() != layer2.eval(x).numpy()) >= 1)
    assert(np.shape(layer1.eval(x).numpy()) == (2, 12, 3))
    assert(np.shape(layer1.w.numpy()) == tuple([num_models, width] + xshape))
    assert(np.shape(layer1.wop.numpy()) == tuple([1, num_models, width] + xshape))
    assert(np.shape(layer1.offset.numpy()) == (12, 3))
    # first 3 = batch_size, num_model, width (reduce across xshape)
    assert(layer1.reduce_dims == [3, 4])
    

    # low rank layers
    low_dim = 2
    LFLR = layers.LayerFactoryLowRankFB(num_models, xshape, width, low_dim, rng)
    layer3 = LFLR.build_layer()
    layer4 = LFLR.build_layer()
    assert(np.sum(layer3.eval(x).numpy() != layer4.eval(x).numpy()) >= 1)
    assert(np.shape(layer3.eval(x).numpy()) == (2, 12, 3))
    assert(np.shape(layer3.w.numpy()) == tuple([num_models, width] + xshape))
    assert(np.shape(layer3.fb.numpy()) == tuple([low_dim] + xshape))
    assert(np.shape(layer3.wb.numpy()) == tuple([num_models, width, low_dim] + xshape))



if __name__ == '__main__':
    test_layers()