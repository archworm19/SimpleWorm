"""Testing forest components"""
from Models.SoftTree import layers
import tensorflow as tf

def test_layers():
    num_models = 12
    batch_size = 2
    xshape = [4, 5]
    width = 3
    x = tf.ones([batch_size] + xshape)
    LFB = layers.LayerFactoryBasic(num_models, xshape, width)
    layer1 = LFB.build_layer()
    layer2 = LFB.build_layer()
    print(layer1.eval(x))
    print(layer2.eval(x))
    print(layer1.l2())

    LFLR = layers.LayerFactoryLowRankTseries(num_models,
                                             xshape[0],
                                             xshape[1],
                                             2,
                                             width)
    layer3 = LFLR.build_layer()
    print(layer3.eval(x))
    print(layer3.l2())

    # low-rank layers

if __name__ == '__main__':
    test_layers()