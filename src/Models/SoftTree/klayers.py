"""Custom Keras Layers"""
import keras
import tensorflow as tf
from typing import List


class MultiDense(keras.layers.Layer):
    # essentially dense layer except:
    # 1. there parallel dims that we won't reduce across
    # 2. we want to reduce across multiple dims (potentially)

    def __init__(self,
                 parallel_dims: List[int],
                 num_output: int):
        """

        Args:
            parallel_dims (List[int]): which dimensions
                of input are parallel
                NOT including batch size
                ... will reduce across all other dims
            num_output (int): number of outputs from
                the layer
        """
        super(MultiDense, self).__init__()
        self.parallel_dims = parallel_dims
        self.num_output = num_output

    def build(self, input_shape):
        # assumes 0th dim is batch dim
        self.reduce_dims = [z for z in range(1, len(input_shape))
                            if (z - 1) not in self.parallel_dims]
        self.w = self.add_weight(shape=input_shape[1:] + [self.num_output],
                                 initializer="random_normal",
                                 trainable=True)
        parallel_shape = [input_shape[ind + 1] for ind in self.parallel_dims]
        self.b = self.add_weight(shape=parallel_shape + [self.num_output],
                                 initializer="random_normal",
                                 trainable=True)

    def call(self, inputs):
        # linear inner produce ~ no activation unit
        # output shape = batch_size x parallel_dims x num_output
        return (tf.math.reduce_sum(tf.expand_dims(self.w, 0) *
                                   tf.expand_dims(inputs, -1),
                                   axis=self.reduce_dims)
               + tf.expand_dims(self.b, 0))
