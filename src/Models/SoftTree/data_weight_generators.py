"""Data Weight Generators"""
import tensorflow as tf
from tensorflow.keras.layers import Hashing
from Models.SoftTree.train_model import DataWeightGenerator


class NullDWG(DataWeightGenerator):
    # return 1 for everything
    def __init__(self, num_model: int,
                    dtype = tf.float32):
        self.num_model = num_model
        self.base_tensor = tf.ones([1, num_model], dtype)

    def gen_data_weights(self, absolute_idxs: tf.Tensor):
        return tf.tile(self.base_tensor, [len(absolute_idxs), 1])


class StandardDWG(DataWeightGenerator):
    # uses a tensorflow hashing layer to convert
    # indices to one-hot weight matrices
    # salt serves as random seed
    # with salt --> uses Siphash64

    def __init__(self,
                 num_model: int,
                 salt: int,
                 dtype = tf.float32):
        # TODO: salt = seed, right?
        self.num_model = num_model
        self.salt = salt
        self.dtype = dtype  # TODO: casting will be really slow: no?
        # NOTE: newer versions of tensorflow allow
        # one-hot to be specified in Hashing
        self.hash_layer = Hashing(num_model, salt=salt)

    def gen_data_weights(self, absolute_idxs: tf.Tensor):
        return tf.cast(tf.one_hot(self.hash_layer(absolute_idxs), self.num_model), self.dtype)
