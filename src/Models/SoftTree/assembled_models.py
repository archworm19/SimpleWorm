"""Model Builder functions

    = common model configurations
"""
import abc
import tensorflow as tf
import numpy.random as npr
from Models.SoftTree.layers import LayerFactoryIface, var_construct
from Models.SoftTree.forest import build_forest
from Models.SoftTree.decoders import GaussFull


class AModel(abc.ABC):
    # Assembeled Model interface
    def loss(self, x, y, data_weights):
        """The loss function

        Args:
            x (tf.tensor or List[tf.tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.tensor): target/truth ~ batch_size x 
            data_weights (tf.tensor): weights on the data points
                batch_size x num_model
        
        Returns:
            tf.tensor: combined loss; scalar
        """
        pass

    def get_trainable_weights(self):
        """Get the trainable weights

        Returns:
            List[tf.tensor]
        """
        pass

    def loss_samples(self, x, y, data_weights):
        """Loss for each sample in batch

        Args:
            x (tf.tensor or List[tf.tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.tensor): target/truth ~ batch_size x 
            data_weights (tf.tensor): weights on the data points
                batch_size x num_model

        Returns:
            tf.tensor: combined loss
                batch_size x num_model
        """
        pass

    # TODO: probably need some prediction methods
    # > predict state? > predict average rep???
