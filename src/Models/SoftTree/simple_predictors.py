"""Simple Predictors implementing the PredSubModel interface"""
import tensorflow as tf
from typing import List
from Models.SoftTree.model_interfaces import PredSubModel
from Models.SoftTree.layers import LayerIface

# TODO: might be easier just to have a separate class to guarantee offset!
class LinearPred(PredSubModel):
    """Linear or offset-only predictions
    linear vs. offset determined by layer passed in
    Make predictions in each sub-state
    NOTE: wrapping layers guarantees states
    are completely parallel"""

    def __init__(self, layer: LayerIface,
                 num_model: int,
                 num_state: int):
        assert(layer.get_num_parallel_models() == num_model * num_state)
        self.layer = layer
        self.num_model = num_model
        self.num_state = num_state
        self.prediction_dim = layer.get_width()
        self.new_shape = [-1, num_model, num_state, self.prediction_dim]

    def get_trainable_weights(self):
        return self.layer.get_trainable_weights()

    def get_num_state(self):
        return self.num_state

    def get_preds(self, x: List[tf.Tensor]):
        """get the predictions

        Args:
            x (List[tf.Tensor]): input

        Returns:
            tf.tensor: predictions
                batch_size x num_model x num_state x prediction dim
        """
        # --> batch_size x parallel_models x width
        # = batch_size x (num_model * num_state) x prediction dim
        raw_eval = self.layer.eval(x)
        # reshape to separate parallel models and states:
        return tf.reshape(raw_eval, self.new_shape)

    def regularization_loss(self, x: List[tf.Tensor], data_weights: tf.Tensor, reg_epoch_scale: tf.constant):
        # TODO: should probably change this!
        return tf.constant(0., dtype=tf.float32)
