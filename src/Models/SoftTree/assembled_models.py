"""Model Builder functions

    = common model configurations
"""
import tensorflow as tf
from typing import List, Union
from Models.SoftTree.model_interfaces import GateSubModel, PredSubModel, AModel


class GatedModel(AModel):
    def __init__(self, gating_model: GateSubModel, pred_model: PredSubModel):
        # TODO: need an objective function / obj func interface!
        self.gating_model = gating_model
        self.pred_model = pred_model
        assert(gating_model.get_num_state() == pred_model.get_num_state()), "state mismatch"

    def get_preds(self, x: Union[tf.Tensor, List[tf.Tensor]]):
        """Get predictions

        Args:
            x (Union[tf.Tensor, List[tf.Tensor]]): input

        Returns:
            tf.Tensor: gated predictions
                batch_size x num_model x num_state x num_pred
        """
        # gating probs = batch_size x num_model x num_state
        gates = self.gating_model.get_state_probs(x)
        # preds = batch_size x num_model x num_state x num_pred
        preds = self.pred_model.get_preds(x)
        return gates * tf.expand_dims(preds, 3)

    def loss_samples_noreg(self, x, y):
        """Loss for each sample in batch

        Args:
            x (tf.tensor or List[tf.tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.tensor): target/truth ~ batch_size x 

        Returns:
            tf.tensor: combined loss
                batch_size x num_model
        """
        # --> batch_size x num_model x num_state x num_pred
        preds = self.get_preds(x)
        # TODO: design objective function interface
        