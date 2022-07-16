"""Model Builder functions

    = common model configurations
"""
import tensorflow as tf
from typing import List, Union
from Models.SoftTree.model_interfaces import GateSubModel, PredSubModel, AModel
from Models.SoftTree.objective_funcs import ObjFunc

# TODO: should we test this against GatedAveraging model?
# ... average predictions instead of losses... yes.. do both

class GatedLossModel(AModel):
    """Gating model --> probabilities in N sub-states
    Prediction model loss generated for each sub-state
    Gating model probabilities scale prediction model loss"""
    def __init__(self,
                 gating_model: GateSubModel,
                 pred_model: PredSubModel,
                 obj_func: ObjFunc):
        self.gating_model = gating_model
        self.pred_model = pred_model
        self.obj_func = obj_func
        assert(gating_model.get_num_state() == pred_model.get_num_state()), "state mismatch"

    def get_trainable_weights(self):
        return self.gating_model.get_trainable_weights() + self.pred_model.get_trainable_weights()

    def get_preds(self, x: List[tf.Tensor]):
        """Get predictions

        Args:
            x (List[tf.Tensor]): input

        Returns:
            List[tf.Tensor]: gated predictions
                1. gating model state probabiltieis
                    batch_size x num_model x num_state
                2. prediction model logits
                    batch_size x num_model x num_state x num_pred
        """
        # gating probs = batch_size x num_model x num_state
        gates = self.gating_model.get_state_probs(x)
        # preds = batch_size x num_model x num_state x num_pred
        preds = self.pred_model.get_preds(x)
        return [gates, preds]

    def loss_samples_noreg(self, x: List[tf.Tensor], y: tf.Tensor):
        """Loss for each sample in batch

        Args:
            x (List[tf.Tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.Tensor): target/truth ~ batch_size x 

        Returns:
            tf.tensor: combined loss
                batch_size x num_model
        """
        # gates: batch_size x num_model x num_state
        # preds: batch_size x num_model x num_state x num_pred
        [gates, preds] = self.get_preds(x)
        # loss for prediction model on each sub-state
        # --> batch_size x parallel_dims = batch_size x num_model x num_state
        parallel_loss = self.obj_func.loss_sample(preds, y)
        # scale losses by gating probabilities
        # and reduce across states
        return tf.reduce_sum(gates * parallel_loss, axis=2)
        
    def loss(self, x: List[tf.Tensor], y: tf.Tensor, data_weights: tf.Tensor,
                reg_epoch_scale: tf.constant = tf.constant(1.)):
        """Loss from loss_samples_noreg + regularization losses

        Args:
            x (List[tf.Tensor]): input
            y (tf.Tensor): truth
            data_weights (tf.Tensor): weights on each datapoint
                in the sample
            reg_epoch_scale (tf.constant): how much to scale regularization by dynamically
                = a function of temperature
        """
        core_loss = tf.reduce_sum(self.loss_samples_noreg(x, y) * data_weights)
        return (core_loss +
                self.pred_model.regularization_loss(x, data_weights, reg_epoch_scale) +
                self.gating_model.regularization_loss(x, data_weights, reg_epoch_scale))


class GatedPredModel(AModel):
    """Gating model --> probabilities in N sub-states
    Prediction model --> predictions for each of the subtates
    Weighted average of predictions where weights = Gating model probs
    --> loss at end"""
    def __init__(self,
                 gating_model: GateSubModel,
                 pred_model: PredSubModel,
                 obj_func: ObjFunc):
        self.gating_model = gating_model
        self.pred_model = pred_model
        self.obj_func = obj_func
        assert(gating_model.get_num_state() == pred_model.get_num_state()), "state mismatch"

    def get_trainable_weights(self):
        return self.gating_model.get_trainable_weights() + self.pred_model.get_trainable_weights()

    def get_preds(self, x: List[tf.Tensor]):
        """Get predictions

        Args:
            x (List[tf.Tensor]): input

        Returns:
            List[tf.Tensor]: gate model averaged predictions
                = [preds]
                preds = batch_size x num_model
        """
        # gating probs = batch_size x num_model x num_state
        gates = self.gating_model.get_state_probs(x)
        # preds = batch_size x num_model x num_state x num_pred
        preds = self.pred_model.get_preds(x)
        # weighted average
        w_ave = tf.reduce_sum(tf.expand_dims(gates, 3) * preds,
                              axis=[2,3])
        return [w_ave]

    def loss_samples_noreg(self, x: List[tf.Tensor], y: tf.Tensor):
        """Loss for each sample in batch

        Args:
            x (List[tf.Tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.Tensor): target/truth ~ batch_size x 

        Returns:
            tf.tensor: combined loss
                batch_size x num_model
        """
        # preds: batch_size x num_model
        preds = self.get_preds(x)[0]
        # loss for prediction model on each sub-state
        # --> batch_size x parallel_dims = batch_size x num_model x num_state
        parallel_loss = self.obj_func.loss_sample(preds, y)
        # scale losses by gating probabilities
        return parallel_loss
        
    def loss(self, x: List[tf.Tensor], y: tf.Tensor, data_weights: tf.Tensor,
                reg_epoch_scale: tf.constant = tf.constant(1.)):
        """Loss from loss_samples_noreg + regularization losses

        Args:
            x (List[tf.Tensor]): input
            y (tf.Tensor): truth
            data_weights (tf.Tensor): weights on each datapoint
                in the sample
                batch_size x num_model
            reg_epoch_scale (tf.constant): how much to scale regularization by dynamically
                = a function of temperature
        """
        core_loss = tf.reduce_sum(self.loss_samples_noreg(x, y) * data_weights)
        return (core_loss +
                self.pred_model.regularization_loss(x, data_weights, reg_epoch_scale) +
                self.gating_model.regularization_loss(x, data_weights, reg_epoch_scale))
