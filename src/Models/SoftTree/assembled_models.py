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
        return gates * parallel_loss
        
    def loss(self, x: List[tf.Tensor], y: tf.Tensor, data_weights: tf.Tensor,
                reg_epoch_scale: float = 1.):
        """Loss from loss_samples_noreg + regularization losses

        Args:
            x (List[tf.Tensor]): input
            y (tf.Tensor): truth
            data_weights (tf.Tensor): weights on each datapoint
                in the sample
            reg_epoch_scale (float): how much to scale regularization by dynamically
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
                reg_epoch_scale: float = 1.):
        """Loss from loss_samples_noreg + regularization losses

        Args:
            x (List[tf.Tensor]): input
            y (tf.Tensor): truth
            data_weights (tf.Tensor): weights on each datapoint
                in the sample
                batch_size x num_model
            reg_epoch_scale (float): how much to scale regularization by dynamically
                = a function of temperature
        """
        core_loss = tf.reduce_sum(self.loss_samples_noreg(x, y) * data_weights)
        return (core_loss +
                self.pred_model.regularization_loss(x, data_weights, reg_epoch_scale) +
                self.gating_model.regularization_loss(x, data_weights, reg_epoch_scale))


if __name__ == "__main__":
    # TODO: move this testing code to a different file

    from Models.SoftTree.forest import build_forest
    from Models.SoftTree.simple_predictors import LinearPred
    from Models.SoftTree.layers import LayerFactoryBasic
    from Models.SoftTree.objective_funcs import MultinomialLoss
    from numpy.random import default_rng
    import numpy.random as npr

    prediction_dim = 6
    layer_width = 2
    base_models = 2
    models_per_base = 4
    xshape = [5, 5]  # input from 5x5 matrix space
    rng = default_rng(42)

    layer_factory = LayerFactoryBasic(base_models, models_per_base, xshape, layer_width, rng)

    # forest = gating model
    depth = 2
    forest, forest_width, _ = build_forest(depth, layer_factory, 0., 0.)

    # simple prediction model
    base_models_pred = forest.get_num_state() * (base_models * models_per_base)
    layer_factory_pred = LayerFactoryBasic(base_models_pred, 1, xshape, prediction_dim, rng)
    pred_model = LinearPred(layer_factory_pred.build_layer(), base_models*models_per_base, forest.get_num_state())

    # objective function:
    norm_dim = 3  # which dim of tensor contains the logits
    obj_func = MultinomialLoss(norm_dim)

    # whole model
    GLM = GatedLossModel(forest, pred_model, obj_func)
    print(len(GLM.get_trainable_weights()))
    # run stuff thru model
    batch_size = 64
    xshape_dat = [batch_size] + xshape
    x = tf.constant(npr.rand(*xshape_dat), dtype=tf.float32)
    model_predz = GLM.get_preds([x])
    print(len(model_predz))
    print(model_predz[0].shape)
    print(model_predz[1].shape)

    # y dictated by objective function
    y = npr.randint(0, 6, (batch_size,))
    y = tf.constant(y, dtype=tf.int32)
    loss_sample = GLM.loss_samples_noreg([x], y)
    print(loss_sample.shape)
