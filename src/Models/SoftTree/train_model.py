"""
    Train the Model!

    Data - model compatability requirements:
    > argument order = x1, x2, ..., y, data_weight
    > > all xs (different inputs) will be packaged into a list

"""
import abc
import tensorflow as tf
import numpy as np
from typing import List
from dataclasses import dataclass
from Models.SoftTree.model_interfaces import AModel


class DataWeightGenerator(abc.ABC):
    # TODO: might need to make this more general later
    def gen_data_weights(self, absolute_idxs: tf.Tensor):
        """maps absolute indexes of a sampel to a data weight matrix

        Args:
            absolute_idxs (tf.Tensor): absolute indexes
                assumed to by len N tensor of integers

        Returns:
            tf.Tensor: data_weights
                N x num_model
        """
        pass


@dataclass
class DataPlan:
    """Index assignment
    = indices within tf.dataset"""
    x_inds: List[int]
    y_ind: int
    absolute_idx_ind: int


class TrainStep:
    """Build training graph
    Initialize with permanent objects/components
        == objects that are reused for all training steps
    Why package this into class?
        > allow for training of multiple model types
            in same session"""
    def __init__(self, model: AModel, optimizer: tf.keras.optimizers.Optimizer):
        self.model = model
        self.optimizer = optimizer

    @tf.function
    def train(self, x: List[tf.Tensor], y: tf.Tensor,
                    data_weights: tf.Tensor, epoch_temperature: tf.constant):
        with tf.GradientTape() as tape:
            loss = self.model.loss(x, y, data_weights, epoch_temperature)
        grads = tape.gradient(loss, self.model.get_trainable_weights())
        self.optimizer.apply_gradients(zip(grads, self.model.get_trainable_weights()))
        return loss

    @tf.function
    def loss(self, x: List[tf.Tensor], y: tf.Tensor,
                   data_weights: tf.Tensor, epoch_temperature: tf.constant):
        return self.model.loss(x, y, data_weights, epoch_temperature)

    @tf.function
    def loss_samples_noreg(self, x: List[tf.Tensor], y: tf.Tensor):
        return self.model.loss_samples_noreg(x, y)


def _gather_data(dat: tf.data.Dataset,
                 data_plan: DataPlan,
                 dw_gen: DataWeightGenerator):
    """gather data from dataset dat
    according to data plan

    Args:
        dat (tf.data.Dataset): dataset
        data_plan (DataPlan): data plan --> guides data extraction

    Returns:
        List[tf.constant]: x = input
        tf.constant: y = target
        tf.constant: data_weights
    """
    x = [dat[i] for i in data_plan.x_inds]
    y = dat[data_plan.y_ind]
    abs_idx = dat[data_plan.absolute_idx_ind]
    return x, y, dw_gen.gen_data_weights(abs_idx)


def train_epoch(train_dataset: tf.data.Dataset,
                data_plan: DataPlan,
                dw_gen: DataWeightGenerator,
                train_step: TrainStep,
                epoch_temperature: tf.constant):
    tr_losses = []
    for _step, dat in enumerate(train_dataset):
        x, y, data_weights = _gather_data(dat, data_plan, dw_gen)
        train_step.train(x, y, data_weights, epoch_temperature)
        tr_losses.append(train_step.loss(x, y, data_weights, epoch_temperature).numpy())
    return tr_losses


def eval_losses_noreg(train_dataset: tf.data.Dataset,
                      data_plan: DataPlan,
                      dw_gen: DataWeightGenerator,
                      train_step: TrainStep):
    """Get loss without regularization for all samples

    Args:
        train_dataset (tf.data.Dataset):
        data_plan (DataPlan):
        train_step (TrainStep):

    Returns:
        np.ndarray: losses
            number of batches x number of models
    """
    combo_losses = []
    for _step, dat in enumerate(train_dataset):
        x, y, data_weights = _gather_data(dat, data_plan, dw_gen)
        c_loss = train_step.loss_samples_noreg(x, y)
        # reshape data_weights:
        N = len(np.shape(c_loss)) - len(np.shape(data_weights))
        dw = np.reshape(data_weights, list(np.shape(data_weights)) + [1 for _ in range(N)])
        # save loss without reduction
        combo_losses.append(dw * c_loss.numpy())
    return np.vstack(combo_losses)


def train(train_dataset: tf.data.Dataset,
          data_plan: DataPlan,
          dw_gen: DataWeightGenerator,
          train_step: TrainStep,
          num_epoch: int = 3, epoch_temps: List[float] = [1., 1., 1.]):
    """Train for a number of epochs

    Args:
        train_dataset (tf.dataset): tensorflow training dataset (should be batched)
        train_step (TrainStep): model and optimizer packaged together
        num_epoch (int, optional): number of epochs. Defaults to 3.
        epoch_temps (List[float]): regularization scales for each epoch
            when = 1. --> no scaling of regularizers

    Returns:
        List[float]: train set loss after each training epoch
    """
    assert(len(epoch_temps) == num_epoch)
    epoch_tr_losses = []
    for epoch in range(num_epoch):
        _tr_losses = train_epoch(train_dataset, data_plan, dw_gen, train_step, tf.constant(epoch_temps[epoch]))
        # evaluate loss for frozen model
        tr_losses = eval_losses_noreg(train_dataset, data_plan, dw_gen, train_step)
        epoch_tr_losses.append(np.sum((tr_losses)))
    return epoch_tr_losses
