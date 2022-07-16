"""
    Train the Model!

    Data - model compatability requirements:
    > argument order = x1, x2, ..., y, data_weight
    > > all xs (different inputs) will be packaged into a list

"""
import tensorflow as tf
import numpy as np
from typing import List
from dataclasses import dataclass
from Models.SoftTree.model_interfaces import AModel


# TODO: checking mechanisms to ensure shape matches

@dataclass
class DataPlan:
    """Index assignment"""
    x_inds: List[int]
    y_ind: int
    data_weight_inds: int


class TrainStep:
    """Build training graph
    Initialize with permanent objects/components
        == objects that are reused for all training steps
    Why package this into class?
        > allow for training of multiple model types
            in same session"""
    # TODO: optimizer type?
    def __init__(self, model: AModel, optimizer):
        self.model = model
        self.optimizer = optimizer

    @tf.function
    def train(self, x, y, data_weights,
                    epoch_temperature: tf.constant):
        with tf.GradientTape() as tape:
            loss = self.model.loss(x, y, data_weights, epoch_temperature)
        grads = tape.gradient(loss, self.model.get_trainable_weights())
        self.optimizer.apply_gradients(zip(grads, self.model.get_trainable_weights()))
        return loss

    @tf.function
    def loss(self, x, y, data_weights, epoch_temperature):
        return self.model.loss(x, y, data_weights, epoch_temperature)


def _gather_data(dat: tf.data.Dataset,
                 data_plan: DataPlan):
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
    data_weights = dat[data_plan.data_weight_inds]
    return x, y, data_weights


# TODO: optimizer type?
def train_epoch(train_dataset: tf.data.Dataset, data_plan: DataPlan,
                train_step: TrainStep,
                epoch_temperature: tf.constant):
    tr_losses = []
    for _step, dat in enumerate(train_dataset):
        x, y, data_weights = _gather_data(dat, data_plan)
        train_step.train(x, y, data_weights, epoch_temperature)
        tr_losses.append(train_step.loss(x, y, data_weights, epoch_temperature).numpy())
    return tr_losses


# TODO: finish this
# TODO: needs support for temperatures
def eval_losses(train_dataset, model: AModel):
    """Get prediction loss for each batch, model combination

    Args:
        train_dataset (tf.dataset): training dataset
        model: the model

    Returns:
        np.ndarray: losses
            number of batches x number of models
    """
    combo_losses = []
    for _step, dat in enumerate(train_dataset):
        x, y, data_weights = _gather_data(dat)
        c_loss = model.loss_samples_noreg(x, y, data_weights)
        combo_losses.append(c_loss.numpy())
    return np.vstack(combo_losses)


def train(train_dataset: tf.data.Dataset, data_plan: DataPlan,
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
        tr_losses = train_epoch(train_dataset, data_plan, train_step, tf.constant(epoch_temps[epoch]))

        # TODO: fix loss eval issues
        #c_loss = eval_losses(train_dataset, model)
        #e_loss.append(np.sum(c_loss))
        epoch_tr_losses.append(sum(tr_losses))
    return epoch_tr_losses
