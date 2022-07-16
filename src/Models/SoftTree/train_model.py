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


@tf.function
def train_step(model: AModel, optimizer, x, y, data_weights,
                epoch_temperature: tf.constant):
    with tf.GradientTape() as tape:
        loss = model.loss(x, y, data_weights, epoch_temperature)
    grads = tape.gradient(loss, model.get_trainable_weights())
    optimizer.apply_gradients(zip(grads, model.get_trainable_weights()))
    return loss


# TODO: optimizer type?
def train_epoch(train_dataset: tf.data.Dataset, data_plan: DataPlan,
                model: AModel, optimizer,
                epoch_temperature: tf.constant):
    tr_losses = []
    for _step, dat in enumerate(train_dataset):
        x, y, data_weights = _gather_data(dat, data_plan)
        train_step(model, optimizer, x, y, data_weights, epoch_temperature)
        tr_losses.append(model.loss(x, y, data_weights, epoch_temperature).numpy())
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
            model: AModel, optimizer,
            num_epoch: int = 3, epoch_temps: List[float] = [1., 1., 1.]):
    """Train for a number of epochs

    Args:
        train_dataset (tf.dataset): tensorflow training dataset (should be batched)
        model (AModel): assembeled model
        optimizer (_type_): tensorflow optimizer
        num_epoch (int, optional): number of epochs. Defaults to 3.
        epoch_temps (List[float]): regularization scales for each epoch
            when = 1. --> no scaling of regularizers

    Returns:
        List[float]: train set loss after each training epoch
    """
    assert(len(epoch_temps) == num_epoch)
    epoch_tr_losses = []
    for epoch in range(num_epoch):
        tr_losses = train_epoch(train_dataset, data_plan, model, optimizer, tf.constant(epoch_temps[epoch]))

        # TODO: fix loss eval issues
        #c_loss = eval_losses(train_dataset, model)
        #e_loss.append(np.sum(c_loss))
        epoch_tr_losses.append(sum(tr_losses))
    return epoch_tr_losses
