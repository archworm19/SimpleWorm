"""
    Train the Model!

    Data - model compatability requirements:
    > argument order = x1, x2, ..., y, data_weight
    > > all xs (different inputs) will be packaged into a list

"""
import tensorflow as tf
import numpy as np
from Models.SoftTree.assembled_models import AModel


# TODO: checking mechanisms to ensure shape matches


@tf.function
def train_step(model: AModel, optimizer, x, y, data_weights):
    with tf.GradientTape() as tape:
        loss = model.loss(x, y, data_weights)
    grads = tape.gradient(loss, model.get_trainable_weights())
    optimizer.apply_gradients(zip(grads, model.get_trainable_weights()))
    return loss


# TODO: this needs to be generalized! = pass in a recipe, maybe
# TODO: we maybe should include timestep identifiers
# TODO/NOTE: this is where data ordering assumptions are applied
def _gather_data(dat):
    x = dat[:-2]
    y = dat[-2]
    data_weights = dat[-1]
    return x, y, data_weights


def train_epoch(train_dataset, model: AModel, optimizer):
    for _step, dat in enumerate(train_dataset):
        x, y, data_weights = _gather_data(dat)
        train_step(model, optimizer, x, y, data_weights)


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


# TODO: support for temperature plans
def train(train_dataset, model: AModel, optimizer, num_epoch=3):
    """Train for a number of epochs

    Args:
        train_dataset (tf.dataset): tensorflow training dataset (should be batched)
        model (AModel): assembeled model
        optimizer (_type_): tensorflow optimizer
        num_epoch (int, optional): number of epochs. Defaults to 3.

    Returns:
        List[float]: train set loss after each training epoch
    """
    e_loss = []
    for _epoch in range(num_epoch):
        train_epoch(train_dataset, model, optimizer)
        c_loss = eval_losses(train_dataset, model)
        e_loss.append(np.sum(c_loss))
    return e_loss
