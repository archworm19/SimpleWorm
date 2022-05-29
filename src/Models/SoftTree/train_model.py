"""
    Train the Model!

    Data - model compatability requirements:
    > argument order = x1, x2, ..., y, data_weight
    > > all xs (different inputs) will be packaged into a list

"""
import tensorflow as tf
import numpy as np

# TODO: need better design for loss evaluation



# TODO: checking mechanisms to ensure shape matches


# TODO: can train step take in model / optimizer??? I believe so
@tf.function
def train_step(model, optimizer, x, y, data_weights):
    with tf.GradientTape() as tape:
        loss = model.loss(x, y, data_weights)
    grads = tape.gradient(loss, model.get_trainable_weights())
    optimizer.apply_gradients(zip(grads, model.get_trainable_weights()))
    return loss


def test_train_step(model, x, y, data_weights):
    with tf.GradientTape() as tape:
        loss = model.loss(x, y, data_weights)
        grads = tape.gradient(loss, model.get_trainable_weights())
        missing = []
        for i, g in enumerate(grads):
            if g is None:
                missing.append(i)
        print(missing)
            


# TODO: we maybe should include timestep identifiers
# TODO/NOTE: this is where data ordering assumptions are applied
def _gather_data(dat):
    x = dat[:-2]
    y = dat[-2]
    data_weights = dat[-1]
    return x, y, data_weights


def train_epoch(train_dataset, model, optimizer):
    for _step, dat in enumerate(train_dataset):
        x, y, data_weights = _gather_data(dat)
        train_step(model, optimizer, x, y, data_weights)
        #test_train_step(model, x, y, data_weights)


# TODO: this is poor design ~ depends on concretions
def eval_losses(train_dataset, model):
    """Get prediction loss for each batch, model combination

    Args:
        train_dataset (tf.dataset): training dataset
        model: the model  TODO: interface

    Returns:
        np.ndarray: losses
            number of batches x number of models
    """
    combo_losses, pred_losses = [], []
    for _step, dat in enumerate(train_dataset):
        x, y, data_weights = _gather_data(dat)
        c_loss, p_loss, _, _ = model.full_loss(x, y, data_weights)
        combo_losses.append(c_loss.numpy())
        pred_losses.append(p_loss.numpy())
    return np.vstack(combo_losses), np.vstack(pred_losses)


# TODO: overall training
def train(train_dataset, model, optimizer, num_epoch=3):
    # TODO: docstring; gets loss at end of each epoch
    e_loss = []
    for _epoch in range(num_epoch):
        train_epoch(train_dataset, model, optimizer)
        c_loss, _ = eval_losses(train_dataset, model)
        e_loss.append(np.sum(c_loss))
    return e_loss
