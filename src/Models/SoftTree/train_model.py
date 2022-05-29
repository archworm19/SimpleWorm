"""
    Train the Model!

    Data - model compatability requirements:
    > argument order = x1, x2, ..., y, data_weight
    > > all xs (different inputs) will be packaged into a list

"""
import tensorflow as tf

# TODO: checking mechanisms to ensure shape matches


# TODO: can train step take in model / optimizer???
@tf.function
def train_step(model, optimizer, x, y, data_weights):
    with tf.GradientTape() as tape:
        loss = model.loss(x, y, data_weights)
    grads = tape.gradient(loss, model.get_trainable_weights())
    optimizer.apply_gradients(zip(grads, model.get_trainable_weights()))
    return loss


def train_epoch(train_dataset, model, optimizer, x, y, data_weights):
    for _step, dat in enumerate(train_dataset):
        x = dat[:-2]
        y = dat[-2]
        data_weights = dat[-1]
        train_step(model, optimizer, x, y, data_weights)


# TODO: overall training