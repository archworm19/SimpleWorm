"""Custom keras model inferface

    Assumptions:
        1. model output = Dict[str, tf.Tensor]
            = mapping from names to tensors
        2. loss is one of the model outputs
"""
import tensorflow as tf
import keras
from keras.models import Model as KModel


class CustomModel(KModel):

    def __init__(self, loss_name: str, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        self.loss_name = loss_name
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # ASSUMPTION 1
            y_pred = self(x, training=True)
            # ASSUMPTION 2
            loss = y_pred[self.loss_name]
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss)
        return {"train_loss": self.total_loss_tracker.result()}
