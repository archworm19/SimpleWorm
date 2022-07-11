""" 
    Useful objective functions
    > those defined here will be used by assembled_models

    Key Concept: parallel dimensions vs. reduction dimensions
    > reduction operations occur across reduction dimensions
    > remaining dims are [batch_size] + [reduction dimensions]
    > reduction dimensions should be specified in init methods

"""
import abc
from typing import List
import tensorflow as tf
from Models.SoftTree.layers import LayerIface
from Models.SoftTree.decoders import log_prob_iface


class ObjFunc(abc.ABC):
    # rely on specific model types to reduce across samples
    def loss_sample(self, predictions: tf.Tensor, y: tf.Tensor):
        """loss for each model, sample combination

        Args:
            predictions (tf.Tensor): predictions
            y (tf.Tensor): truths

        Returns:
            tf.tensor: batch_size x parallel_dimensions ...
        """
        pass


class BinaryLoss(ObjFunc):

    def __init__(self, total_dims: int):
        """
        don't really need total_dims but maybe useful later?
        Assumes no reduction dim ~ already done
        total_dims should include batch dimension"""
        self.total_dims = total_dims
        self.new_shape = [-1] + [1 for _ in range(total_dims-1)]

    def loss_sample(self, predictions: tf.Tensor, y: tf.Tensor):
        """binary cross-entropy loss using sigmoid function
        p = exp(pred) / (exp(pred) + 1)
        x-entropy = y log p + (1 - y) log (1 - p)
        > log p = pred - log(exp(pred) + 1)
        > log(1-p) = log[ exp(pred) + 1 - exp(pred) / (exp(pred) + 1)]
        = log [1 / (exp(pred) + 1)] = -log(exp(pred) + 1)
        in sum:
        y [pred - log(exp(pred) + 1)] - (1 - y)log(exp(pred) + 1)
        K = log exp(pred) + 1
        --> y pred - y K - K + yK = y pred - K
        = y pred - log(exp(pred) + 1)
        ... same as multinomial but 1 for off logit

        Args:
            predictions (tf.Tensor): prediction logits
            y (tf.Tensor): binary/boolean tensor

        Returns:
            tf.Tensor: num_parallel_dims ...
        """
        # reshape y:
        y2 = tf.reshape(y, self.new_shape)
        # norm term = logsumexp
        K = tf.math.log(tf.math.exp(predictions) + 1)
        # primary term
        yp = y2 * predictions
        log_like = yp - K
        return -1 * log_like


class MultinomialLoss(ObjFunc):

    def __init__(self, class_dim: int):
        """
        class_dim = classification dimension = specifies
            dimension containing classes = reduction dimension
        total_dims = total number of dimensions including batch dimension"""
        self.class_dim = class_dim

    def loss_sample(self, predictions: tf.Tensor, y: tf.Tensor):
        """multinomial loss for each sample, model combination
        Multinomial log-likelihood = 
            sum_i [ y_i * log pred_i]
            where y = one-hot vector
        Softmax construction:
            pred_i_prob = exp(pred_i) / sum[ exp(pred_j) ]
            log pred_i_prob = pred_i - log sum[ exp(pred_j) ]
             = pred_i - k(x)
            ... sub into multinomial -->
            sum_i [ y_i * [pred_i - k(x)]]
            = sum_i[ y_i * pred_i] - k(x)
            ... assuming y_i is one-hot vector

        Args:
            predictions (tf.Tensor): predictions logits
                batch_size x ...
                where class_dim specifies classification dimension idx
            y (tf.Tensor): truths = tensor of indices (index of correct class)
                batch_size

        Returns:
            tf.Tensor: num_parallel_dims ...
        """
        # norm term
        kx = tf.math.log(tf.reduce_sum(tf.math.exp(predictions), axis=self.class_dim))
        # gather the correct dimension (and reduce)
        pred_i = tf.reduce_sum(tf.gather(predictions, tf.reshape(y, [-1,1]), axis=self.class_dim, batch_dims=1),
                               axis=self.class_dim)
        # log likelihood
        log_like = pred_i - kx
        # negate for loss
        return -1 * log_like


class QuantileLoss(ObjFunc):

    def __init__(self, total_dims: int, tau_dim: int, taus: tf.Tensor):
        """
        total_dims includes batch_dim
        tau_dim = dimension containing prediction for each quantile
        taus (tf.Tensor): taus define the quantile
            must be between 0 and 1, non-inclusive
            num_quantile
        """
        self.taus = taus
        self.total_dims = total_dims
        assert(tf.reduce_all(taus > 0.)), "illegal taus low"
        assert(tf.reduce_all(taus < 1.)), "illegal taus hi"
        self.new_yshape = [-1] + [1 for _ in range(total_dims-1)]
        # reshape taus:
        new_taushape = [1 for _ in range(total_dims)]
        new_taushape[tau_dim] = len(taus)
        self.taus_shaped = tf.reshape(self.taus, new_taushape)

    def loss_sample(self, predictions: tf.Tensor, y: tf.Tensor):
        """Quantile loss = pinball loss
        = [ (tau - 1) sum_[yi < q] (yi - q)
            + tau sum_[yi >= q] (yi - q)]
        where q = prediction
        and tau defines the quantile
            Ex: tau = .5 = 50% quantile = median

        Args:
            predictions (tf.tensor): predictions
                batch_size x ... x num_quantile x ...
            y (tf.tensor): truths
                batch_size
                NOTE: quantile_loss is only defined for
                    scalar truths/predictions

        Returns:
            tf.tensor: quantile loss
                shape matches predictions
        """
        # dt = (y_i - q)
        # --> batch_size x num_model x num_quantile
        dt = tf.reshape(y, self.new_yshape) - predictions

        # left side = (tau - 1) terms
        left = ((self.taus_shaped - 1) *
                tf.stop_gradient(tf.cast(dt < 0, predictions.dtype)) *
                dt)

        # right side = tau terms
        right = (self.taus_shaped *
                tf.stop_gradient(tf.cast(dt >= 0, predictions.dtype)) *
                dt)

        return left + right
