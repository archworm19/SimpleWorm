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
    def loss_sample(self, predictions, y):
        """loss for each model, sample combination

        Args:
            predictions (tf.tensor): predictions
            y (tf.Tensor): truths = one-hot vector represented by indices
                batch_size

        Returns:
            tf.tensor: batch_size x parallel_dimensions ...
        """
        pass


# TODO: binary classification loss
class BinaryLoss(ObjFunc):

    def __init__(self, total_dims: int):
        """
        don't really need total_dims but maybe useful later?
        Assumes no reduction dim ~ already done"""
        self.total_dims = total_dims

    # TODO: adapt to work for tensor
    # ... relatively easy if know total dims...
    def loss_sample(self, predictions: tf.Tensor, y: bool):
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
            y (bool): target

        Returns:
            tf.Tensor: num_parallel_dims ...
        """
        yint = y * 1
        K = tf.math.log(tf.math.exp(predictions) + 1)
        yp = yint * predictions
        log_like = yp - K
        return -1 * log_like


class MultinomialLoss(ObjFunc):

    def __init__(self, class_dim: int, total_dims: int):
        """
        class_dim = classification dimension = specifies
            dimension containing classes = reduction dimension
        total_dims = total number of dimensions including batch dimension"""
        self.class_dim = class_dim
        # slicing: beginning tensor
        self.begin_tensor = tf.zeros([total_dims], tf.int32)
        # slicing: length tensor
        lt = [-1 for _ in range(total_dims)]
        lt[class_dim] = 1
        self.len_tensor = tf.constant(lt, tf.int32)
        # slicing: begin mask tensor:
        bmt = [0 for _ in range(total_dims)]
        bmt[class_dim] = 1
        self.begin_mask = tf.constant(bmt, tf.int32)

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
            y (tf.Tensor): truths = one-hot vector represented by indices
                batch_size

        Returns:
            tf.Tensor: num_parallel_dims ...
        """
        # TODO: slice operation is more complex now! with batches
        # HOW? for each batch --> slice a different section
        # ... sounds like a job for gather_nd?
        # TODO: pretty sure just tf.gather_nd(predictions, reshape(y, (-1,1)) but need to mess around with it

        kx = tf.math.log(tf.reduce_sum(tf.math.exp(predictions), axis=self.class_dim))
        # use slicing to get correct dim:
        begin_local = self.begin_tensor + self.begin_mask * y
        pred_i = tf.reduce_sum(tf.slice(predictions, begin_local, self.len_tensor), axis=self.class_dim)
        # log likelihood
        log_like = pred_i - kx
        # negate for loss
        return -1 * log_like


# TODO: make this work for new, more general system
class QuantileLoss(ObjFunc):

    def __init__(self, taus):
        """
        taus (tf.tensor): taus define the quantile
            num_quantile
        """
        self.taus = taus

    def loss_sample(self, predictions, y):
        """Quantile loss = pinball loss
        = [ (tau - 1) sum_[yi < q] (yi - q)
            + tau sum_[yi >= q] (yi - q)]
        where q = prediction
        and tau defines the quantile
            Ex: tau = .5 = 50% quantile = median

        Args:
            predictions (tf.tensor): predictions
                batch_size x num_model x num_quantile
            y (tf.tensor): truths
                batch_size
                NOTE: quantile_loss is only defined for
                    scalar truths/predictions

        Returns:
            tf.tensor: batch_size x num_model
        """
        # dt = (y_i - q)
        # --> batch_size x num_model x num_quantile
        dt = tf.reshape(y, [-1, 1, 1]) - predictions

        # reshape taus:
        re_taus = tf.reshape(self.taus, [1, 1, -1])

        # left side = (tau - 1) terms
        left = ((re_taus - 1) *
                tf.stop_gradient(dt < 0) *
                dt)

        # right side = tau terms
        right = (re_taus *
                tf.stop_gradient(dt >= 0) *
                dt)

        return tf.reduce_sum(left + right, axis=-1)


# TODO: move this to SoftForest --> make it fulfill model interfaces
def forest_loss(forest_eval, data_weights):
    """Forest evaluation Loss
    Designed to force the forest outputs to occupy all states
    equally on average

    > Each model outputs N states = N leaves
    > > for M batch_size
    > weighted average across batch, within model
    > > weights = data weights
    > > Gets us N-len vector for each model (vals gauranteed between 0, 1)
    > Calc entropy for each model
    > Return neg-entropy as loss --> maximize entropy

    Args:
        forest_eval (tf.tensor): output of forest
            batch_size x num_model x num_leaves
        data_weights (tf.tensor): weights on the data points
            batch_size x num_model

    Returns:
        tf.tensor: prediction losses for given batch + each model
            num_model
    """
    # reshape for legal mult:
    dw = tf.expand_dims(data_weights, 2)
    # weighted average across batch:
    # --> num_model x num_leaves
    wave = tf.math.divide(tf.reduce_sum(forest_eval * dw, axis=0),
                            tf.reduce_sum(dw, axis=0))
    # negentropy across state/leaves
    # --> num_model
    negent = tf.reduce_sum(wave * tf.math.log(wave), axis=1)
    # scale by batch_size to make it batch_size dependent
    return negent * tf.cast(tf.shape(forest_eval)[0], negent.dtype)


def spread_loss(ref_layers: List[LayerIface]):
    """Spread penalty for model layers

    Returns:
        tf.tensor scalar: spread error summed across all layers

    """
    sp_errs = [l.spread_error() for l in ref_layers]
    return tf.add_n(sp_errs)


if __name__ == "__main__":
    # TODO: move this to test file
    import numpy as np
    ar_np = np.zeros((3, 2, 4))
    # TESTING: class 0 --> loss to 0
    ar_np[1, 1, 0] = 100
    # TESTING: class 1 --> huge loss
    ar_np[2, 1, 1] = 100
    ar = tf.constant(ar_np)
    ML = MultinomialLoss(2, 3)
    print(ML.loss_sample(ar, 0))

    BL = BinaryLoss(3)
    # if use ar again with this dude --> classes = hidden 4th dim
    #print(BL.loss_sample(ar, ))
