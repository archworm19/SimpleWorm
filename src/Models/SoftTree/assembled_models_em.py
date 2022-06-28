"""Assembled models trained via Expectaction Maximization
requires components for E-step"""

import abc
import tensorflow as tf
import numpy.random as npr
from Models.SoftTree.layers import LayerFactoryIface, var_construct
from Models.SoftTree.forest import build_forest
from Models.SoftTree.decoders import GaussFull
from Models.SoftTree.objective_funcs import forest_loss, spread_loss


class AModelEM(abc.ABC):
    # Assembeled Model interface
    # NOTE: EM framwork --> loss funcs will need latent state probs

    def latent_posterior(self, x, y, data_weights):
        """Calculate the posterior probabilities
        of the latent state ~ p(Z | X, theta_t)
        used by the E step in expectation maximization

        Args:
            x (tf.tensor or List[tf.tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.tensor): target/truth ~ batch_size x 
            data_weights (tf.tensor): weights on the data points
                batch_size x num_model
            latent_probs (tf.tensor): latent posteriors calculated from
                latent_posterior method ~ treated as static here
                batch_size x num_model x num_state x ...
        
        Returns:
            tf.tensor: combined loss; scalar
        """
        pass

    def loss(self, x, y, data_weights):
        """The loss function
        Used by the M-step of expectation maximization
        Includes regularization loss

        Args:
            x (tf.tensor or List[tf.tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.tensor): target/truth ~ batch_size x 
            data_weights (tf.tensor): weights on the data points
                batch_size x num_model
        
        Returns:
            tf.tensor: combined loss; scalar
        """
        pass

    def get_trainable_weights(self):
        """Get the trainable weights

        Returns:
            List[tf.tensor]
        """
        pass

    def loss_samples_noreg(self, x, y, latent_probs):
        """Loss for each sample in batch
        with no regularization

        Args:
            x (tf.tensor or List[tf.tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.tensor): target/truth ~ batch_size x dim of decoder domain
            latent_probs (tf.tensor): latent posteriors calculated from
                latent_posterior method ~ treated as static here
                batch_size x num_model x num_state x num_mix

        Returns:
            tf.tensor: loss without regularization
                batch_size x num_model x num_state x num_mix
        """
        pass


class GMMforestEM(AModelEM):
    """forest_penalty = scale on state occupancy loss
    spread penalty = scale on spread loss function (applied in layers)"""

    def __init__(self, depth: int, layer_factory: LayerFactoryIface,
                    num_mix: int, gauss_dim: int,
                    forest_penalty: float, spread_penalty: float,
                    rng: npr.Generator):
        """Model construction"""
        self.soft_forest, self.width, self.ref_layers = build_forest(depth, layer_factory)
        self.num_state = int(self.width ** depth)
        self.num_model = layer_factory.get_num_models()
        self.num_mix = num_mix
        self.gauss_dim = gauss_dim
        self.forest_penalty = forest_penalty
        self.spread_penalty = spread_penalty
        self.decoder = GaussFull(layer_factory.get_num_models(),
                                    self.num_state * num_mix,
                                    gauss_dim, rng)

        # variable for mixing coeffs: num_model x num_state x num_mixture
        self.mix_coeffs = var_construct(rng, [layer_factory.get_num_models(),
                                        self.num_state, num_mix])
        # trainable weights:
        self.trainable_weights = [self.mix_coeffs, self.decoder.get_trainable_weights()]
        for rl in self.ref_layers:
            self.trainable_weights.extend(rl.get_trainable_weights())

    def _get_mixture_prob(self):
        return tf.nn.softmax(self.mix_coeffs, axis=-1)

    def _forward_probabilities(self, y):
        """Calculate Gaussian Forward probabilities

        Args:
            y (tf.tensor): target/truth ~ batch_size x gauss_dim

        Returns:
            tf.tensor: batch_size x num_model x num_state x num_mix
        """
        # log probs --> batch_size x num_model x (num_state x num_mix)
        log_probs = self.decoder.calc_log_prob(y)
        # gaussian probs:
        gprobs = tf.math.exp(log_probs)
        # scale by mixture probs/coeffs
        full_probs = gprobs * self._get_mixture_prob()
        # reshape --> batch_size x num_model x num_state x num_mix
        return tf.reshape(full_probs, [-1, self.num_model, self.num_state, self.num_mix])
 
    def _posterior_probabilities(self, forward_probs):
        """calculate posterior probabilities for each state
        --> normalize across mixture within each state

        Args:
            forward_probs (tf.tensor): batch_size x num_model x num_state x num_mix

        Returns:
            tf.tensor: posterior probabilities
                same shape as input
        """
        return tf.math.divide(forward_probs,
                              tf.math.reduce_sum(forward_probs, axis=-1, keepdims=True))

    def latent_posterior(self, x, y, data_weights):
        """Calculate the posterior probabilities
        of the latent state ~ p(Z | X, theta_t)
        used by the E step in expectation maximization

        NOTE: calculates posterior within each mixture
        and DOES NOT use forest scaling here
        forest scaling gets factored in loss
        --> sum for single sample = 1 * num_model * num_state

        Args:
            x (tf.tensor or List[tf.tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.tensor): target/truth ~ batch_size x gauss_dim
            data_weights (tf.tensor): weights on the data points
                batch_size x num_model
        
        Returns:
            tf.tensor: probabilities of latent states
                for this model
                batch_size x num_model x num_state x num_mix
        """
        # forward probabilities
        for_probs = self._forward_probabilities(y)

        # posterior probabilities
        post_probs = self._posterior_probabilities(for_probs)

        # scale posteriors by forest probs
        return post_probs

    def _loss_samples_noreg(self, x, y, latent_probs):
        """The loss function for predictions
        If no regularization, loss --> _pred_loss

        Args:
            x (tf.tensor or List[tf.tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.tensor): target/truth ~ batch_size x 
            latent_probs (tf.tensor): latent posteriors calculated from
                latent_posterior method ~ treated as static here
                batch_size x num_model x num_state x num_mix
        
        Returns:
            tf.tensor: undreduced prediction loss without regularization
                batch_size x num_model x num_state x num_mix
            tf.tensor: forest probabilities
        """
        # run thru forest to get outer state probs --> batch_size x num_model x num_state
        forest_probs = self.soft_forest.eval(x)

        # forward log probabilities
        # --> batch_size x num_model x num_state x num_mix
        for_probs = self._forward_probabilities(y)

        # scaling forward log probabilties
        # scale = forest_probs * latent_probs; and data_weights select
        pred_loss =  (tf.reshape(forest_probs, [-1, self.num_model, self.num_state, 1]) * 
                      latent_probs *
                      for_probs)
        return pred_loss, forest_probs
    
    def loss_samples_noreg(self, x, y, latent_probs):
        """thin wrapper to make interface work"""
        pred_loss, _ = self._loss_samples_noreg(x, y, latent_probs)
        return pred_loss
    
    def loss(self, x, y, data_weights, latent_probs):
        """The loss function
        Used by the M-step of expectation maximization

        Args:
            x (tf.tensor or List[tf.tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.tensor): target/truth ~ batch_size x 
            data_weights (tf.tensor): weights on the data points
                batch_size x num_model
            latent_probs (tf.tensor): latent posteriors calculated from
                latent_posterior method ~ treated as static here
                batch_size x num_model x num_state x num_mix
        
        Returns:
            tf.tensor: combined loss; scalar
        """
        data_weights = tf.reshape(data_weights, [-1, self.num_model, 1, 1])

        # prediction loss
        pred_loss, forest_pred = tf.reduce_sum(data_weights * self._loss_samples_noreg(x, y, latent_probs))

        return (pred_loss + 
                self.forest_penalty * tf.reduce_sum(forest_loss(forest_pred, data_weights)) +
                self.spread_penalty * spread_loss(self.ref_layers))
