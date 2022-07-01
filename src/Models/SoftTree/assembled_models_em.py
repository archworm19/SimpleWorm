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

    def latent_posterior(self, x, y):
        """Calculate the posterior probabilities
        of the latent state ~ p(Z | X, theta_t)
        used by the E step in expectation maximization

        Args:
            x (tf.tensor or List[tf.tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.tensor): target/truth ~ batch_size x 
        
        Returns:
            tf.tensor: combined loss; scalar
        """
        pass

    def loss(self, x, y, data_weights, latent_probs):
        """The loss function
        Used by the M-step of expectation maximization
        Includes regularization loss

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
    # TODO/NOTE: should be able to make a more generic version of this 
    # that can be injected with a state predictor model (like a soft forest)
    """forest_penalty = scale on state occupancy loss
    spread penalty = scale on spread loss function (applied in layers)
    
    
    SSM (state-space model) ~ Half-Bayesian approach (EM)
    X = output here; input factored into theta
    > Goal = estimate L(theta; X) = integral_z [p(X, Z | theta]
    > We use Expectation Maximization approx -->
    > exp_[Z|X, theta_t] [log L(theta; X, Z)]
    > = exp_z [log P(X, Z | theta_t)] = exp_z [log p(X | Z, theta_t) + log p(Z, theta_t)]
    For EM, need to estimate p(Z | X, theta_t) ~ expectation over this distro
    > p(Z | X, theta_t) = p(X | Z, theta_t) p(Z | theta_t) / norm ~ from Bayes
    > Assumption for this model! p(X | Z, theta_t) = p(X | Z, theta_gauss)
    > > theta_gauss = the gaussian/decoding subset of model params theta
    """

    # TODO: gauss mean and variance initialization is super critical
    # --> add initialization scheme!
    def __init__(self, depth: int, layer_factory: LayerFactoryIface,
                    gauss_dim: int,
                    forest_penalty: float, spread_penalty: float,
                    rng: npr.Generator):
        """Model construction"""
        self.soft_forest, self.width, self.ref_layers = build_forest(depth, layer_factory)
        self.num_state = int(self.width ** depth)
        self.num_model = layer_factory.get_num_models()
        self.gauss_dim = gauss_dim
        self.forest_penalty = forest_penalty
        self.spread_penalty = spread_penalty
        self.decoder = GaussFull(layer_factory.get_num_models(),
                                    self.num_state,
                                    gauss_dim, rng)

        # variable for mixing coeffs: num_model x num_state x num_mixture
        self.mix_coeffs = var_construct(rng, [layer_factory.get_num_models(),
                                              self.num_state])
        # trainable weights:
        self.trainable_weights = [self.mix_coeffs]
        self.trainable_weights.extend(self.decoder.get_trainable_weights())
        for rl in self.ref_layers:
            self.trainable_weights.extend(rl.get_trainable_weights())

    def get_trainable_weights(self):
        return self.trainable_weights

    def _get_mixture_prob(self):
        """returns num_model x num_state tensor"""
        return tf.nn.softmax(self.mix_coeffs, axis=-1)

    def latent_posterior(self, x, y):
        """Calculate the posterior probabilities
        For EM, need to estimate p(Z | X, theta_t) ~ expectation over this distro
        > p(Z | X, theta_t) = p(X | Z, theta_t) p(Z | theta_t) / norm ~ from Bayes
        > Assumption for this model! p(X | Z, theta_t) = p(X | Z, theta_gauss)
        > > theta_gauss = the gaussian/decoding subset of model params theta

        NOTE: occurs separately from opt (M step) --> might as well call forest directly

        Args:
            x (tf.tensor or List[tf.tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.tensor): target/truth ~ batch_size x gauss_dim
        
        Returns:
            tf.tensor: probabilities of latent states
                for this model
                batch_size x num_model x num_state
        """
        # p(X | Z, theta_gauss)
        # log probs --> batch_size x num_model x num_state
        # --> batch_size x num_model x num_state
        log_probs = self.decoder.calc_log_prob(y)
        # gaussian probs:
        gprobs_nomix = tf.math.exp(log_probs)
        # mixure coeffs:
        gprobs = gprobs_nomix * tf.expand_dims(self._get_mixture_prob(), 0)

        # p(Z | theta_t)
        # --> batch_size x num_model x num_state
        forest_probs = self.soft_forest.eval(x)

        # forward probs:
        fprobs = gprobs * forest_probs

        # normalize --> posterior:
        return tf.stop_gradient(tf.math.divide(fprobs, tf.reduce_sum(fprobs, axis=-1, keepdims=True)))


    def _loss_samples_noreg(self, x, y, latent_probs):
        """The loss function for predictions; noreg = no regularization
        > Goal = estimate L(theta; X) = integral_z [p(X, Z | theta]
        > We use Expectation Maximization approx -->
        > exp_[Z|X, theta_t] [log L(theta; X, Z)]
        > = exp_z [log P(X, Z | theta_t)] = exp_z [log p(X | Z, theta_t) + log p(Z, theta_t)]

        # TODO: for gaussian, probs faster (but less stable) to not use gradient descent

        Args:
            x (tf.tensor or List[tf.tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.tensor): target/truth ~ batch_size x 
            latent_probs (tf.tensor): latent posteriors calculated from
                latent_posterior method ~ treated as static here
                batch_size x num_model x num_state
        
        Returns:
            tf.tensor: undreduced prediction loss without regularization
                batch_size x num_model x num_state x num_mix
            tf.tensor: forest probabilities
        """
        # log p(X | Z, theta_t) = log p(X | Z, theta_gauss) by assumption
        # log probs --> batch_size x num_model x num_state
        # --> batch_size x num_model x num_state
        log_probs = self.decoder.calc_log_prob(y)
        # log mixture probs
        # TODO: this term can def be simplified
        log_mix = tf.math.log(tf.expand_dims(self._get_mixture_prob(), 0))

        # log p(Z, theta_t)
        # --> batch_size x num_model x num_state
        # TODO: more math can be done for this as well...
        forest_pred = self.soft_forest.eval(x)
        log_forest = tf.math.log(forest_pred)

        # scale probs and return negative
        return -1 * latent_probs * (log_probs + log_mix + log_forest), forest_pred
    
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
        # prediction loss full:
        pred_loss_full, forest_pred = self._loss_samples_noreg(x, y, tf.stop_gradient(latent_probs))
        pred_loss_full = tf.expand_dims(data_weights, 2) * pred_loss_full
        pred_loss = tf.reduce_sum(pred_loss_full)

        return (pred_loss + 
                self.forest_penalty * tf.reduce_sum(forest_loss(forest_pred, data_weights)) +
                self.spread_penalty * spread_loss(self.ref_layers))
