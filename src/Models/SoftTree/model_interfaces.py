"""Model Builder functions

    = common model configurations
"""
import abc
import tensorflow as tf
from typing import Union, List


class GateSubModel(abc.ABC):
    # Some models are composed of multiple models
    # Ex: have a gating model and a prediction model
    def get_num_state(self):
        """Get the number of states

        Returns:
            int: number of states
        """
        pass

    def get_state_probs(self, x: List[tf.Tensor]):
        """Get State probabilities

        Args:
            x (List[tf.Tensor]): input
                batch_size x x1 x x2 x ...
                where x1, x2, ... match layer factory
                    used to create this model 
        
        Returns:
            tf.Tensor: state probabilities for batch, model combo
                batch_size x ... x num_state x ...
        """
        pass

    def get_trainable_weights(self):
        """Get the trainable weights

        Returns:
            List[tf.tensor]
        """
        pass

    def regularization_loss(self,
                            x: List[tf.Tensor],
                            data_weights: tf.Tensor,
                            reg_epoch_scale: float):
        """Regularization loss
        
        Args:
            x (List[tf.Tensor]): input
            data_weights (tf.Tensor): weights on the data points
                batch_size x num_model
            reg_epoch_scale (float): how much to scale regularization
                by as a function of epoch == f(temperature)

        Returns:
            scalar: sum loss across batch/models
        """
        pass


class PredSubModel(abc.ABC):
    def get_num_state(self):
        """Get the number of states

        Returns:
            int: number of states
        """
        pass

    def get_preds(self, x: List[tf.Tensor]):
        """Get predictions

        Args:
            x (List[tf.Tensor]): input
                batch_size x x1 x x2 x ...
                where x1, x2, ... match layer factory
                    used to create this model 
        
        Returns:
            tf.tensor: predictions
                batch_size x ... x prediction dimension size x ...
        """
        pass

    def get_trainable_weights(self):
        """Get the trainable weights

        Returns:
            List[tf.tensor]
        """
        pass

    def regularization_loss(self,
                            x: List[tf.Tensor],
                            data_weights: tf.Tensor,
                            reg_epoch_scale: float):
        """Regularization loss
        
        Args:
            x (List[tf.Tensor]): input
            data_weights (tf.Tensor): weights on the data points
                batch_size x ...
                often batch_size x num_parallel_model
            reg_epoch_scale (float): how much to scale regularization
                by as a function of epoch == f(temperature)

        Returns:
            scalar: sum loss across batch/models
        """
        pass


class AModel(abc.ABC):
    # Assembeled Model interface
    def loss(self, x: List[tf.Tensor], y: tf.Tensor, data_weights: tf.Tensor):
        """The loss function

        Args:
            x (List[tf.Tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.Tensor): target/truth
                batch_size
            data_weights (tf.tensor): weights on the data points
                batch_size x ...
                often batch_size x num_parallel_model
        
        Returns:
            tf.tensor: combined loss; scalar
        """
        pass

    def get_trainable_weights(self):
        """Get the trainable weights

        Returns:
            List[tf.Tensor]
        """
        pass

    def loss_samples_noreg(self, x: List[tf.Tensor], y: tf.Tensor):
        """Loss for each sample in batch

        Args:
            x (Union[tf.Tensor, List[tf.Tensor]]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.Tensor): target/truth
                batch_size

        Returns:
            tf.Tensor: combined loss
                batch_size x ...
                often times: batch_size x num_parallel_models
        """
        pass

    def get_preds(self, x: List[tf.Tensor]):
        """Predictions

        Args:
            x (List[tf.Tensor]): inputs
                type/shape must match what is specified by layer/layer factory
        
        Returns:
            List[tf.Tensor]: set of predictions (strong model-dependence)
                batch_size x ...
                often times: batch_size x num_parallel_models
        """
        pass

    # TODO: probably need some prediction methods
    # > predict state? > predict average rep???


class AModelEM(abc.ABC):
    # Assembeled Model interface
    # NOTE: EM framwork --> loss funcs will need latent state probs

    def latent_posterior(self, x: List[tf.Tensor], y: tf.Tensor):
        """Calculate the posterior probabilities
        of the latent state ~ p(Z | X, theta_t)
        used by the E step in expectation maximization

        Args:
            x (List[tf.Tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.Tensor): target/truth ~ batch_size x 
        
        Returns:
            tf.Tensor: posterior probabilties for each latent state z
                batch_size x ... x num_z_dimensions
                Common example: trying to solve a GMM in N different subspaces
                    number of mixtures = M --> latents in half-bayesian formulation
                    = categorical distro --> z = i --> observation from the ith gaussian
                    this will return batch_size x num_state x num_gaussians_per_state
                    = batch_size x N x M
        """
        pass

    def loss(self, x: List[tf.Tensor], y: tf.Tensor,
                   data_weights: tf.Tensor, latent_probs: tf.Tensor):
        """The loss function
        Used by the M-step of expectation maximization
        Includes regularization loss

        Args:
            x (List[tf.Tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.Tensor): target/truth
                batch_size
            data_weights (tf.Tensor): weights on the data points
                batch_size x ...
                often batch_size x num_parallel_model
            latent_probs (tf.Tensor): latent posteriors calculated from
                latent_posterior method ~ treated as static here
                batch_size x ... x num_z_state
                often batch_size x num_parallel_model x num_state x num_z_state
                must be normalized across z states (within other states if applicable)
        
        Returns:
            tf.Tensor: combined loss; scalar
                includes regularization penalties
        """
        pass

    def get_trainable_weights(self):
        """Get the trainable weights

        Returns:
            List[tf.Tensor]
        """
        pass

    def loss_samples_noreg(self, x: List[tf.Tensor],
                                 y: tf.Tensor, latent_probs: tf.Tensor):
        """Loss for each sample in batch
        with no regularization

        Args:
            x (List[tf.Tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.Tensor): target/truth
                batch_size
            data_weights (tf.Tensor): weights on the data points
                batch_size x ...
                often batch_size x num_parallel_model
            latent_probs (tf.Tensor): latent posteriors calculated from
                latent_posterior method ~ treated as static here
                batch_size x ... x num_z_state
                often batch_size x num_parallel_model x num_state x num_z_state
                must be normalized across z states (within other states if applicable)

        Returns:
            tf.Tensor: loss without regularization
                batch_size x ... x num_z_state x ...
                often: batch_size x num_parallel_model x num_state x num_z_state
        """
        pass
