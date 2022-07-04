"""Model Builder functions

    = common model configurations
"""
import abc


class GateSubModel(abc.ABC):
    # Some models are composed of multiple models
    # Ex: have a gating model and a prediction model
    def get_num_state(self):
        """Get the number of states

        Returns:
            int: number of states
        """
        pass

    def get_state_probs(self, x):
        """Get State probabilities

        Args:
            x (tf.tensor): input
                batch_size x x1 x x2 x ...
                where x1, x2, ... match layer factory
                    used to create this model 
        
        Returns:
            tf.tensor: state probabilities for batch, model combo
                batch_size x num_model x num_state
        """
        pass

    def get_trainable_weights(self):
        """Get the trainable weights

        Returns:
            List[tf.tensor]
        """
        pass

    def regularization_loss(self, x):
        """Return the regularization loss

        Args:
            x (tf.tensor): input
                batch_size x x1 x x2 x ...
                where x1, x2, ... match layer factory
                    used to create this model 
        
        Returns:
            tf.tensor: scalar
        """
        pass


class PredSubModel(abc.ABC):
    def get_num_state(self):
        """Get the number of states

        Returns:
            int: number of states
        """
        pass

    def get_preds(self, x):
        """Get predictions

        Args:
            x (tf.tensor): input
                batch_size x x1 x x2 x ...
                where x1, x2, ... match layer factory
                    used to create this model 
        
        Returns:
            tf.tensor: predictions
                batch_size x num_model x num_state x num_pred
        """
        pass

    def get_trainable_weights(self):
        """Get the trainable weights

        Returns:
            List[tf.tensor]
        """
        pass

    def regularization_loss(self, x):
        """Return the regularization loss

        Args:
            x (tf.tensor): input
                batch_size x x1 x x2 x ...
                where x1, x2, ... match layer factory
                    used to create this model 
        
        Returns:
            tf.tensor: scalar
        """
        pass


class AModel(abc.ABC):
    # Assembeled Model interface
    def loss(self, x, y, data_weights):
        """The loss function

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

    def loss_samples_noreg(self, x, y):
        """Loss for each sample in batch

        Args:
            x (tf.tensor or List[tf.tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.tensor): target/truth ~ batch_size x 

        Returns:
            tf.tensor: combined loss
                batch_size x num_model
        """
        pass

    def get_preds(self, x):
        """Predictions

        Args:
            x (tf.tensor or List[tf.tensor]): inputs
                type/shape must match what is specified by layer/layer factory
        
        Returns:
            tf.tensor: predictions
                batch_size x num_model x ...
        """
        pass

    # TODO: probably need some prediction methods
    # > predict state? > predict average rep???


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
