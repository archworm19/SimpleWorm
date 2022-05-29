"""Model Builder functions

    = common model configurations
"""
import tensorflow as tf
import numpy.random as npr
from Models.SoftTree.layers import LayerFactoryIface, var_construct
from Models.SoftTree.forest import build_forest
from Models.SoftTree.decoders import GaussFull

# TODO: interface


# TODO: have to make this interface work with tfrecords stuff
# > Trainer function
# > Trainer takes in a converter class or function or some other system
# --> figures out who should be sent to x vs. y vs. dataweights


# TODO: implement interface
class GMMforest:
    """forest_penalty = scale on state occupancy loss
    spread penalty = scale on spread loss function (applied in layers)"""

    def __init__(self, depth: int, layer_factory: LayerFactoryIface,
                    num_mix: int, gauss_dim: int,
                    forest_penalty: float, spread_penalty: float,
                    rng: npr.Generator):
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
        self.mix_coeffs = tf.nn.softmax(var_construct(rng, [layer_factory.get_num_models(),
                                                            self.num_state, num_mix]), axis=-1)
        # trainable weights:
        self.trainable_weights = self.ref_layers + self.decoder.get_trainable_weights()
    
    # TODO: probably need some prediction methods
    # > predict state? > predict average rep???

    def _pred_loss(self, forest_eval, y, data_weights):
        """prediction Loss function

        NOTE: num_state = num_leaves

        Args:
            forest_eval (tf.tensor): output of forest
                batch_size x num_model x num_leaves
            y (tf.tensor): target/truth ~ batch_size x 
            data_weights (tf.tensor): weights on the data points
                batch_size x num_model
        
        Returns:
            tf.scalar
        """
        # scales:
        dw = tf.expand_dims(data_weights, 2)
        dw = tf.expand_dims(data_weights, 2)
        # --> batch_size x num_model x num_state x num_mix
        weights = (tf.expand_dims(forest_eval, 3) * tf.expand_dims(self.mix_coeffs, 0)
                        * dw)

        # log-likes of each gaussian:
        # --> batch_size x num_model x (num_state * num_mix)
        ll = self.decoder.calc_log_prob(y)
        ll_res = tf.reshape(ll, [-1, self.num_model, self.num_state, self.num_mix])
    
        # negate for loss:
        return -1 * tf.reduce_sum(weights * ll_res)

    def _forest_loss(self, forest_eval, data_weights):
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

        Returns: tf.scalar
        """
        # reshape for legal mult:
        dw = tf.expand_dims(data_weights, 2)
        # weighted average across batch:
        # --> num_model x num_leaves
        wave = tf.math.divide(tf.reduce_sum(forest_eval * dw, axis=0),
                                tf.reduce_sum(dw, axis=0))
        # negentropy across state/leaves
        # --> num_model
        negent = tf.reduce_sum(wave * tf.log(wave), axis=1)
        return tf.reduce_mean(negent)
    
    def _spread_loss(self):
        """Spread penalty for model layers
        """
        sp_errs = [l.spread_error() for l in self.ref_layers]
        return tf.add_n(sp_errs)

    def loss(self, x, y, data_weights):
        """The loss function

        Args:
            x (tf.tensor or List[tf.tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.tensor): target/truth ~ batch_size x 
            data_weights (tf.tensor): weights on the data points
                batch_size x num_model
        """
        # forest evaluation: used by several loss funcs:
        forest_eval = self.soft_forest.eval(x)
        # combined loss function:
        return (self._pred_loss(forest_eval, y, data_weights)
                + self.forest_penalty * self._forest_loss(forest_eval, data_weights)
                + self.spread_penalty * self._spread_loss())
    
    def get_trainable_weights(self):
        """Get the trainable weights

        Returns:
            List[tf.tensor]
        """
        return self.trainable_weights
