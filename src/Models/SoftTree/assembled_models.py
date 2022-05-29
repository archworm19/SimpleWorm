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
        self.trainable_weights = self.decoder.get_trainable_weights()
        for rl in self.ref_layers:
            self.trainable_weights.extend(rl.get_trainable_weights())
                                   
    
    # TODO: probably need some prediction methods
    # > predict state? > predict average rep???

    def _weight_calc(self, forest_eval, data_weights):
        """Helper function for pred loss --> isolate important tests"""
        # scales:
        dw = tf.expand_dims(data_weights, 2)
        dw = tf.expand_dims(dw, 2)

        # --> batch_size x num_model x num_state x num_mix
        weights = (tf.expand_dims(forest_eval, 3) * tf.expand_dims(self.mix_coeffs, 0)
                        * dw)
        return weights


    def _pred_loss(self, forest_eval, y, data_weights):
        """prediction Loss function

        NOTE: num_state = num_leaves

        Args:
            forest_eval (tf.tensor): output of forest
                batch_size x num_model x num_leaves
            y (tf.tensor): target/truth ~ batch_size x gauss_dim
            data_weights (tf.tensor): weights on the data points
                batch_size x num_model
        
        Returns:
            tf.tensor: prediction losses for each batch elem, model combo
                batch_size x num_model
        """
        weights = self._weight_calc(forest_eval, data_weights)

        # log-likes of each gaussian:
        # --> batch_size x num_model x (num_state * num_mix)
        ll = self.decoder.calc_log_prob(y)
        ll_res = tf.reshape(ll, [-1, self.num_model, self.num_state, self.num_mix])
    
        # negate for loss:
        return -1 * tf.reduce_sum(weights * ll_res, axis=[2,3])

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
        return negent
    
    def _spread_loss(self):
        """Spread penalty for model layers

        Returns:
            tf.tensor scalar: spread error summed across all layers

        """
        sp_errs = [l.spread_error() for l in self.ref_layers]
        return tf.add_n(sp_errs)

    def full_loss(self, x, y, data_weights):
        """The full loss function ~ returns some intermediate losses

        Args:
            x (tf.tensor or List[tf.tensor]): inputs
                type/shape must match what is specified by layer/layer factory
            y (tf.tensor): target/truth ~ batch_size x 
            data_weights (tf.tensor): weights on the data points
                batch_size x num_model
        
        Returns:
            tf.tensor: combined loss; scalar
            tf.tensor: prediction loss; batch_size x num_model
            tf.tensor: forest loss; num_model (quantity implicitly averages across batch)
            tf.tensor: spread penalty; scalar
        """
        # forest evaluation: used by several loss funcs:
        forest_eval = self.soft_forest.eval(x)

        # batch_size x num_model
        ploss = self._pred_loss(forest_eval, y, data_weights)
        # num_model
        floss = self._forest_loss(forest_eval, data_weights)
        # scalar
        sloss = self._spread_loss()

        # combo loss = combine all losses with penalties
        combo_loss = (tf.reduce_sum(ploss) 
                        + self.forest_penalty * tf.reduce_sum(floss)
                        + self.spread_penalty * sloss)

        # return combo and intermediates
        return combo_loss, ploss, floss, sloss

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
        combo_loss, _ploss, _floss, _sloss = self.full_loss(x, y, data_weights)
        return combo_loss
    
    def get_trainable_weights(self):
        """Get the trainable weights

        Returns:
            List[tf.tensor]
        """
        return self.trainable_weights
