"""
    Layers

    Factories that generate layers
"""
from typing import List
import abc
import numpy as np
import numpy.random as npr
import tensorflow as tf


# utilities


def var_construct(rng: npr.Generator, v_shape: List[int], v_scale: int = 1.):
    """Construct a uniform random tensorflow variable
    of specified shape

    Args:
        v_shape (List[int]): full shape of variable
    """
    v_np = v_scale * (rng.random(v_shape) - 0.5)
    return tf.Variable(v_np.astype(np.float32))


# layers


class LayerIface(abc.ABC):

    def eval(self, x: List[tf.Tensor]):
        """Evaluate layer

        Args:
            x (List[tf.Tensor]): input tensor(s)
        
        Returns:
            tf.Tensor: reduced tensor
                batches x parallel_dim1 x parallel_dim2 x ... x width
        """
        pass

    def get_width(self):
        """Report the width"""
        pass

    def spread_error(self):
        """Calculate the spread error for the layer

        Returns:
            tf.tensor scalar: L1 norm summed across all spread components
                scalar
        """
        pass

    def get_trainable_weights(self):
        """Get the trainable weights
        
        Returns:
            List[tf.tensor]
        """
        pass


class LayerBasic(LayerIface):
    """Key: shared and spread components
    > N shared components and M spread components per shared comp
    > For each shared component --> there is a model group
        --> weights for model group = w_shared (single) + w_spread
            = M spread components"""

    def __init__(self,
                 base_models: int,
                 models_per_base: int,
                 xshape: List[int],
                 width: int,
                 rng: npr.Generator):
        """Initialize a basic layer
        > total num_models = base_models * models_per_base
        > special case: base_models = 0 --> there are
            [models_per_base] independent models

        Args:
            num_models (int): number of base models
            models_per_base (int): number of models per
                each base model. These models all share a
                common component
            xshape (List[int]): shape of extra dimensions
                doesn't include parallel models or batch_size
            width (int): number of outputs from the layer
            rng (npr.Generator): numpy random generator
        """
        assert(base_models >= 0), "positive base models number"
        assert(models_per_base > 0), "models per base 1 or more"
        self.rng = rng
        self.width = width
        self.base_models = base_models
        self.models_per_base = models_per_base
        self.xshape = xshape
        # coefficients
        if base_models > 0:
            self.shared_comps = var_construct(rng, [base_models, 1, width] + xshape)
        else:
            self.shared_comps = 0
        # SPREAD component = base_models x models_per_base x width
        self.spread_comps = var_construct(rng, [base_models, models_per_base, width] + xshape)
        # offsets
        self.offset = var_construct(rng, [base_models * models_per_base, width])
        # first 3 dims are batch_size, parallel models, width
        # --> should not be reduced
        self.reduce_dims = [3 + i for i in range(len(xshape))]

    def _get_wop(self):
        """Get operable W tensor; W = training weights
        Have to do this here because tensorflow defaults to eager mode
        """
        # --> base_models x models_per_base x ...
        raw_w = self.shared_comps + self.spread_comps
        # --> num_model x ...
        w = tf.reshape(raw_w, [self.base_models * self.models_per_base, self.width] + self.xshape)
        # add dim for batch
        wop = tf.expand_dims(w, 0)
        return wop

    def eval(self, x: List[tf.Tensor]):
        """Basic evaluation

        Args:
            x (List[tf.Tensor]): input tensor
                batches x ...
        
        Returns:
            tf.tensor: reduced tensor
                batches x parallel models x width
        """
        # add dims for parallel models and width
        x = x[0]
        x = tf.expand_dims(x, 1)
        x = tf.expand_dims(x, 1)
        return (self.offset + tf.math.reduce_sum(x * self._get_wop(),
                                                axis=self.reduce_dims))
    
    def get_width(self):
        return self.width

    def spread_error(self):
        """Calculate the spread error for the layer
        > Spread Error = defined for a given model group (base component + sub_models)
            = L2 norm of each spread component
        > Where w for model set = w_shared (single shared component)
                                 + w_spread (components for each model in the group)
        
        Returns:
            tf.tensor scalar: L1 norm summed across all spread components
                scalar
        """
        return tf.reduce_sum(tf.abs(self.spread_comps))

    def get_trainable_weights(self):
        return [self.shared_comps, self.spread_comps, self.offset]


class LayerFB(LayerIface):
    """FilterBank = single FilterBank shared across all layers within a model group
        For a single model:
            > FB (filter bank) = fb_dim x xdims
            > W_mod = width x fb_dim
            > mult FB and W_mod --> width x xdims
        For a single model group (one base and models_per_base spread)
            > FB = fb_dim x xdims
            > W_core = 1 x width x fb_dim
            > W_spread = models_per_base x width x fb_dim
            > W = FB W_core + FB W_spread
        Model groups are independent"""

    def __init__(self,
                 base_models: int,
                 models_per_base: int,
                 xshape: List[int],
                 width: int,
                 fb_dim: int,
                 rng: npr.Generator):
        """Initialize a basic layer
        > total num_models = base_models * models_per_base
        > special case: base_models = 0 --> there are
            [models_per_base] independent models

        Args:
            num_models (int): number of base models
            models_per_base (int): number of models per
                each base model. These models all share a
                common component
            xshape (List[int]): shape of extra dimensions
                doesn't include parallel models or batch_size
            width (int): number of outputs from the layer
            fb_dim (int): number of filters in filter bank
            rng (npr.Generator): numpy random generator
        """
        assert(base_models >= 0), "positive base models number"
        assert(models_per_base > 0), "models per base 1 or more"
        assert(fb_dim >= 1)
        self.rng = rng
        self.width = width
        self.base_models = base_models
        self.models_per_base = models_per_base
        self.xshape = xshape
        # coefficients
        # dims overall = (base_models, models_per_base, width, fb_dim, ... xdims ...)
        self.train_vars = []
        if base_models > 0:
            self.fb = var_construct(rng, [base_models, 1, 1, fb_dim] + xshape)
            self.w_shared = var_construct(rng, [base_models, 1, width, fb_dim]
                                                + [1 for _ in xshape])
            self.train_vars = [self.fb, self.w_shared]
        else:
            self.w_shared = 0
        # SPREAD component
        self.w_spread = var_construct(rng, [base_models, models_per_base, width, fb_dim]
                                             + [1 for _ in xshape])
        # offsets
        self.offset = var_construct(rng, [base_models * models_per_base, width])
        # first 3 dims are batch_size, parallel models, width
        # --> should not be reduced
        self.reduce_dims = [3 + i for i in range(len(xshape))]
        self.train_vars.extend([self.w_spread, self.offset])

    def _get_wop(self):
        """Get operable W tensor; W = training weights
        Have to do this here because tensorflow defaults to eager mode
        """
        # raw_r = fb w_shared + fb w_spread
        # --> base_models x models_per_base x width x xdims
        raw_w = tf.reduce_sum(self.fb * self.w_shared + self.fb * self.w_spread, axis=3)
        # --> num_model x ...
        w = tf.reshape(raw_w, [self.base_models * self.models_per_base, self.width] + self.xshape)
        # add dim for batch
        wop = tf.expand_dims(w, 0)
        return wop

    def eval(self, x: List[tf.Tensor]):
        """Basic evaluation

        Args:
            x (List[tf.Tensor]): input tensor
                batches x ...
        
        Returns:
            tf.tensor: reduced tensor
                batches x parallel models x width
        """
        # add dims for parallel models and width
        x = x[0]
        x = tf.expand_dims(x, 1)
        x = tf.expand_dims(x, 1)
        return (self.offset + tf.math.reduce_sum(x * self._get_wop(),
                                                axis=self.reduce_dims))
    
    def get_width(self):
        return self.width

    def spread_error(self):
        """Calculate the spread error for the layer
        > Spread Error = defined for a given model group (base component + sub_models)
            = L2 norm of each spread component
        > Where w for model set = w_shared (single shared component)
                                 + w_spread (components for each model in the group)
        
        Returns:
            _type_: L1 norm summed across all spread components
                scalar
        """
        # first: get spread in xdim space (apples to apples comparison with other layer types)
        # --> (base_models, models_per_base, width, ... xdims ...)
        sp_xdim = tf.reduce_sum(self.fb * self.w_spread, axis=3)
        return tf.reduce_sum(tf.abs(sp_xdim))

    def get_trainable_weights(self):
        return self.train_vars


class LayerMulti(LayerIface):
    """A layer composed of multiple sub-layers
        Will ADD together outputs of layers for all methods"""

    def __init__(self, sub_layers: List[LayerIface]):
        """Compose the multi-layer
        > sub_layers must have the same width

        Args:
            sub_layers (List[LayerIface]):
        """
        self.sub_layers = sub_layers
        self.width = sub_layers[0].get_width()
        for sl in sub_layers:
            assert(sl.get_width() == self.width)

    def get_width(self):
        return self.width
    
    def eval(self, x: List[tf.Tensor]):
        """x = List[tf.Tensor]"""
        evs = []
        for xi, sl in zip(x, self.sub_layers):
            evs.append(sl.eval([xi]))
        return tf.add_n(evs)
    
    def spread_error(self):
        serr = [sl.spread_error() for sl in self.sub_layers]
        return tf.add_n(serr)

    def get_trainable_weights(self):
        twz = []
        for sl in self.sub_layers:
            twz.extend(sl.get_trainable_weights())
        return twz


# layer factories


class LayerFactoryIface(abc.ABC):

    def build_layer(self):
        """Construct and return a new layer
        """
        pass

    def get_width(self):
        """Get number of outputs for each layer"""
        pass

    def get_num_models(self):
        pass


class LayerFactoryBasic(LayerFactoryIface):

    def __init__(self,
                 base_models: int,
                 models_per_base: int,
                 xshape: List[int],
                 width: int,
                 rng: npr.Generator):
        """Build the factory (which will build layers)

        Args:
            base_models (int): number of base models / model groups
            models_per_base (int): number of models (fit in parallel)
                per each base model / model group
            xshape (List[int]): shape of input dims
            width (int): number of outputs for each model
            rng (npr.Generator): numpy random generator
        """
        self.rng = rng
        self.base_models = base_models
        self.models_per_base = models_per_base
        self.xshape = xshape
        self.width = width
    
    def build_layer(self):
        return LayerBasic(self.base_models, self.models_per_base, self.xshape, self.width, self.rng)
    
    def get_width(self):
        return self.width
    
    def get_num_models(self):
        return self.base_models * self.models_per_base


class LayerFactoryFB(LayerFactoryIface):

    def __init__(self,
                 base_models: int,
                 models_per_base: int,
                 xshape: List[int],
                 width: int,
                 fb_dim: int,
                 rng: npr.Generator):
        """Build the factory (which will build layers)

        Args:
            base_models (int): number of base models / model groups
            models_per_base (int): number of models (fit in parallel)
                per each base model / model group
            xshape (List[int]): shape of input dims
            width (int): number of outputs for each model
            fb_dim (int): number of filters in each filterbank
                restricts dimensionality of layers
            rng (npr.Generator): numpy random generator
        """
        self.rng = rng
        self.base_models = base_models
        self.models_per_base = models_per_base
        self.xshape = xshape
        self.width = width
        self.fb_dim = fb_dim

    def build_layer(self):
        return LayerFB(self.base_models, self.models_per_base, self.xshape, self.width,
                        self.fb_dim, self.rng)
    
    def get_width(self):
        return self.width
    
    def get_num_models(self):
        return self.base_models * self.models_per_base


class LayerFactoryMulti(LayerFactoryIface):
    """Multi layers are compsed of other layer types
        --> add together their outputs for iface methods"""
    def __init__(self, lfacts: List[LayerIface]):
        self.lfacts = lfacts
        self.width = self.lfacts[0].get_width()
        self.num_models = self.lfacts[0].get_num_models()
        for slf in self.lfacts:
            assert(slf.get_width() == self.width)
            assert(slf.get_num_models() == self.num_models)
        
    def build_layer(self):
        raw_layers = [slf.build_layer() for slf in self.lfacts]
        return LayerMulti(raw_layers)
    
    def get_width(self):
        return self.width
    
    def get_num_models(self):
        return self.num_models
