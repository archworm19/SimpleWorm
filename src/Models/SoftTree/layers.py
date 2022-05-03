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

    # TODO: tensorflow types??
    def eval(self, x):
        """Evaluate layer

        Args:
            x (TODO): input tensor
                batches x parallel models x ...
        
        Returns:
            TODO:TYPE?: reduced tensor
                batches x num_parallel_models x width
        """
        pass

    def get_width(self):
        """Report the width"""
        pass



# TODO: new LayerBasic
# should replace OG LayerBasic 
class LayerBasic2(LayerIface):
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
            shared_component (bool): if set to True -->
                layer_i = core_layer + layer_offset_i
        """
        assert(base_models >= 0), "positive base models number"
        assert(models_per_base > 0), "models per base 1 or more"
        self.rng = rng
        self.width = width
        # coefficients
        if base_models > 0:
            self.shared_comps = var_construct(rng, [base_models, 1, width] + xshape)
        else:
            self.shared_comps = 0
        # SPREAD component = base_models x models_per_base x width
        self.spread_comps = var_construct(rng, [base_models, models_per_base, width] + xshape)
        # --> base_models x models_per_base x ...
        raw_w = self.shared_comps + self.spread_comps
        # --> num_model x ...
        self.w = tf.reshape(raw_w, [base_models * models_per_base, width] + xshape)
        # add dim for batch
        self.wop = tf.expand_dims(self.w, 0)
        # offsets
        self.offset = var_construct(rng, [base_models * models_per_base, width])
        # first 3 dims are batch_size, parallel models, width
        # --> should not be reduced
        self.reduce_dims = [3 + i for i in range(len(xshape))]

    def eval(self, x):
        """Basic evaluation

        Args:
            x (TODO): input tensor
                batches x ...
        
        Returns:
            TODO/TYPE: reduced tensor
                batches x parallel models x width
        """
        # add dims for parallel models and width
        x = tf.expand_dims(x, 1)
        x = tf.expand_dims(x, 1)
        return self.offset + tf.math.reduce_sum(x * self.wop,
                                                axis=self.reduce_dims)
    
    def get_width(self):
        return self.width

    # TODO: there should probably be a separate interface for this
    # TODO: maybe a better name too
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
        return tf.reduce_sum(tf.abs(self.spread_comps))


class LayerBasic(LayerIface):

    def __init__(self,
                 num_models: int,
                 xshape: List[int],
                 width: int,
                 rng: npr.Generator,
                 shared_component: bool = False):
        """Initialize a basic layer

        Args:
            num_models (int): number of parallel models
            xshape (List[int]): shape of extra dimensions
                doesn't include parallel models or batch_size
            width (int): number of outputs from the layer
            rng (npr.Generator): numpy random generator
            shared_component (bool): if set to True -->
                layer_i = core_layer + layer_offset_i
        """
        self.rng = rng
        self.width = width
        # coefficients
        if shared_component:
            core_layer = var_construct(rng, [1, width] + xshape)
            off_layer = var_construct(rng, [num_models, width] + xshape)
            self.w = core_layer + off_layer
        else:
            self.w = var_construct(rng, [num_models, width] + xshape)
        self.wop = tf.expand_dims(self.w, 0)
        # offsets
        self.offset = var_construct(rng, [num_models, width])
        # first 3 dims are batch_size, parallel models, width
        # --> should not be reduced
        self.reduce_dims = [3 + i for i in range(len(xshape))]

    def eval(self, x):
        """Basic evaluation

        Args:
            x (TODO): input tensor
                batches x ...
        
        Returns:
            TODO/TYPE: reduced tensor
                batches x parallel models x width
        """
        # add dims for parallel models and width
        x = tf.expand_dims(x, 1)
        x = tf.expand_dims(x, 1)
        return self.offset + tf.math.reduce_sum(x * self.wop,
                                                axis=self.reduce_dims)
    
    def get_width(self):
        return self.width


class LayerLowRankFB(LayerIface):
    # ... designed with flat models in mind
    # FilterBank: within layer, share a bank of filters across models/widths
    # FB (filter bank) = low_dim x dims
    # W = num_model x width x low_dim x dims

    def __init__(self,
                 num_models: int,
                 xshape: List[int],
                 width: int,
                 low_dim: int,
                 rng: npr.Generator,
                 shared_component: bool = False):
        """Initialize a low rank, FilterBank (FB) layer

        Args:
            num_models (int): number of parallel models
            xshape (List[int]): shape of extra dimensions
                doesn't include parallel models or batch_size
            width (int): number of outputs from the layer
            low_dim (int): limiting dimension
                filter bank = low_dim x xshape (other dims)
            rng (npr.Generator): numpy random generator
            shared_component (bool): if set to True -->
                layer_i = core_layer + layer_offset_i
        """
        self.rng = rng
        self.width = width
        # coefficients
        # NOTE: scales up coeffs to make up for squaring
        # filterbank shared either way
        self.fb = var_construct(rng, [low_dim] + xshape, 2.)
        if shared_component:
            wb_core = var_construct(rng, [1, width, low_dim] + xshape, 2.)
            wb_off = var_construct(rng, [num_models, width, low_dim] + xshape, 2.)
            self.wb = wb_core + wb_off
        else:
            self.wb = var_construct(rng, [num_models, width, low_dim] + xshape, 2.)
        # --> num_models x width x low_dim x xshape
        wbig = (tf.reshape(self.fb, [1, 1, low_dim] + xshape) *
                    tf.reshape(self.wb, [num_models, width, low_dim] + xshape))
    
        # offsets
        self.offset = var_construct(rng, [num_models, width])
        # reduce along lowrank
        # --> num_models x width x xshape
        self.w = tf.reduce_sum(wbig, axis=[2])
        self.wop = tf.expand_dims(self.w, 0)
        # protected dims: batch, num_model x width
        self.reduce_dims = [3 + i for i in range(len(xshape))]
    
    def eval(self, x):
        """Basic evaluation

        Args:
            x (TODO): input tensor
                batches x parallel models x 1 x ...
        
        Returns:
            TODO/TYPE: reduced tensor
                batches x parallel models x width
        """
        x = tf.expand_dims(x, 1)
        x = tf.expand_dims(x, 1)
        # add dims for parallel models and width
        return self.offset + tf.math.reduce_sum(x * self.wop,
                                                axis=self.reduce_dims)

    def get_width(self):
        return self.width

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
                 num_models: int,
                 xshape: List[int],
                 width: int,
                 rng: npr.Generator):
        """Build the factory (which will build layers)

        Args:
            num_models (int): number of parallel models
            xshape (List[int]): shape of input dims
            width (int): number of outputs for each model
            rng (npr.Generator): numpy random generator
        """
        self.rng = rng
        self.num_models = num_models 
        self.xshape = xshape
        self.width = width
    
    def build_layer(self):
        return LayerBasic(self.num_models, self.xshape, self.width, self.rng)
    
    def get_width(self):
        return self.width
    
    def get_num_models(self):
        return self.num_models


class LayerFactoryLowRankFB(LayerFactoryIface):

    def __init__(self,
                 num_models: int,
                 xshape: List[int],
                 width: int,
                 low_dim: int,
                 rng: npr.Generator):
        """Build Factory for building low-rank
        layers. Low-rank layers are only compatible
        with square, matrix inputs ~ primarily
        designed for timeseries data

        Args:
            num_models (int): number of parallel models
            xshape (List[int]): shape of input dims
            width (int): number of outputs for each model
            low_dim (int): limiting dimension
                filter bank = low_dim x xshape (other dims)
            rng (npr.Generator): numpy random generator
        """
        self.rng = rng
        self.num_models = num_models
        self.xshape = xshape
        self.width = width
        self.low_dim = low_dim

    def get_num_models(self):
        return self.num_models

    def get_width(self):
        return self.width
    
    def build_layer(self):
        return LayerLowRankFB(self.num_models, self.xshape, self.width,
                                self.low_dim, self.rng)
