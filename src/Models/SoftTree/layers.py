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

    def l2(self):
        """L2 calculation
        """
        pass

    def get_width(self):
        """Report the width"""
        pass



class LayerBasic(LayerIface):

    def __init__(self,
                 num_models: int,
                 xshape: List[int],
                 width: int,
                 rng: npr.Generator):
        """Initialize a basic layer

        Args:
            num_models (int): number of parallel models
            xshape (List[int]): shape of extra dimensions
                doesn't include parallel models or batch_size
            width (int): number of outputs from the layer
            rng (npr.Generator): numpy random generator
        """
        self.rng = rng
        self.width = width
        # coefficients
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

    def l2(self):
        """L2 calculation
        currently does an averaging out of lazyness
        """
        return tf.math.reduce_mean(tf.pow(self.wop, 2),
                                   axis=[0] + self.reduce_dims)
    
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
                 rng: npr.Generator):
        """Initialize a low rank, FilterBank (FB) layer

        Args:
            num_models (int): number of parallel models
            xshape (List[int]): shape of extra dimensions
                doesn't include parallel models or batch_size
            width (int): number of outputs from the layer
            low_dim (int): limiting dimension
                filter bank = low_dim x xshape (other dims)
            rng (npr.Generator): numpy random generator
        """
        self.rng = rng
        self.width = width
        # coefficients
        # NOTE: scales up coeffs to make up for squaring
        self.fb = var_construct(rng, [low_dim] + xshape, 2.)
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

    def l2(self):
        """L2 calculation
        currently does an averaging out of lazyness
        """
        # does l2 calc in the outer product space
        return tf.math.reduce_mean(tf.pow(self.wop, 2),
                                   axis=[0] + self.reduce_dims)

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
