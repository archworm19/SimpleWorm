"""
    Layers

    Factories that generate layers
"""
from typing import List
import abc
import numpy as np
import numpy.random as npr
import tensorflow as tf


DRNG = npr.default_rng(42)


# utilities


def var_construct(v_shape: List[int], v_scale: int = 1.):
    """Construct a uniform random tensorflow variable
    of specified shape

    Args:
        v_shape (List[int]): full shape of variable
    """
    v_np = v_scale * (DRNG.random(v_shape) - 0.5)
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



class LayerBasic(LayerIface):

    def __init__(self,
                 num_models: int,
                 xshape: List[int],
                 width: int):
        """Initialize a basic layer

        Args:
            num_models (int): number of parallel models
            xshape (List[int]): shape of extra dimensions
                doesn't include parallel models or batch_size
            width (int): number of outputs from the layer
        """
        self.w = var_construct([num_models, width] + xshape)
        self.wop = tf.expand_dims(self.w, 0)
        # first 3 dims are batch_size, parallel models, width
        # --> should not be reduced
        self.reduce_dims = [3 + i for i in range(len(xshape))]
        self.x_reshape = [-1, 1, 1] + xshape

    def eval(self, x):
        """Basic evaluation

        Args:
            x (TODO): input tensor
                batches x ...
        
        Returns:
            TODO/TYPE: reduced tensor
                batches x parallel models x width
        """
        x2 = tf.reshape(x, self.x_reshape)
        return tf.math.reduce_sum(x2 * self.wop,
                                  axis=self.reduce_dims)

    def l2(self):
        """L2 calculation
        currently does an averaging out of lazyness
        """
        return tf.math.reduce_mean(tf.pow(self.wop, 2),
                                   axis=[0] + self.reduce_dims)


class LayerLowRankTseries(LayerIface):

    def __init__(self,
                 num_models: int,
                 ch_dim: int,
                 t_dim: int,
                 lowrank: int,
                 width: int):
        """Initialize a low rank, time-series layer
        Specifically designed with timeseries in mind
        > w shape = width x ch_dim x t_dim
        and reduce along ch_dim, t_dim
        > low-rank input transformation:
        > > w_ch = width x ch_dim x q x t_dim
        > > w_t = 1 x 1 x q x t_dim *
        > > broadcast + reduce q --> w_shape
        > Variable comparison
        > Ex: width=3; dim_ch = 5; dim-t = 10; q=2
        # > > rull-rank = 150
        # > > low-rank = 30 + 20 = 50
        # > > low-rank + q=1: 15 + 10 = 25

        * forces sharing of w_t components
        across widths

        Args:
            num_models (int): number of parallel models
            dim_ch (int): number of input dimensions along channel
                axis = number of channels
            dim_t (int): number of input dimensions along timeseries
                axies = timeseries length
            lowrank (int): the rank constraint
            width (int): number of outputs from the layer
        """
        self.w_ch = var_construct([num_models, width, ch_dim, lowrank, 1], 2.)
        self.w_t = var_construct([num_models, 1, 1, lowrank, t_dim], 2.)
        # --> num_models x width x ch_dim x lowrank x t_dim
        wbig = self.w_ch * self.w_t
        # reduce along lowrank
        # --> num_models x width x ch_dim x t_dim
        self.w = tf.reduce_sum(wbig, axis=[3])
        self.wop = tf.expand_dims(self.w, 0)
        # protected dims: batch, num_model x width
        self.reduce_dims = [3,4]
        self.x_reshape = [-1, 1, 1] + [ch_dim, t_dim]
    
    def eval(self, x):
        """Basic evaluation

        Args:
            x (TODO): input tensor
                batches x parallel models x 1 x ...
        
        Returns:
            TODO/TYPE: reduced tensor
                batches x parallel models x width
        """
        x2 = tf.reshape(x, self.x_reshape)
        return tf.math.reduce_sum(x2 * self.wop,
                                  axis=self.reduce_dims)

    def l2(self):
        """L2 calculation
        currently does an averaging out of lazyness
        """
        # does l2 calc in the outer product space
        return tf.math.reduce_mean(tf.pow(self.wop, 2),
                                   axis=[0] + self.reduce_dims)

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
                 width: int):
        """Build the factory (which will build layers)

        Args:
            num_models (int): number of parallel models
            xshape (List[int]): shape of input dims
            width (int): number of outputs for each model
        """
        self.num_models = num_models 
        self.xshape = xshape
        self.width = width
    
    def build_layer(self):
        return LayerBasic(self.num_models, self.xshape, self.width)
    
    def get_width(self):
        return self.width
    
    def get_num_models(self):
        return self.num_models


class LayerFactoryLowRankTseries(LayerFactoryIface):

    def __init__(self,
                 num_models: int,
                 dim_ch: int,
                 dim_t: int,
                 low_rank: int,
                 width: int):
        """Build Factory for building low-rank
        layers. Low-rank layers are only compatible
        with square, matrix inputs ~ primarily
        designed for timeseries data

        Args:
            num_models (int): number of parallel models
            dim_ch (int): number of input dimensions along channel
                axis = number of channels
            dim_t (int): number of input dimensions along timeseries
                axies = timeseries length
            lowrank (int): the rank constraint
            width (int): number of outputs for each model
        """
        self.num_models = num_models
        self.dim_ch = dim_ch
        self.dim_t = dim_t
        self.low_rank = low_rank
        self.width = width

    def get_num_models(self):
        return self.num_models

    def get_width(self):
        return self.width
    
    def build_layer(self):
        return LayerLowRankTseries(self.num_models,
                                   self.dim_ch,
                                   self.dim_t,
                                   self.low_rank,
                                   self.width)
