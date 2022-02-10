"""Soft Forest

    Fit a number of soft decision trees simultaneously

    NOTE: SoftForest does NOT implement model interface
    > It is a component of a model because
    it still needs a decoding layer
"""
import enum
from typing import List
import tensorflow as tf
from Models.SoftTree.layers import LayerFactoryIface, LayerIface


# TODO: SoftForest should do 0 building ~ just a container for data
# 
class SoftForest:

    def __init__(self,
                 layer_factories: List[LayerFactoryIface],
                 depth: int):
        """Initialize the Soft Forest
        total number of layers = (width+1)^depth
        > each branch
        > > val for each child where num_child = width
        > > stack with a constant 0 to give width+1 values
        > > normalize across these for probabilities

        Args:
            layer_factories List[LayerFactoryIface]: set of layer
                factories that create the layers at each branch
            depth (int): depth of each tree
        """
        self.depth = depth
        # make sure all layer factories have same width
        self.width = layer_factories[0].get_width()
        for lf in layer_factories[1:]:
            assert(self.width == lf.get_width()), "layer factory widths must match"
        # TODO: build structs

    # TODO: build forest

    def _build_branch(self,
                     cdepth: int):
        """Build branch
        --> return all children

        Args:
            cdepth (int): depth of this layer
        """
        # TODO: build layer components
        # returns list of children


class ForestNode:

    def __init__(self,
                 width: int,
                 layers: List[LayerIface]):
        """Build a Forest Node = branch
        Each node can contain N layers that
        operate on N different input dataz
        """
        self.width = width
        self.layers = layers
        self.children = []
    
    def add_children(self, children: List):
        """Add ForestNode children

        Args:
            children (List[ForestNode])
        """
        assert(self.width + 1 == len(children)), "width - children mismatch"
        self.children = children
    
    def get_children(self):
        """Get children
        Will return empty list if there are no children
        """
        return self.children
    
    def raw_eval(self,
                 x_list: List):
        """Raw evaluation
        Raw = no processing (softmax, etc)
        Just add up all of the layers applied
        to x

        Args:
            x_list (List[TODO:tftypes]): input
        """
        v = []
        for i, layer in enumerate(self.layers):
            v.append(layer.eval(x_list[i]))
        return tf.math.add_n(v)
