"""Soft Forest

    Fit a number of soft decision trees simultaneously

    NOTE: SoftForest does NOT implement model interface
    > It is a component of a model because
    it still needs a decoding layer
"""
from typing import List
import tensorflow as tf
from Models.SoftTree.layers import LayerFactoryIface, LayerIface


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

        Returns:
            TODO:tftype: branch evaluation unnormalized
                batch_size x num_model x width
        """
        v = []
        for i, layer in enumerate(self.layers):
            v.append(layer.eval(x_list[i]))
        return tf.math.add_n(v)


# TODO: SoftForest should do 0 building ~ just a container for data
# 
class SoftForest:

    def __init__(self,
                 num_model: int,
                 width: int,
                 root_node: ForestNode):
        """Initialize the Soft Forest
        total number of layers = (width+1)^depth
        > each branch
        > > val for each child where num_child = width
        > > stack with a constant 0 to give width+1 values
        > > normalize across these for probabilities

        Args:
            root_node (ForestNode)
        """
        self.root_node = root_node
        self.offset = tf.zeros([1, num_model, width])
    
    def eval(self, x_list: List):
        """evaluate the whole tree

        Args:
            x_list (List): list of inputs
                each input will match up with
                a different layer factory
        """
        # initial probs:
        # all prz = batch_size x num_model x width + 1
        prz = self._eval_branch(self.root_node, x_list)
        children = self.root_node.get_children()
        while len(children) > 0:
            # stack children along axis2
            ch_prz, new_childz = [], []
            for i, ch in enumerate(children):
                # mult with parent
                # --> batch_size x num_model x width+1
                # TODO: too slow?
                chpr = prz[:,:,i:i+1] * self._eval_branch(ch, x_list)
                ch_prz.append(chpr)
                new_childz.extend(ch.get_children())
            # --> batch_size x num_model x total_children
            prz = tf.concat(ch_prz, axis=2)
            # get children:
            children = new_childz
        return prz

    def _eval_branch(self,
                     node: ForestNode,
                     x_list: List):
        """Evaluate current branch

        Args:
            node (ForestNode):
            x_list (List): input data

        Returns:
            [type]: probabilities within current
                branch ~ doesn't know anything about
                parent
                batch_size x num_model x width + 1
        """
        # batch_size x num_model x width
        rawv = node.raw_eval(x_list)
        # stack a constant:
        # --> batch_size x num_model x width + 1
        rawfull = tf.concat([rawv, self.offset], axis=2)
        # softmax across children
        probs = tf.nn.softmax(rawfull, axis=2)
        return probs


def _fcheckz(layer_factories: List[LayerFactoryIface]):
    """All layer factories must have
    same width and num_models"""
    width0 = layer_factories[0].get_width()
    nm0 = layer_factories[0].get_num_models()
    for lf in layer_factories[1:]:
        assert(width0 == lf.get_width()), "width mismatch"
        assert(nm0 == lf.get_num_models()), "num model mismatch"


def build_forest(depth: int,
                 layer_factories: List[LayerFactoryIface]):
    """Build a forest

    Args:
        depth (int): number of layers
            depth = 1 means just the root node
        layer_factories (List[LayerFactoryIface]): set of layer factories
            order should match the order of inputs submitted to eval 
    
    Returns:
        SoftForsest object
    """
    _fcheckz(layer_factories)

    # start with root node:
    num_models = layer_factories[0].get_num_models()
    width = layer_factories[0].get_width()
    layers = [lf.build_layer() for lf in layer_factories]
    root_node = ForestNode(width, layers)

    # build the rest:
    clevl = [root_node]
    for _ in range(1, depth):
        next_gen = []
        for parent in clevl:
            children = []
            for j in range(width):
                layers = [lf.build_layer() for lf in layer_factories]
                children.append(ForestNode(width, layers))
            # assign children to parent:
            parent.add_children(children)
            next_gen.extend(children)
        clevl = next_gen
    return SoftForest(num_models, width, root_node)
        


    


