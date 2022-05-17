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
                 layer: LayerIface):
        """Build a Forest Node = branch
        Each node can contain N layers that
        operate on N different input dataz
        """
        self.layer = layer
        self.children = []
    
    def add_children(self, children: List):
        """Add ForestNode children

        Args:
            children (List[ForestNode])
        """
        lwidth = self.layer.get_width()
        assert(lwidth + 1 == len(children)), "width ({0}) - children mismatch ({1})".format(lwidth + 1,
                                                                                            len(children))
        self.children = children
    
    def get_children(self):
        """Get children
        Will return empty list if there are no children
        """
        return self.children
    
    def raw_eval(self,
                 x):
        """Raw evaluation
        Raw = no processing (softmax, etc)

        Args:
            x (TODO:tftypes): input

        Returns:
            TODO:tftype: branch evaluation unnormalized
                batch_size x num_model x width
        """
        return self.layer.eval(x)


class SoftForest:

    def __init__(self,
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
    
    def eval(self, x):
        """evaluate the whole tree

        Args:
            x (tf.tensor of List[tf.tensor]): input
                order/shape of x must match the supplied
                LayerFactory

        Returns:
            TODO: tf type: leaf probabilities
                batch_size x num_models x num_leaves
                normalized across leaves (within batch ind / model)
        """
        # initial probs:
        # all prz = batch_size x num_model x width + 1
        prz = self._eval_branch(self.root_node, x)
        children = self.root_node.get_children()
        while len(children) > 0:
            # stack children along axis2
            ch_prz, new_childz = [], []
            for i, ch in enumerate(children):
                # mult with parent
                # --> batch_size x num_model x width+1
                # TODO: too slow?
                chpr = prz[:,:,i:i+1] * self._eval_branch(ch, x)
                ch_prz.append(chpr)
                new_childz.extend(ch.get_children())
            # --> batch_size x num_model x total_children
            prz = tf.concat(ch_prz, axis=2)
            # get children:
            children = new_childz
        return prz

    def _eval_branch(self,
                     node: ForestNode,
                     x):
        """Evaluate current branch

        Args:
            node (ForestNode):
            x: input data

        Returns:
            [type]: probabilities within current
                branch ~ doesn't know anything about
                parent
                batch_size x num_model x width + 1
        """
        # batch_size x num_model x width
        rawv = node.raw_eval(x)
        # stack a constant (source of width+1):
        # --> batch_size x num_model x width + 1
        sl_offset = tf.stop_gradient(rawv[:,:,:1] * 0)
        rawfull = tf.concat([rawv, sl_offset], axis=2)
        # softmax across children
        probs = tf.nn.softmax(rawfull, axis=2)
        return probs


def build_forest(depth: int,
                 layer_factory: LayerFactoryIface):
    """Build a forest

    Args:
        depth (int): number of layers
            depth = 1 means just the root node
        layer_factory (LayerFactoryIface) layer factory
            to be used for each layer
    
    Returns:
        SoftForest object
        int: width of softforest
    """
    # NOTE: forest width = layer width + 1 (includes offset)
    width = layer_factory.get_width() + 1
    root_node = ForestNode(layer_factory.build_layer())

    # build the rest:
    clevl = [root_node]
    for _ in range(1, depth):
        next_gen = []
        for parent in clevl:
            children = []
            for _ in range(width):
                children.append(ForestNode(layer_factory.build_layer()))
            # assign children to parent:
            parent.add_children(children)
            next_gen.extend(children)
        clevl = next_gen
    return SoftForest(root_node), width
