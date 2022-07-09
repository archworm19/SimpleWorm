"""Soft Forest

    Fit a number of soft decision trees simultaneously

    NOTE: SoftForest does NOT implement model interface
    > It is a component of a model because
    it still needs a decoding layer
"""
from typing import List, Union
import tensorflow as tf
from Models.SoftTree.layers import LayerFactoryIface, LayerIface
from Models.SoftTree.model_interfaces import GateSubModel


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
        """Get children of current node
        Will return empty list if there are no children

        Returns:
            List[ForestNode]
        """
        return self.children
    
    def raw_eval(self,
                 x: Union[tf.Tensor, List[tf.Tensor]]):
        """Raw evaluation
        Raw = no processing (softmax, etc)

        Args:
            x (Union[tf.Tensor, List[tf.Tensor]]): input
                has to match layer shape

        Returns:
            tf.Tensor: branch evaluation unnormalized
                batch_size x num_model x width
        """
        return self.layer.eval(x)


def _get_all_nodes(root_node: ForestNode, nodes: List[ForestNode]):
    """Get all nodes in the forest

    Args:
        root_node (ForestNode): root node
        nodes (List[ForestNode]): current set of nodes
            Adds to this list
    """
    nodes.append(root_node)
    childz = root_node.get_children()
    if len(childz) == 0:
        return
    for ch in childz:
        _get_all_nodes(ch, nodes)
    

class SoftForest(GateSubModel):

    def __init__(self,
                 root_node: ForestNode,
                 forest_penalty: float,
                 spread_penalty: float):
        """Initialize the Soft Forest
        total number of layers = (width+1)^depth
        > each branch
        > > val for each child where num_child = width
        > > stack with a constant 0 to give width+1 values
        > > normalize across these for probabilities

        Args:
            root_node (ForestNode)
            forest_penalty (float): regularization strength
                on soft forest negentropy
            spread_penalty (float): regularization strength
                on spread loss in layers
        """
        self.root_node = root_node
        self.forest_penalty = forest_penalty
        self.spread_penalty = spread_penalty
        self.all_nodes = []
        _get_all_nodes(root_node, self.all_nodes)
    
    def eval(self, x: Union[tf.Tensor, List[tf.Tensor]]):
        """evaluate the whole tree

        Args:
            x (Union[tf.Tensor, List[tf.Tensor]]): input
                order/shape of x must match the supplied
                LayerFactory

        Returns:
            tf.Tensor: leaf probabilities
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
                     x: Union[tf.Tensor, List[tf.Tensor]]):
        """Evaluate current branch

        Args:
            node (ForestNode):
            x (Union[tf.Tensor, List[tf.Tensor]]): input data

        Returns:
            tf.Tensor: probabilities within current
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

    def get_trainable_weights(self):
        """get all trainable weights within forest
        
        Returns: 
            List[tf.Tensor]
        """
        wz = []
        for n in self.all_nodes:
            wz.extend(n.layer.get_trainable_weights())
        return wz

    def _forest_loss(self, x: Union[tf.Tensor, List[tf.Tensor]],
                           data_weights: tf.Tensor,
                           reg_epoch_scale: float):
        """Forest entropy loss
        = negentropy of output probabilities
        -->

        Args:
            x (Union[tf.Tensor, List[tf.Tensor]]): input
            data_weights (tf.Tensor): weights on the data points
                batch_size x num_model
            reg_epoch_scale (float): how much to scale regularization
                by as a function of epoch == f(temperature)

        Returns:
            scalar: sum loss across batch/models
        """
        # forest evaluation:
        # --> batch_size x num_models x num_leaves
        forest_eval = self.eval(x)

        # negative entropy calc across leaves:
        # --> batch_size x num_models
        neg_entropy = tf.reduce_sum(forest_eval * tf.math.log(forest_eval), axis=2)

        # scale by data_weights and temperature:
        return (tf.stop_gradient(reg_epoch_scale) *
                self.forest_penalty *
                tf.reduce_sum(data_weights * neg_entropy))

    def _spread_loss(self):
        """Spread penalty for model layers

        Returns:
            tf.tensor scalar: spread error summed across all layers

        """
        sp_errs = [n.layer.spread_error() for n in self.all_nodes]
        return tf.add_n(sp_errs) * self.spread_penalty

    def regularization_loss(self,
                            x: Union[tf.Tensor, List[tf.Tensor]],
                            data_weights: tf.Tensor,
                            reg_epoch_scale: float):
        """Regularization loss
        
        Args:
            x (Union[tf.Tensor, List[tf.Tensor]]): input
            data_weights (tf.Tensor): weights on the data points
                batch_size x num_model
            reg_epoch_scale (float): how much to scale regularization
                by as a function of epoch == f(temperature)

        Returns:
            scalar: sum loss across batch/models
        """
        return self._forest_loss(x, data_weights, reg_epoch_scale) + self._spread_loss()


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
        List[LayerIface]: reference to all of the created layers
    """
    assert(depth >= 1), "should at least have root node"
    # NOTE: forest width = layer width + 1 (includes offset)
    width = layer_factory.get_width() + 1
    init_layer = layer_factory.build_layer()
    root_node = ForestNode(init_layer)

    # layer references
    layer_refs = [init_layer]

    # build the rest:
    clevl = [root_node]
    for _ in range(1, depth):
        next_gen = []
        for parent in clevl:
            children = []
            for _ in range(width):
                c_layer = layer_factory.build_layer()
                children.append(ForestNode(c_layer))
                layer_refs.append(c_layer)
            # assign children to parent:
            parent.add_children(children)
            next_gen.extend(children)
        clevl = next_gen
    return SoftForest(root_node), width, layer_refs
