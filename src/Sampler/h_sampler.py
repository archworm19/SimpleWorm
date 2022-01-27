"""
    Hierarchical Samplers
"""
from typing import Dict
import numpy as np
import numpy.random as npr
import sampler_iface


class HTreeNode:
    """HTreeNode
    Either points to other HTreeNodes or to ndarray
    """

    def __init__(self):
        self.children = {}
        self.leaf = []
    
    def get_child(self, group_name: str):
        return self.children.get(group_name, None)
    
    def insert_child(self, group_name: str, node):
        self.children[group_name] = node
    
    def update_leaf(self, leaf: np.ndarray):
        """Assumes flat array

        Args:
            leaf (np.ndarray): [description]
        """
        self.leaf = np.hstack((self.leaf, leaf))


class HTree:

    def __init__(self):
        self.root = HTreeNode()

    def add_group(self,
                  group_names: List,
                  leaf: np.ndarray):
        cnode = self.root
        for g in group_names:
            new_node = cnode.get_child(g)
            if new_node is None:
                new_node = HTreeNode()
                cnode.insert_child(new_node)
            cnode = new_node
        cnode.update_leaf(leaf)
    

# TODO: why is this design bad?
# > Train / Test are NOT static
# ... that's the point of sampling
# Better design?
# > all possible window starts are the leaves
# How do you keep track of who is next?
# > 


class SmallHSampler(sampler_iface.Sampler):
    """Small Hierarchical sampler
    Small size --> store the whole dataset in memory"""

    # TODO: generic way to represent group hierarchy?
    # 
    def __init__(self, data: np.ndarray, sample_tree: HTree):
        """Initialize HSampler

        Args:
            data (np.ndarray): N x M array of data
        """
        self.data = data
        self.sample_tree = sample_tree

    def set_new_epoch(self):
        

    def pull_train_samples(self, num_samples: int):



def build_from_wormwindows(npz_dat: Dict,
                           twindow_size: int,
                           train_percentage: float,
                           rand_seed: int = 42):
    """build_from_wormwindows
    npz_dat assumption:
    > each Dict value is a numpy array
    AND this is the top level of hierarchy
    > Each Dict is a T x z array where T
    is the number of timepoints

    Size assumption: small ~ no need for memory mapping
    
    Procedure:
    > Break up into train - test sets according
    to file > timewindow hierarchy

    Args:
        npz_dat (Dict): Dict where values are
            numpy arrays
        
    """
    htree = HTree()
    # seeding random number generator makes 
    # process deterministic
    rng_gen = npr.default_rng(rand_seed)
    full_dat = []
    # iter thru files ~ top-level
    kz = list(npz_dat.keys())
    offset = 0
    for k in kz:
        # random start of first window:
        rt0 = rng_gen.integers(0, twindow_size, 1)[0]
        # update groups:
        dat_i = npz_dat[k]
        train_inds, test_inds = [], []
        for j in range(rt0, len(dat_i), twindow_size):
            if rng_gen.random() < train_percentage:
                train_inds.extend([offset + z for z in range(j, j+twindow_size)])
            else:
                test_inds.extend([offset + z for z in range(j, j+twindow_size)])
        # insert groups into the tree
        htree.add_group([k, 'train'], train_inds)
        htree.add_group([k, 'test'], test_inds)

        # save data:
        full_dat.append(npz_dat[k])
        # update offset for set completion:
        offset += np.shape(npz_dat[k])[0]
