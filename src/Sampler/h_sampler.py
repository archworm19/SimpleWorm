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

    def __init__(self, leaf: np.ndarray = None):
        self.leaf = leaf
        # TODO: sampling system?





class HTree:

    def __init__(self):

class SmallHSampler(sampler_iface.Sampler):
    """Small Hierarchical sampler
    Small size --> store the whole dataset in memory"""

    # TODO: generic way to represent group hierarchy?
    # 
    def __init__(self, data: np.ndarray, data_groups: np.ndarray):
        """Initialize HSampler
        K = number of groups

        Args:
            data (np.ndarray): N x M array of data
            data_groups (np.ndarray): N x K array of group labels
        """


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
    # seeding random number generator makes 
    # process deterministic
    rng_gen = npr.default_rng(rand_seed)
    full_dat = []
    # iter thru files ~ top-level
    kz = list(npz_dat.keys())
    for i, k in enumerate(kz):
        # random start of first window:
