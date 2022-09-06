"""  
    Group/Set Sampling
    > distinct from source sampling
    > sets/groups composed of data sources

"""
import numpy as np
import numpy.random as npr
from typing import List
from dataclasses import dataclass
from source_sampling import DataSource


@dataclass
class DataGroup:
    data_src: List[DataSource]
    sub_sets: List  # List of DataGroup(s)


# utils


def gather_leaves(grp_root: DataGroup):
    """get leaves at bottom of tree ~ ignores internal leaves

    Args:
        grp_root (DataGroup): root of group
    """
    if grp_root.sub_sets is None or len(grp_root.sub_sets) == 0:
        return [ds for ds in grp_root.data_src]
    rets = []
    for ss in grp_root.sub_sets:
        rets.extend(gather_leaves(ss))
    return rets


# set sampling functions


def _split(rng: npr.Generator,
           split_prob: float,
           v: List):
    inds = np.arange(len(v))
    N = int(len(inds) * split_prob)
    rng.shuffle(inds)
    return [v[ind] for ind in inds[:N]], [v[ind] for ind in inds[N:]]


def _split_leaves(grp_root: DataGroup,
                  rng: npr.Generator,
                  split_prob: float,
                  prime_node: DataGroup,
                  comp_node: DataGroup):
    # Key: grp_root, prime_node, comp_node assumed to be
    # at same tree level
    if grp_root.sub_sets is None or len(grp_root.sub_sets) == 0:
        ds1, ds2 = _split(rng, split_prob, grp_root.data_src)
        prime_node.data_src = ds1
        comp_node.data_src = ds2
        return
    for ss in grp_root.sub_sets:
        prime_node.sub_sets.append(DataGroup(None, []))
        comp_node.sub_sets.append(DataGroup(None, []))
        _split_leaves(ss, rng, split_prob,
                      prime_node.sub_sets[-1],
                      comp_node.sub_sets[-1])


def split_leaves(grp_root: DataGroup,
                 rng: npr.Generator,
                 split_prob: float = 0.5):
    """Split leaves within each group
    > ignores internal leaves

    Args:
        grp_root (DataGroup): root of group
        rng:
        split_prob (float): splitting probability
            = probability of element ending up
            in primary set (comp = 1 - split_prob)

    Returns:
        DataGroup: root of primary tree
        DataGroup: root of complementary tree
    """
    prime_root = DataGroup(None, [])
    comp_root = DataGroup(None, [])
    _split_leaves(grp_root, rng, split_prob,
                  prime_root, comp_root)
    return prime_root, comp_root
