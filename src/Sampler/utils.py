"""Utilites used across samplers"""
from typing import List
import numpy as np
import numpy.random as npr


def shuffle_sample(num_anml: int,
                    gen,
                    train_prob: float = .667):
    """Shuffle-based Sampling

    Args:
        num_anml (int): number of animals in a 
            set
        gen ([type]): random number generator
    
    Returns:
        np.ndarray: boolean array of length
            num_anml
            True -> training set
    """
    # shuffle-based sampling
    inds = np.arange(num_anml)
    gen.shuffle(inds)
    numtr = int(train_prob * num_anml)
    train_anmls = np.full((num_anml,), False)
    train_anmls[inds[:numtr]] = True
    return train_anmls


def generate_anml_split(useable_dat: np.ndarray,
                        gen,
                        tr_prob: float = .5):
    """Generate split across animals
    using provided random seed
    Operates on a single set

    Args:
        useable_dat (np.ndarray): boolean array
            where len = number of animals in set
            true --> caller can access animal
        gen: random generator
        tr_prob (float): training probability
            probability of given sample ending up
            in training set
    
    Returns:
        np.ndarray: indices of training animals
        np.ndarray: indices of testing animals
    """
    # sample animals for training
    # True --> train; False --> Cross
    use_inds = np.where(useable_dat)[0]
    train_boolz = shuffle_sample(len(use_inds), gen, tr_prob)
    train_anmls = use_inds[train_boolz]
    test_anmls = use_inds[np.logical_not(train_boolz)]
    return train_anmls, test_anmls


def get_data_offsets(data: List[np.ndarray]):
    """get data offsets
    Used to keep track of dataset splits post stacking

    Args:
        data (List[np.ndarray]): data to be stacked

    Returns:
        np.ndarray: integer array where
            len = len(data)
            v[i] is where ith dataset begins
            in the stacked array 
    """
    dlens = np.array([len(dati) for dati in data])
    offsets = np.hstack(([0], np.cumsum(dlens)[:-1]))
    return offsets.astype(np.int32)
