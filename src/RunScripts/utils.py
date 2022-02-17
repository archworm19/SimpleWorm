"""Load functions"""
from typing import List
import numpy as np


def _check_shape(sh1: List[int], sh2: List[int]):
    """Check if shapes are the same

    Args:
        sh1 (List[int]): shape of an array
        sh2 (List[int]): ...

    Returns:
        bool: True if same
    """
    if len(sh1) != len(sh2):
        return False
    v = True
    for i in range(len(sh1)):
        if sh1[i] != sh2[i]:
            v = False
    return v


def _array_dist(ar1: np.ndarray, ar2: np.ndarray):
    """Check if two arrays are approximately equal
    Is an indication of poor data handling
    Asserts they are no approximately equal

    Args:
        ar1 (np.ndarray): numpy array
        ar2 (np.ndarray): numpy array

    Returns:
        float if shape matches;
        otherwise --> nan
    """
    # check shape
    if _check_shape(np.shape(ar1), np.shape(ar2)):
        return np.mean((ar1 - ar2)**2.)
    else:
        return np.nan


def get_all_array_dists(arz: List[np.ndarray]):
    """Get all pair-wise array distances

    Args:
        arz (List[np.ndarray]): list of numpy arrays
    
    Returns:
        np.ndarray: matrix of pair-wise distances
            if dist calc doesn't make sense --> nan
            diagonals will be nans
    """
    arlen = len(arz)
    m = np.full([arlen, arlen], np.nan, np.float32)
    for i in range(arlen - 1):
        for j in range(i + 1, arlen):
            m[i,j] = _array_dist(arz[i], arz[j])
            m[j,i] = m[i,j]
    return m
