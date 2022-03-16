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


def _sub_cells_helper(targ_inp: np.ndarray,
                      raw_input_dim: int,
                      targ_cell_dim: int,
                      arz: List[np.ndarray],
                      miss_tolerance: float):
    """Search target input against raw inputs
    of all arrays:
    > if raw input of current array is approximately the same
    as target raw input AND current array has no NaNs in 
    targ_cell_dim --> use this arrays target cells

    Args:
        targ_inp (np.ndarray): raw input against which
            raw input of arz[i] will be searched
        raw_input_dim (int): dimension containing raw input
            assumed to be same across arz
        targ_cell_dim (int): dimension of cell of interest
            assumed to be same across arz
        arz (List[np.ndarray]): raw data
        miss_tolerance (float): if dist(targ_inp and inp from arz[i])
            is less than miss_tolerance --> arz[i] will be used

    Returns:
        np.ndarray: find all useable arz --> take their cells
            --> average --> return
    """
    sub_cells = []
    for ari in arz:
        if np.shape(ari)[0] != len(targ_inp):
            continue
        if np.sum(np.isnan(ari[:,targ_cell_dim])) > 0:
            continue
        di = np.sqrt(np.sum((targ_inp - ari[:,raw_input_dim])**2.))
        if di < miss_tolerance:
            sub_cells.append(ari[:,targ_cell_dim])
    sub_cells = np.array(sub_cells)
    return np.mean(sub_cells, axis=0)


def sub_cells(arz: List[np.ndarray],
              raw_input_dim: int,
              miss_tolerance: int):
    """Substitute missing cells (contains nans) with matching cells
    from a different animal
    > Operates on a single set of animals
    > Checks to ensure raw input aligns

    Args:
        arz (List[np.ndarray]): each numpy array is a different animal
            assumes to be T x num_cells
        raw_input_dim (int): which dimension contains the raw input
    """
    ret_arz = []
    # iter thru animals < cells
    for i in range(len(arz)):
        new_ar = []
        for j in range(np.shape(arz[i])[1]):
            if np.sum(np.isnan(arz[i][:,j])) > 0:
                v = _sub_cells_helper(arz[i][:,raw_input_dim],
                                      raw_input_dim,
                                      j,
                                      arz,
                                      miss_tolerance)
            else:
                v = arz[i][:,j]
            new_ar.append(v)
        new_ar = np.array(new_ar).T
        ret_arz.append(new_ar)
    return ret_arz


def filter_cells(arz: List[np.ndarray],
                 target_cell: int):
    """filter out animals that are missing
    target cell

    Args:
        arz (List[np.ndarray]): each array = 
            different animal
        target_cell (int): index of target cell
            if has nans --> animal filtered out
    
    Returns:
        List[np.ndarray]: animals not missing target
            cell
    """
    ar2 = []
    for ari in arz:
        if np.sum(np.isnan(ari[:, target_cell])) < 1:
            ar2.append(ari)
    return ar2


def get_max_time(setz: List[List[np.ndarray]]):
    """Search across all animals --> max time

    Args:
        setz (List[List[np.ndarray]]): inner list = a single set
            each array is a single animal
    
    Returns:
        int: max time
    """
    maxt = -1
    for cset in setz:
        for anml in cset:
            ctime = np.shape(anml)[0]
            if ctime > maxt:
                maxt = ctime
    return maxt


def build_standard_id(setz: List[List[np.ndarray]],
                      set_enums: List[np.ndarray]):
    """Build standard identity data
    > 0th dim = time rep
    > all other dims from set_enums

    Args:
        setz (List[List[np.ndarray]]): inner list = a single set
            each array is a single animal
        set_enums (List[np.ndarray]): representation of the sets
            will make up all of the non t-dimensions
    """
    assert(len(setz) == len(set_enums)), "must have enum for each set"
    maxt = get_max_time(setz) * 1.
    id_dat = []
    for i, cset in enumerate(setz):
        id_sub = []
        for anml in cset:
            numt = np.shape(anml)[0]
            curt = np.arange(numt) / maxt
            se_tile = np.tile(np.reshape(set_enums[i], (1,-1)),
                              (numt, 1))
            id_sub.append(np.hstack((curt[:,None], se_tile)))
        id_dat.append(id_sub)
    return id_dat
