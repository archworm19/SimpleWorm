"""Cell / Window Sampling utils

    Why here? Because they are specific to the worm experiments
        = makes a bunch of assumptions about the shapes of the input arrays
        + should be done before cross-validation/test set sampling

    Design:
    > params
    > > cell ids
    > > t0, t1
    > constaints
    > > for a given array, all cells share t0, t1
    ... to use multiple windows --> create separate arrays
    > enforcing consistency
    > > when sampling, returns legal t0s for all animals
    > > provides checking functionality for legal windows for all data
"""
from typing import List
import numpy as np


def cell_sample(x: np.ndarray, select_inds: np.ndarray, t_offset: int, t_winsize: int):
    """Sample cells + get timewindows
    Assumes: x = T x N array = cell format

    Offset = the start of the window relative to the legal_t returned
        for the window
    Ex: for offset = -5 --> 0th legal t = t0+5 while the values
        returned for the target cell will be for t0:t0+t_winsize
    NOTE: for -t_offset, it is possible to get windows that are outside the t_range
        originally defined for the dataset (if |t_offset| > t_winsize)
        conversely, if t_offset is large enough --> it is possible to get negative
        legal_t0s

    Args:
        x (np.ndarray): cell array
            T x N array (time x num cells)
        select_inds (np.ndarray): which cells to use
        t_offset (int): time offset
        t_winsize (int): time window size

    Returns:
        np.ndarray: windowed, selected data
            T1 x t_winsize x N1
        np.ndarray: t0s for the legal timewindows
            associated with the windowed data
    """
    # get cells first:
    x_cell = x[:, select_inds]

    # approach
    # > get all possible assuming offset = 0
    # > assign legal_ts
    T = len(x_cell)
    x_wins = [x_cell[i:T-t_winsize+1+i][:,None,:] for i in range(t_winsize)]
    xfull = np.concatenate(x_wins, axis=1)
    return xfull, np.arange(-1 * t_offset, len(xfull) - 1 * t_offset)
    

def common_legal_ts(legal_ts: List[np.ndarray]):
    """Find and return the common legal timepoints

    Args:
        legal_ts (List[np.ndarray]): legal timepoints
            typically, across the different covariates
            for the same animal
            Ex: x, y from one animal 
            ASSUMES: unique

    Returns:
        np.ndarray: intersection of all legal_ts
            in sorted order
        List[np.ndarray]: indices for each passed in legal_ts
            that map to the intersected array (first returned value)

    """
    # intersect all
    isect_legal = legal_ts[0].copy()
    for lti in legal_ts[1:]:
        isect_legal = np.intersect1d(isect_legal, lti, assume_unique=True)
    # get indices by repeating intersection
    inds = []
    for lti in legal_ts:
        _, _, cinds = np.intersect1d(isect_legal, lti, assume_unique=True, return_indices=True)
        inds.append(cinds)
    return isect_legal, inds


if __name__ == "__main__":

    x = np.tile(np.arange(500)[:,None], (1, 5))
    x_wins, legal_ts = cell_sample(x, np.array([0,1,2]), -5, 10)
    assert(len(x_wins) == len(legal_ts))
    print(x_wins[0,:,0])
    print(x_wins[-1,:,0])
    print(legal_ts)
    
    # test intersection
    x2_wins, legal_ts2 = cell_sample(x, np.array([0,1,2]), -3, 10)
    print(legal_ts2)

    isect_legal, inds = common_legal_ts([legal_ts, legal_ts2])
    print(isect_legal)

    # align:
    x_align = x_wins[inds[0]]
    x2_align = x2_wins[inds[1]]
    print(x_align[0])
    print(x2_align[0])


        
        