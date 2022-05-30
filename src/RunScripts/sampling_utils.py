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
import numpy as np


def cell_sample(x: np.ndarray, select_inds: np.ndarray, t_offset: int, t_winsize: int):
    """Sample cells + get timewindows
    Assumes: x = T x N array = cell format

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

    # constraint; when t_offset = 0
    # T_legal = T - T_winsize
    # what about with t_offset?
    # T_legal = T - T_winsize - abs(t_offset) + 1
    legal_t0 = max(-1 * t_offset, 0)
    T_legal = len(x_cell) - t_winsize - np.abs(t_offset) + 1
    legal_ts = np.arange(legal_t0, legal_t0 + T_legal)

    x_wins = []
    for i in range(t_winsize):
        ct0 = legal_t0 + t_offset + i
        ctend = ct0 + T_legal
        x_wins.append(x_cell[ct0:ctend,:][:,None,:])
    x_wins = np.concatenate(x_wins, axis=1)
    return x_wins, legal_ts


# TODO: cell sampling for all animals


if __name__ == "__main__":

    x = np.tile(np.arange(500)[:,None], (1, 5))
    x_wins, legal_ts = cell_sample(x, np.array([0,1,2]), -5, 10)
    print(x_wins[0,:,0])
    print(x_wins[-1,:,0])
    print(legal_ts)


        
        