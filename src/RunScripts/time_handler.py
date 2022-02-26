""" 
    Time Handler
    A number of utilities to process time


    Time Discretization
    > Break up time axis (0,1,2,...) into time windows
    > > (0,1,2,...) -> (0,0,0,...)
    > Why?
    > > To probe whether there is a general region effect of time
    > > Compare it to time memorization

"""
from typing import List, Dict
import numpy as np

def time_discretize(time_vals: np.ndarray,
                    window_size: int):
    """Discretize time values

    Args:
        time_vals (np.ndarray): array containing 
            time values
        window_size (int): size of each window

    Returns:
        np.ndarray: array of the same length as
            time_vals ~ ordered integers
            where each integer indicates time region
    """
    v = time_vals / (1. * window_size)
    return v.astype(np.int32)


def _break_windows(x: np.ndarray,
                   time_windows: np.ndarray,
                   save_dict: Dict):
    """Break windows for a single animal

    Args:
        x (np.ndarray): target values
        time_windows (np.ndarray): time windows
            shape must match x
            must be ordered
        save_dict: mapping from time window to
            List[np.ndarray] = datapoints for given
            time window; this func updates this guy
    """
    uniqs, uidx = np.unique(time_windows, return_index=True)
    twins = np.split(x, uidx)[1:]
    for i, twin in enumerate(twins):
        k = uniqs[i]
        if k not in save_dict:
            save_dict[k] = []
        save_dict[k].append(twin)


def window_break_anmls(x: List[np.ndarray],
                       time_windows: List[np.ndarray]):
    """Calculate the variance in each time window across anmls

    Args:
        x (List[np.ndarray]): target value
            each array is a different animal
        time_windows (List[np.ndarray]): time windows
            matches size of x

    Returns:
        Dict: mapping from time window to array
            array = all datapoints (subset of x)
            that occur in this time window across anmls
    """
    wins = {}
    for i in range(len(x)):
        _break_windows(x[i], time_windows[i], wins)
    for k in wins:
        wins[k] = np.hstack(wins[k])
    return wins


def window_variance_anml(x: List[np.ndarray],
                         time_windows: List[np.ndarray]):
    """Calculate variance for each time window across 
    all animals

    Args:
        x (List[np.ndarray]): target data
        time_windows (List[np.ndarray]): time windows
            matches shape of x
    """
    wd = window_break_anmls(x, time_windows)
    # order the keys:
    kord = np.sort(list(wd.keys()))
    varz = []
    for k in kord:
        varz.append(np.var(wd[k]))
    return varz


def variance_anml(x: List[np.ndarray],
                  time_vals: List[np.ndarray],
                  window_size: int):
    """Run full variance experiment

    Args:
        x (List[np.ndarray]): target data
        time_vals (List[np.ndarray]): time for 
            each datapoint in target data
    """
    time_windows = [time_discretize(tvi, window_size)
                     for tvi in time_vals]
    return window_variance_anml(x, time_windows)





if __name__ == '__main__':
    ar = np.arange(100)
    ar2 = np.arange(75)
    time_wins = time_discretize(ar, 20)
    time_wins2 = time_discretize(ar, 20)

    x = np.zeros((100))
    x[30:60] = 1
    x2 = np.zeros((75))
    x2[30:60] = 1

    d = {}
    _break_windows(x, time_wins, d)
    print(d)

    v = window_break_anmls([x, x2], [time_wins, time_wins2])
    print(v)

    window_sizes = [5, 10, 20, 40, 60]
    varz = []
    for i, ws in enumerate(window_sizes):
        v_sub = variance_anml([x, x2], [ar, ar2], ws)
        print(v_sub)
        input('cont?')
 
    