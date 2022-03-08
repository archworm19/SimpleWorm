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
from re import sub
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
    """Break up each animal into distinct time windows

    Args:
        x (List[np.ndarray]): target value
            each array is a different animal
        time_windows (List[np.ndarray]): time windows
            matches size of x

    Returns:
        Dict[int, List[np.ndarray]]: mapping from time window
            to list of arrays. Each array in this list
            is from a different animal
    """
    wins = {}
    for i in range(len(x)):
        _break_windows(x[i], time_windows[i], wins)
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
        st_dat = np.hstack(wd[k])
        varz.append(np.var(st_dat))
    return varz


def window_supervariance_anml(x: List[np.ndarray],
                              time_windows: List[np.ndarray]):
    """Calculate super variance for each time window across
    all animals
    Supervariance: calc variance within each animal
        -> average across animals, within time window

    Args:
        x (List[np.ndarray]): target data
        time_windows (List[np.ndarray]): time windows
            matches shape of x
    """
    wd = window_break_anmls(x, time_windows)
    # order the keys:
    kord = np.sort(list(wd.keys()))
    super_varz = []
    for k in kord:
        sub_dat = wd[k]
        sub_varz = [np.var(sdi) for sdi in sub_dat]
        super_varz.append(np.mean(sub_varz))
    return super_varz


def variance_anml(x: List[np.ndarray],
                  time_vals: List[np.ndarray],
                  window_size: int,
                  super_var: bool = True):
    """Run full variance experiment

    Args:
        x (List[np.ndarray]): target data
        time_vals (List[np.ndarray]): time for 
            each datapoint in target data
        window_size (int)
        super_var (bool): if true --> get super-variance
            if false --> get variance of stacked data
    """
    time_windows = [time_discretize(tvi, window_size)
                     for tvi in time_vals]
    if super_var:
        return window_supervariance_anml(x, time_windows)
    else:
        return window_variance_anml(x, time_windows)


def _doublewin_variance(x_win: List[np.ndarray],
                        T_awin: int):
    """Double Window Variance Calculation
    Considers a single time window across animals
    > Find all analysis time windows in this time window
    > > across animals
    > Subtract from mean representation

    How differ from variance?
    Consider where T_awin approx = len(x_win)
    AND every animal has same complex patrn for given x_win
    > Variance will be high
    > Double window variance = 0

    Args:
        x_win (List[np.ndarray]): single window each
            array is a different anml
        T_awin (int): Analysis timewindow

    Returns:
        float: variance calculation for the window across
            time and across animals
    """
    x_align_stack = []
    for xi in x_win:
        if len(xi) - T_awin <= 0:
            continue
        for z in range(0, len(xi) - T_awin):
            x_align_stack.append(xi[z:z+T_awin])
    x_align_stack = np.array(x_align_stack)
    mu = np.mean(x_align_stack, axis=0, keepdims=True)
    var = np.mean((x_align_stack - mu) ** 2.0)
    return var

def doublewindow_variance_anml(x: List[np.ndarray],
                              time_windows: List[np.ndarray],
                              T_awin: int):
    """Runs double window variance calculation for 
        all timewindows and all animals

    Args:
        x (List[np.ndarray]): target data
        time_windows (List[np.ndarray]): time windows
            matches shape of x
    """
    wd = window_break_anmls(x, time_windows)
    # order the keys ~ different timewindows
    kord = np.sort(list(wd.keys()))
    varz = []
    for k in kord:
        varz.append(_doublewin_variance(wd[k], T_awin))
    return varz

def run_double(x: List[np.ndarray],
                  time_vals: List[np.ndarray],
                  window_size: int,
                  T_awin: int):
    """Run Double Window Experiment
    See _doublewin_variance DocString

    Args:
        x (List[np.ndarray]): target data
        time_vals (List[np.ndarray]): time for 
            each datapoint in target data
        window_size (int)
        T_awin (int): Analysis Timewindow
            should be < window_size

    """
    time_windows = [time_discretize(tvi, window_size)
                    for tvi in time_vals]
    return doublewindow_variance_anml(x, time_windows, T_awin)


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

    window_sizes = [10, 20, 40, 60]
    varz = []
    for i, ws in enumerate(window_sizes):
        #v_sub = variance_anml([x, x2], [ar, ar2], ws, super_var=True)
        v_sub = run_double([x, x2], [ar, ar2], ws, 5)
        print(v_sub)
        input('cont?')
 
    