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


def binary_window_variance(binary_input: np.ndarray,
                           time_windows: np.ndarray):
    """Return variance of input within each time window
    As windows get very low --> variance goes to 0
    Assumes: time_windows are ordered

    Args:
        binary_input (np.ndarray): array of length T
            containing 1s and 0s
        time_windows (np.ndarray): ordered array
            of length T describing the windows
    
    Returns:
        np.ndarray: variance for each timewindow
            array where length = number of time windows
    """
    _, uniq_idx = np.unique(time_windows, return_index=True)
    wins = np.split(binary_input, uniq_idx)[1:]
    vars = [np.var(win) for win in wins]
    return np.array(vars)


if __name__ == '__main__':
    ar = np.arange(100)
    time_wins = time_discretize(ar, 20)
    inp = np.zeros((100))
    inp[30:60] = 100
    print(binary_window_variance(inp, time_wins))

    muv = []
    twins = [50, 25, 15, 10, 5]
    for tw in twins:
        time_wins = time_discretize(ar, tw)
        muv.append(np.mean(binary_window_variance(inp, time_wins)))
    import pylab as plt
    plt.figure()
    plt.plot(twins, muv)
    plt.show()
    