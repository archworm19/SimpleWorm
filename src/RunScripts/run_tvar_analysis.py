"""T Variance Analysis

    Stimulus memorization tradeoff:
    > 2 window sizes
    > > T_disc
    > > > Discretization window sizes
    > > T_awin
    > > > Analysis window size
    > > Consider all analysis windows within each
    > > Discretization window
    > > Combine across animals
    > Q? for fixed T_awin, at what T_disc does variance drop off?
    > == this is the variance where stim memorization is possible
"""
import numpy as np
import pylab as plt
import data_loader
import time_handler

if __name__ == "__main__":
    # NOTE: T_awin should be < T_disc
    T_awin = 24
    T_discs = [30 + 10 * i for i in range(25)]
    cell_index = 6  # Stimulus

    # load data sets:
    cell_clusts, _, _ = data_loader.load_all_data()

    # iter thru each set:
    plt.figure()
    for i, cset in enumerate(cell_clusts):
        time_vals = [np.arange(len(csz)) for csz in cset]
        x = [csz[:,cell_index] for csz in cset]
        vz = []
        for j, T_disc in enumerate(T_discs):
            vz.append(np.nanmean(time_handler.run_double(x, time_vals, T_disc, T_awin)))
        plt.plot(T_discs, vz)
    plt.xlabel('Time Discretization Window Size')
    plt.ylabel('Double Variance')
    plt.show()