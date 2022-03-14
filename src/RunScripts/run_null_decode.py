""" 
    Run Null Decode

    Goal/Q: how well can we decode input from time alone?
    > Decoding primary sensory neurons from just time
    > Look at training / test performance as decrease variance of time variable

"""
from typing import List
import numpy as np
import pylab as plt
import data_loader
import utils

def _test_replacement(cset_og: List[np.ndarray],
                      cset_new: List[np.ndarray]):
    for i in range(len(cset_og)):
        for j in range(np.shape(cset_og[i])[1]):
            di = cset_og[i][:,j] - cset_new[i][:,j]
            if np.sum(np.isnan(di)) > 0 or np.nansum(di**2.) > 1.:
                plt.figure()
                plt.plot(cset_og[i][:,6])  # input
                plt.plot(cset_og[i][:,j])
                plt.plot(cset_new[i][:,j])
                plt.show()


if __name__ == '__main__':

    # TODO: run params ~ variances / timewindows

    # load data sets:
    cell_clusts, _, _ = data_loader.load_all_data()

    # ON cell targets
    # animals must have ON cells
    cell_clusts_on = [utils.filter_cells(cset, 4) for cset in cell_clusts]
    # Off cell targets ~ fairly rare in No Stim dataset
    cell_clusts_off = [utils.filter_cells(cset, 5) for cset in cell_clusts]

    # TESTING:
    for i in range(len(cell_clusts)):
        print('Num animals: total; ON; OFF')
        print(len(cell_clusts[i]))
        print(len(cell_clusts_on[i]))
        print(len(cell_clusts_off[i]))
