""" 
    Run Null Decode

    Goal/Q: how well can we decode input from time alone?
    > Decoding primary sensory neurons from just time
    > Look at training / test performance as decrease variance of time variable

"""
from statistics import variance
from typing import List
import numpy as np
import pylab as plt
import data_loader
import utils

from Sampler.experiments import build_anml_factory_multiset
from Models.KNN import knn


# TODO: this should be moved to some other file
# ... might need to be imported from anothe run script
def build_standard_null_masks(twindow_size: int,
                              num_cells: int,
                              target_cell_idx: int,
                              num_id_dims: int):
    """Build the standard null masks
    builds 4 masks
    > t_mask (timeseries) and id_mask(identity)
        for independent and dependent variables
    
    Assumptions:
    > Predicting values for a single cell
    > (null assumption): no cells are used for independent vars
    > (null assumption): all of identity vars are used
    > timeseries data = twindow_size x num_cell
    > identity data = num_id_dims

    Args:
        twindow_size (int): 
        num_cells (int): number of cells per animal
        target_cell_idx (int): index of target cell
            target cell = the one cell we are trying
            to predict
        num_id_dims (int): number of dims for identity data
    
    Returns: (all masks are booleans)
        np.ndarray: independent timeseries mask
            twindow_size x num_cell
        np.ndarray: independent identity mask
            num_id_dims
        np.ndarray: dependendent timeseries mask
            twindow_size x num_cell
        np.ndarray: dependent identity mask
            num_id_dims

    """
    # independent first:
    t_indep = np.full((twindow_size, num_cells), False)
    id_indep = np.full((num_id_dims, ), True)
    # dependent:
    t_dep = np.full((twindow_size, num_cells), False)
    t_dep[:, target_cell_idx] = True
    id_dep = np.full((num_id_dims, ), False)
    return t_indep, id_indep, t_dep, id_dep


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
    twindow_size = 12
    train_sample_perc = 1. / twindow_size

    # set enumerations ~ will be incorporated into id_dat
    # standard set ordering = [Zim+op50, Npal+op50, Zim+buffer]
    # dims = [zim yea, npal yea, op50 yea, buffer yea]
    set_enums = [[1, 0, 1, 0],
                 [0, 1, 1, 0],
                 [1, 0, 0, 1]]
    
    # masks: 7 cells (6 cells + stimulus)
    target_idx_dim = 4  # ON cell
    t_indep_mask, id_indep_mask, t_dep_mask, id_dep_mask = build_standard_null_masks(twindow_size,
                                                                  7,
                                                                  target_idx_dim,
                                                                  1 + len(set_enums[0]))

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

    # pull identity data:
    id_dat = utils.build_standard_id(cell_clusts_on,
                                     set_enums)
    

    # TESTING ~ ON Cells
    train_factory, test_sampler = build_anml_factory_multiset(cell_clusts_on,
                                                              id_dat,
                                                              twindow_size,
                                                              t_dep_mask,
                                                              id_dep_mask,
                                                              t_indep_mask,
                                                              id_indep_mask,
                                                              rand_seed=42)
    
    # get the samplers
    train_sampler, test_sampler = train_factory.generate_split()

    # TESTING: time variance alteration
    t_vars = np.array([.0001, 1., 0.1, 0.01])
    variances = np.ones((len(t_vars), 5))
    variances[:,0] = t_vars
    # TODO: training stuff is below here
    knn_model = knn.KNN(8, variances, train_sample_perc=(1. / twindow_size))
    _, lls = knn_model.train_epoch(train_sampler)
    # print(lls)

    # TESTING: test log-like
    lls = knn_model.test_loglike(train_sampler, test_sampler, 2)
    print(np.mean(lls))

