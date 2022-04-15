""" 
    Run Null Decode

    Goal/Q: how well can we decode input from time alone?
    > Decoding primary sensory neurons from just time
    > Look at training / test performance as decrease variance of time variable

    TODO: not sure if I want null as standalone run script
    BUT: ok for testing for now

"""
from tarfile import BLOCKSIZE
from typing import List
import numpy as np
import numpy.random as npr
import pylab as plt
import data_loader
import utils

from Sampler.set_sampling import sample_avail_files, get_anml_sample_switch, get_anml_sample_allt0
from build_masks import build_standard_null_masks
from Models.KNN import knn


# TODO: I've forgotten the point of this
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
    # consecutive t0 windows going in train vs. cross-validation
    BLOCK_SIZE = 200
    
    # masks: 7 cells (6 cells + stimulus)
    target_idx_dim = 4  # ON cell
    # NOTE: there are 5 id dims for data loader ~ time + 4 for set indicators
    t_indep_mask, id_indep_mask, t_dep_mask, id_dep_mask = build_standard_null_masks(twindow_size,
                                                                  7,
                                                                  target_idx_dim,
                                                                  5)

    # load data sets:
    cell_clusts, id_data, cell_names, set_names = data_loader.load_all_data()

    # convert to file reps format:
    root_set = utils.build_multi_file_set(cell_clusts, id_data, "root_set")

    # NOTE: correct order = 
    # 1. Train/CV vs. Test
    # 2. Train vs. CV
    # 3. On and Off separately (where defined)
    sampler_rng = npr.default_rng()
    # Train/CV vs. Test
    trcv_set, test_set = get_anml_sample_allt0(root_set, .667, sampler_rng)

    NUM_BOOT = 5
    for _ in NUM_BOOT:
        # train vs. cross-validation
        tr_set, cv_set = get_anml_sample_switch(trcv_set, 0.5, BLOCK_SIZE, twindow_size, 0, sampler_rng)

        # On vs. Off cell availability

        ontr_set = sample_avail_files(tr_set, 4)
        offtr_set = sample_avail_files(tr_set, 5)

        oncv_set = sample_avail_files(cv_set, 4)
        offcv_set = sample_avail_files(cv_set, 5)

        # TODO: training / test code



    """
    # TODO: this is old training code... need updating???
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
    """

