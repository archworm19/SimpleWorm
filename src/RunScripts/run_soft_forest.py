"""Soft Forest Run Script
    Should supplant other run scripts"""

import os
import numpy as np
import numpy.random as npr
import data_loader
import utils_tf
from sampling_utils import cell_sample, common_legal_ts


def _make_worm(cell_clusts, id_data, cell_inds_x, cell_inds_y,
                offset_x, offset_y, t_window):
    """package data for a single worm
    calculates offsets and filters for target cells
    ASSUMES: only predicting cells"""
    x, legal_txs = cell_sample(cell_clusts, cell_inds_x, offset_x, t_window)
    y, legal_tys = cell_sample(cell_clusts, cell_inds_y, offset_y, t_window)
    legal_ids = np.arange(0, len(id_data))
    legal_ts, inds = common_legal_ts([legal_txs, legal_tys, legal_ids])
    return legal_ts, x[inds[0]], y[inds[1]], legal_ids[inds[2]]


if __name__ == "__main__":
    # experimental params
    twindow_size = 12
    # consecutive t0 windows going in train vs. cross-validation
    BLOCK_SIZE = 200
    
    # 4 = ON cell
    target_inds = np.array([4])
    input_inds = np.arange(4)  # all cells
    # time params
    time_offset = 0
    time_window = 16


    # Soft Forest Run params:
    temp_dir_name = "TEMP/"
    num_base_model = 8
    num_model_per_base = 16
    num_models = num_base_model * num_model_per_base
    data_weight_prob = 1. / num_model_per_base
    data_weight_gen = npr.default_rng(42)



    # load data sets:
    cell_clusts, id_data, cell_names, set_names = data_loader.load_all_data()

    # cell selection system + offsetting:
    # and convert to tf-based file sets
    all_file_sets = []
    for cell_set, id_set, set_name in zip(cell_clusts, id_data, set_names):
        x, y, data_weights = [], [], []
        for cell_worm, id_worm in zip(cell_set, id_set):
            legal_ts, xi, yi, xidi = _make_worm(cell_worm, id_worm,
                                                input_inds, target_inds,
                                                time_offset, time_offset, time_window)
            x.append([xi, xidi])
            y.append(yi)
            # TODO: save legal_ts?
            # calculate dataweights: T x 
            data_weights.append(npr.random([len(legal_ts), num_models]) < data_weight_prob)
        # tensorflow file sets:
        file_set = utils_tf.build_file_set(x, y, data_weights, os.path.join(temp_dir_name, set_name))
        all_file_sets.append(file_set)
    root_file_set = utils_tf.combine_file_sets(all_file_sets)

    # TODO: cv/test sampling

    # TODO: we need a helper function to gather all files (exists) and package into single tf dataset