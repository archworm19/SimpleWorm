"""Soft Forest Run Script
    Should supplant other run scripts"""

import data_loader
import utils
import utils_tf


if __name__ == "__main__":
    # TODO: run params ~ variances / timewindows
    twindow_size = 12
    # consecutive t0 windows going in train vs. cross-validation
    BLOCK_SIZE = 200
    
    # masks: 7 cells (6 cells + stimulus)
    target_idx_dim = 4  # ON cell

    # load data sets:
    cell_clusts, id_data, cell_names, set_names = data_loader.load_all_data()

    # TODO: cell selection system

    # TODO: convert to tf-based file sets

    # TODO: cv/test sampling

    # TODO: we need a helper function to gather all files (exists) and package into single tf dataset