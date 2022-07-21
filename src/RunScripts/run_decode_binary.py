"""Binary Decoding Problem + Soft Forest
    = Predict probability that stimulus is On vs Off"""

import os
import numpy as np
import numpy.random as npr
import data_loader
import utils_tf
from dataclasses import dataclass
from typing import List
from sampling_utils import cell_sample, common_legal_ts
from Sampler.set_sampling import get_anml_sample_allt0, get_anml_sample_switch
from numpy.random import default_rng


@dataclass
class ExperimentConfig:
    # size of the surrounding timewindow
    twindow_size: int
    # which cells to use
    input_cells: List[int]
    # target cell (y)
    target_cell: int


def _format_experiment_config(ecf: ExperimentConfig):
    """returns a string representation"""
    str_input_cells = "_".join([str(ic) for ic in ecf.input_cells])
    return "binaryDecode_softForest_twin{0}_inp{1}_targ{2}".format(str(ecf.twindow_size),
                                                                   str_input_cells,
                                                                   str(ecf.target_cell))


def _get_experiment_configs(data_loader_cell_names: List[str]):
    """returns list of experiment configs"""
    # only 1 experiment here --> all cells as inputs
    # and input as target
    input_cells = ['AVA', 'RME', 'SMDV', 'SMDD']
    target_cell = 'STIM'
    # convert input cells to indices
    input_cell_inds = []
    for ic in input_cells:
        match_inds = [z for z, dlc in enumerate(data_loader_cell_names)
                      if ic == dlc]
        assert(len(match_inds) == 1)
        input_cell_inds.append(match_inds[0])
    # convert target to index
    targ_cell_ind_raw = [z for z, dlc in enumerate(data_loader_cell_names)
                         if target_cell == dlc]
    assert(len(targ_cell_ind_raw) == 1)
    targ_cell_ind = targ_cell_ind_raw[0]
    # package into object
    return [ExperimentConfig(32, input_cell_inds, targ_cell_ind)]


def _apply_timewindow(ar: np.ndarray, twindow_size: int, twindow_offset: int):
    """Get timewindows for the array,
            starting from offset
       Assumes: array = T x ...
       
       Returns indices (T1,) and windowed array (T1, twindow_size, ...)"""
    inds = [z for z in range(twindow_offset, len(ar) - twindow_size)]
    raw_ars = [ar[ind:ind+twindow_size] for ind in inds]
    return np.array(inds), np.array(raw_ars)


def _convert_to_tf_filesystem(cell_clusts: List[List[np.ndarray]],
                              id_data: List[List[np.ndarray]],
                              set_names: List[str],
                              ecf: ExperimentConfig,
                              temp_dir_name: str):
    """convert to tensorflow-based filesystem + perform pre-processing
    NOTE: this file sets created by this function are independent
        of run parameters --> you can run multiple models with
        its output
    KEY/NOTE: puts y at the center of x-window
    Assumes: set of animals
        > sub-lists = different animal types / setup types
        > arrays = different animals
    Returns a root file set"""
    all_file_sets = []
    assert(twindow_size % 2 == 0)
    # NOTE: ASSUMPTION = start in middle of window
    # TODO: should maybe be in configuration
    kOFF = int(ecf.twindow_size / 2)
    # iter thru sets
    current_absolute_idx = 0  # start from 0
    for cell_set, id_set, set_name in zip(cell_clusts, id_data, set_names):
        x, y = [], []
        # iter thru animals
        for cell_worm, id_worm in zip(cell_set, id_set):
            # cell index selection
            cells_in = cell_worm[:,np.array(ecf.input_cells)]
            cells_out = cell_worm[:,ecf.target_cell]

            # timewindow packaging
            # x1: input_cells
            _legal_ts_x1, x1 = _apply_timewindow(cells_in, ecf.twindow_size, 0)
            # x2: id data (no twindows needed):
            _legal_ts_x2, x2 = _apply_timewindow(id_worm, 1, kOFF)
            x2 = x2[:-1*kOFF + 1]
            # y: target_cells
            _legal_ts_y, y1 = _apply_timewindow(cells_out, 1, kOFF)
            y1 = y1[:-1*kOFF + 1]
            assert(len(x1) == len(x2))
            assert(len(x1) == len(y1))
            # this check is technically redundent but why not?
            assert((_legal_ts_x1[0] + kOFF) == _legal_ts_x2[0])
            x.append([x1, x2])
            y.append(y1)
            # TODO: save legal_ts?
        # tensorflow file sets:
        file_set, current_absolute_idx = utils_tf.build_file_set(x, y, current_absolute_idx,
                                                                 os.path.join(temp_dir_name, set_name))
        all_file_sets.append(file_set)
    root_file_set = utils_tf.combine_file_sets(all_file_sets)
    return root_file_set


if __name__ == "__main__":

    # TODO: should probably do multiprocessing somewhere == parallelize across models
    # ... probably... == better from space perspective

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


    # rngs:
    rng_sampler = default_rng(42)


    # Soft Forest Run params:
    # TODO: need multiple run param configs!!!
    # TODO: match style of Experiment configs...
    temp_dir_name = "TEMP/"
    num_base_model = 8
    num_model_per_base = 16
    num_models = num_base_model * num_model_per_base
    data_weight_prob = 1. / num_model_per_base
    data_weight_gen = npr.default_rng(42)

    # load data sets:
    cell_clusts, id_data, cell_names, set_names = data_loader.load_all_data()

    # ensure cell names constant across sets:
    for cn in cell_names:
        assert(all(cell_names[0][i] == cn[i]) for i in range(len(cn)))

    # iter thru experiment configs:
    for ecf in _get_experiment_configs(cell_names[0]):

        # TODO: update temp_dir_name for current configuration
        # = each config will have a separate underlying dataset
        # cuz dataset is preprocessed into independent(ish) samples
        temp_dir_name_ecf = _format_experiment_config(ecf)
        os.makedirs(temp_dir_name_ecf)

        # convert to tensorflow filesystem:
        # ~ does some preprocessing as well
        root_set = _convert_to_tf_filesystem(cell_clusts, id_data, set_names,
                                             ecf, temp_dir_name_ecf)

        # cv/test sampling ~ keep all t0s
        train_cv_root, test_root = get_anml_sample_allt0(root_set, 0.5, rng_sampler)


        # TODO: maybe print the file ids for the sampled sets as a check?


        # TODO: there's a design issue where _convert_to_tf...
        # saves x tensors separetely with the name scheme:
        # x1, x2
        # TODO: how to integrate this with DataPlan?


        # TODO: need a better run system



    # TODO: we need a helper function to gather all files (exists) and package into single tf dataset