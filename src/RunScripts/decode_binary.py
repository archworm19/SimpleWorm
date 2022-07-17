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
    return "binaryDecode_softForest_twin{0}_inp{1}_targ{2}".format(str(ecf.twindow_size),
                                                                   str(ecf.input_cells),
                                                                   str(ecf.target_cell))


def _experiment_configs(data_loader_cells: List[str]):
    """returns list of experiment configs"""
    # only 1 experiment here --> all cells as inputs
    # and input as target
    input_cells = ['AVA', 'RME', 'SMDV', 'SMDD']
    target_cell = 'STIM'
    # convert input cells to indices
    input_cell_inds = []
    for ic in input_cells:
        match_inds = [z for z, dlc in enumerate(data_loader_cells)
                      if ic == dlc]
        assert(len(match_inds) == 1)
        input_cell_inds.append(match_inds[0])
    # convert target to index
    targ_cell_ind_raw = [z for z, dlc in enumerate(data_loader_cells)
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
                              ecf: ExperimentConfig):
    # TODO: before packaging into tf.dataset, right???
    """convert to tensorflow-based filesystem + perform pre-processing
    Assumes: set of animals
        > sub-lists = different animal types / setup types
        > arrays = different animals"""
    all_file_sets = []
    assert(twindow_size % 2 == 0)
    # NOTE: ASSUMPTION = start in middle of window
    # TODO: should maybe be in configuration
    kOFF = int(ecf.twindow_size / 2)
    # iter thru sets
    for cell_set, id_set, set_name in zip(cell_clusts, id_data, set_names):
        x, y, data_weights = [], [], []
        # iter thru animals
        for cell_worm, id_worm in zip(cell_set, id_set):
            # cell index selection
            cells_in = cell_worm[ecf.input_cells]
            cells_out = cell_worm[ecf.target_cell]

            # timewindow packaging
            # x1: input_cells
            _legal_ts_x1, x1 = _apply_timewindow(cell_worm, ecf.twindow_size, 0)
            # x2: id data (no twindows needed):
            _legal_ts_x2, x2 = _apply_timewindow(id_worm, 1, kOFF)[-1*kOFF]
            # y: target_cells
            _legal_ts_y, y1 = _apply_timewindow(cell_worm, 1, kOFF)[-1*kOFF]

            assert(len(x1) == len(x2))
            assert(len(x1) == len(y))
            x.append([x1, x2])
            y.append(y1)
            # TODO: save legal_ts?

            # TODO: data_weight formulation?
            # calculate dataweights: T x 
            # data_weights.append(npr.random([len(x1), num_models]) < data_weight_prob)
        # tensorflow file sets:
        file_set = utils_tf.build_file_set(x, y, data_weights, os.path.join(temp_dir_name, set_name))
        all_file_sets.append(file_set)
    root_file_set = utils_tf.combine_file_sets(all_file_sets)


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


    # TODO: should probably do multiprocessing somewhere == parallelize across models
    # ... probably... == better from space perspective


    # TODO: complete _convert_to_tf_filesystem
    # ... issue = data_weight formulation = depends on model...


    # TODO: cv/test sampling

    # TODO: we need a helper function to gather all files (exists) and package into single tf dataset