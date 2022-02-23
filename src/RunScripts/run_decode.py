"""Run decoder modeling"""
import os
import copy
import numpy as np

from data_loader import load_all_data

from Sampler.experiments import build_anml_factory_multiset


def load_null_masks(tsize: int,
                    cells_out: np.ndarray,
                    num_id_dims: int):
    """Build masks for null experiment
    null experiment = don't use cell data

    Args:
        tsize (int): number of timepoints
            in each window
        cells_out (np.ndarray): boolean array
            of output cells
        num_id_dims (int): number of id dimensions
    """ 
    # dep_t_mask = dependent timeseries dims
    num_cells = len(cells_out)
    co_copy = copy.deepcopy(cells_out)

    dep_t_mask = np.tile(co_copy[None], (tsize,1))
    # indep_t_mask = independent timseries dims
    indep_t_mask = np.full((tsize,num_cells), False)
    
    # dep_id_mask = dependent identity dims:
    dep_id_mask = np.full((tsize,num_id_dims), False)
    # indep_id_mask = independent identity dims:
    indep_id_mask = np.full((tsize,num_id_dims), True)
    return [dep_t_mask, indep_t_mask, dep_id_mask, indep_id_mask]


def load_full_masks(tsize: int,
                    cells_in: np.ndarray,
                    cells_out: np.ndarray,
                    num_id_dims: int):
    """Build masks for full experiment
    full experiment = use all cells other
    than target cells

    Args:
        tsize (int): number of timepoints
            in each window
        cells_in (np.ndarray): boolean array
            of input cells
        cells_out (np.ndarray): boolean array
            of output cells
        num_id_dims (int): number of id dimensions
    """ 
    assert(np.sum(cells_out == cells_in) < 1), "in out cells not disjoint"
    assert(len(cells_out) == len(cells_in)), "in out cell mismatch"

    # dep_t_mask = dependent timeseries dims
    cin_copy = copy.deepcopy(cells_in)
    co_copy = copy.deepcopy(cells_out)

    dep_t_mask = np.tile(co_copy[None], (tsize, 1))
    # indep_t_mask = independent timseries dims
    indep_t_mask = np.tile(cin_copy[None], (tsize, 1))
    
    # dep_id_mask = dependent identity dims:
    dep_id_mask = np.full((tsize,num_id_dims), False)
    # indep_id_mask = independent identity dims:
    indep_id_mask = np.full((tsize,num_id_dims), True)
    return [dep_t_mask, indep_t_mask, dep_id_mask, indep_id_mask]


if __name__ == '__main__':
    rdir = '/Users/ztcecere/Data/ProcCellClusters'

    # id logic: [food type dims (2), strain dims (2)]

    # window tsize for experiment
    EXP_TSIZE = 24
    RAND_SEED = 42

    # load the core dataset:
    t_dat, id_dat = load_all_data()

    # masks define the experiment
    cells_in = np.full((6,), False)
    cells_in[:4] = True
    cells_out = np.full((6,), False)
    cells_out[4:] = True

    exp_masks = [load_null_masks(EXP_TSIZE, cells_out, 4),
                 load_full_masks(EXP_TSIZE, cells_in, cells_out, 4)]
 
    for i, masks in enumerate(exp_masks):
        [dep_t_mask, indep_t_mask, dep_id_mask, indep_id_mask] = masks

        # some safety checks
        assert(np.sum(indep_t_mask * dep_t_mask) < 1), "safety mask check: t"
        assert(np.sum(indep_id_mask * dep_id_mask) < 1), "safety mask check: id"

        # build the train factory and test set
        train_factory, test_sampler = build_anml_factory_multiset(t_dat,
                                                                  id_dat,
                                                                  EXP_TSIZE,
                                                                  dep_t_mask,
                                                                  dep_id_mask,
                                                                  indep_t_mask,
                                                                  indep_id_mask,
                                                                  RAND_SEED)






