"""
    (Boolean) Masks are used to define experiments
    Ex: build a dependant variable mask that has Trues
        for only 1 cell
        --> in experiment, model will try to predict
        the values for that 1 cell

"""
import numpy as np

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
