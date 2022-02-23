"""Hardcoded function to get worm data in correct format"""
from typing import List
import os
import copy
import numpy as np
import utils

def _load_helper(rdir: str, 
                 fns: List[List[str]],
                 num_cells_use: int = 4):
    """Loads data --> returns as flat list of numpy arrays
    Expected to recieve filenames for a single set
    
    rdir (str): root directory ~ where files are stored
    fns (List[str]): cell and input relative filenames
        each sub-list is a cell - input pair
    num_cells_use (int): use the first [num_cells_use] cells    
    """
    rars = []
    for v in fns:
        cellz = np.load(os.path.join(rdir,v[0]))
        inpz = np.load(os.path.join(rdir, v[1]))
        for k in cellz:
            # ensure input is correct shape:
            inpc = np.reshape(inpz[k], (-1,1))
            rars.append(np.hstack((cellz[k][:,:num_cells_use], inpc)))
    return rars


def _load_zim(rdir: str):
    """Load zim + op50 data"""
    # OP50 + zim:
    zim_op50 = [['Yanno_op50_SF.npz', 'inpfull_op50_SF.npz'],
                ['Yanno_op50_mixMotif.npz', 'inpfull_op50_mixMotif.npz'],
                ['Yanno_op50_pulseBias.npz', 'inpfull_op50_pulseBias.npz']]
    
    return _load_helper(rdir, zim_op50)


def _load_npal(rdir: str):
    npal_op50 = [['Yanno_Neuropal.npz', 'inpfull_Neuropal.npz']]
    return _load_helper(rdir, npal_op50)


def _load_nostim(rdir: str):
    nostim = [['ztc_nostimclusts.npz', 'ztc_nostimstim.npz'],
              ['jh_DIACneg5_trial2_bufferclusts.npz', 'jh_DIACneg5_trial2_bufferstim.npz'],
              ['jh_DIACneg7_trial2_bufferclusts.npz', 'jh_DIACneg7_trial2_bufferstim.npz'],
              ['jh_IAAneg4_run1_trial2_bufferclusts.npz', 'jh_IAAneg4_run1_trial2_bufferstim.npz'],
              ['jh_IAAneg6_run1_trial2_bufferclusts.npz', 'jh_IAAneg6_run1_trial2_bufferstim.npz']]
    v = _load_helper(rdir, nostim)
    return v


def _get_max_tlen(ars: List[np.ndarray]):
    tmax = -1
    for ar in ars:
        if np.shape(ar)[0] > tmax:
            tmax = np.shape(ar)[0]
    return tmax


def _make_id_data(ars: List[np.ndarray],
                  base_ids: np.ndarray,
                  max_tlen: int):
    """Make id data for a single set

    Args:
        ars (List[np.ndarray]): each array is a different
            animal in the set
        base_ids (np.ndarray): ids that describe every 
            animal in the set
        max_tlen (int): maximum tlength across all sets
    
    Returns:
        List[np.ndarray]: each array is an identity array
            = [time, ... other dims specified by base_ids]
    """
    idz = []
    for ar in ars:
        cur_t = np.shape(ar)[0]
        bid = copy.deepcopy(base_ids)[None]
        bid = np.tile(bid, (cur_t, 1))
        t_ar = np.arange(cur_t) / max_tlen
        idz.append(np.hstack((t_ar[:,None], bid)))
    return idz


def _similarity_filter(dmat: np.ndarray,
                       sim_thresh: float = .01):
    """Return indices that pass similarity filter check
    Do not pass if too similar to an index that has already passed
    check

    Args:
        dmat (np.ndarray): distance matrix with nans
            where non-comperable 
    """
    inds = []
    for i in range(np.shape(dmat)[0]):
        sinds = np.where(dmat[i,:] < sim_thresh)[0]
        ipass = True
        for si in sinds:
            if si in inds:
                ipass = False
        if ipass:
            inds.append(i)
    return inds        


def _select_subsets(dsets: List[List[np.ndarray]],
                    use_inds: List):
    """use_inds = flat; dsets = structure
    select dsets according to use_inds"""
    use_inds = set(use_inds)
    ds2 = []
    count = 0
    for ds in dsets:
        sub_ds2 = []
        for dsi in ds:
            if count in use_inds:
                sub_ds2.append(dsi)
            count += 1
        ds2.append(sub_ds2)
    return ds2


def load_all_data():
    """Hardcoded function to represent all data
    """
    rdir = '/Users/ztcecere/Data/ProcCellClusters'
    # filenames

    # OP50 + zim:
    zim_op50 = _load_zim(rdir)
    zim_op50_ids = [1, 0, 1, 0]

    # OP50 + npal:
    npal_op50 = _load_npal(rdir)
    npal_op50_ids = [1, 0, 0, 1]

    # nostim
    nostim = _load_nostim(rdir)
    nostim_ids = [0, 1, 1, 0]

    # get tmax:
    tmax = _get_max_tlen(zim_op50 + npal_op50 + nostim)

    # make ids
    cell_clusts = [zim_op50, npal_op50, nostim]
    id_dat = [zim_op50_ids, npal_op50_ids, nostim_ids]
    all_ids = []
    for i in range(len(cell_clusts)):
        print('making id data')
        base_id_ar = np.array(id_dat[i])
        all_ids.append(_make_id_data(cell_clusts[i], base_id_ar, tmax))

    # ensure we don't have any extra data:
    dmat = utils.get_all_array_dists(zim_op50 + npal_op50 + nostim)
    pass_inds = _similarity_filter(dmat)
    print(len(pass_inds))

    # select passing indices:
    cell_clusts = _select_subsets(cell_clusts, pass_inds)
    all_ids = _select_subsets(all_ids, pass_inds)

    dmat = utils.get_all_array_dists(cell_clusts[0] + cell_clusts[1] + cell_clusts[2])
    print('min defined tday distance: {0}'.format(np.nanmin(dmat)))

    return cell_clusts, all_ids


if __name__ == '__main__':
    load_all_data()
    