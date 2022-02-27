"""Hardcoded functions to get worm data in correct format"""
from typing import List
import os
import copy
import numpy as np
import utils


def _make_2d(ar: np.ndarray):
    """If array is NOT 2d --> reshape it
    into a column array

    Args:
        ar (np.ndarray): target array
    """
    if len(np.shape(ar)) == 1:
        return np.reshape(ar, (-1,1))
    return ar


def _get_reference_keys(fn: str):
    """get keys from npz file

    Args:
        fn (str): name of npz file
    """
    return list(np.load(fn).keys())


def _load_npz(fn: str,
              keys: List[str]):
    """Load npz file

    Args:
        fn (str): name of npz
        keys (List[str], optional): set of keys to looks for
            return in order of keys
            Throws error if missing key 
    
    Returns: List[np.ndarray]
        in order defined by keys

    """
    dat = np.load(fn)
    dat2 = []
    for k in keys:
        dat2.append(dat[k])
    return dat2


def _zip_stack(dat: List[List[np.ndarray]]):
    """Sub-lists are aligned --> stack dat[i][j] with dat[k][j]

    Args:
        dat (List[List[np.ndarray]]):
    """
    flat_dat = []
    for i in range(len(dat[0])):
        v = [dat[j][i] for j in range(len(dat))]
        flat_dat.append(np.hstack(v))
    return flat_dat


def _load_set(rdir: str, fns: List[str],
              ave_bools: List[bool]):
    """Load a single set; all fns in the set
    must have the same reference keys

    Args:
        rdir (str): root directory
        fns (List[str]): filenames
        ave_bool (List[bool]): booleans that
            indicate whether subset should be averaged
            Shape must match fns 
    """
    ref_keys = _get_reference_keys(os.path.join(rdir, fns[0]))
    dat = []
    for i, ave_bool in enumerate(ave_bools):
        fn = os.path.join(rdir, fns[i])
        ar_l = _load_npz(fn, ref_keys)
        # ensure 2d and contract if necessary:
        ar_l2 = []
        for j in range(len(ar_l)):
            v = _make_2d(ar_l[j])
            if ave_bool:
                v = np.mean(v, axis=1, keepdims=True)
            ar_l2.append(v)
        dat.append(ar_l2)
    return _zip_stack(dat)


def _load_zim(rdir: str):
    """Load zim + op50 data
    NOTE: zim data needs primary sensory neurons inserted"""
    # OP50 + zim:
    zim_op50 = [['Yanno_op50_SF.npz', 'Yop50_SF_psON.npz', 'Yop50_SF_psOFF.npz', 'inpfull_op50_SF.npz'],
                ['Yanno_op50_mixMotif.npz', 'Yop50_mixMotif_psON.npz', 'Yop50_mixMotif_psOFF.npz', 'inpfull_op50_mixMotif.npz'],
                ['Yanno_op50_pulseBias.npz', 'Yop50_pulseBias_psON.npz', 'Yop50_pulseBias_psOFF.npz', 'inpfull_op50_pulseBias.npz']]
    ave_bools = [False, True, True, False]
    flat_dat = []
    for fn_set in zim_op50:
        flat_dat.extend(_load_set(rdir, fn_set, ave_bools))
    return flat_dat


def _load_npal(rdir: str):
    fns = ['Yanno_Neuropal.npz', 'YNeuropal_psON.npz', 'YNeuropal_psOFF.npz', 'inpfull_Neuropal.npz']
    ave_bools = [False, True, True, False]
    return _load_set(rdir, fns, ave_bools)


def _load_nostim(rdir: str):
    nostim = [['ztc_nostimclusts.npz', 'ztc_nostimstim.npz'],
              ['jh_DIACneg5_trial2_bufferclusts.npz', 'jh_DIACneg5_trial2_bufferstim.npz'],
              ['jh_DIACneg7_trial2_bufferclusts.npz', 'jh_DIACneg7_trial2_bufferstim.npz'],
              ['jh_IAAneg4_run1_trial2_bufferclusts.npz', 'jh_IAAneg4_run1_trial2_bufferstim.npz'],
              ['jh_IAAneg6_run1_trial2_bufferclusts.npz', 'jh_IAAneg6_run1_trial2_bufferstim.npz']]
    ave_bools = [False, False]
    flat_dat = []
    for fn_set in nostim:
        flat_dat.extend(_load_set(rdir, fn_set, ave_bools))
    return flat_dat


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
    cc, ids = load_all_data()
    import pylab as plt
    for k, v in enumerate(cc):
        print('\n ' + str(k))
        for vi in v:
            print(np.shape(vi))
            plt.figure()
            plt.plot(vi[:,0])
            plt.plot(vi[:,4])
            plt.plot(vi[:,5])
            plt.plot(vi[:,6])
            plt.show()
    