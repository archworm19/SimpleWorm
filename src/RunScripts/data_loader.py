"""Hardcoded function to get worm data in correct format"""
import os
import copy
import numpy as np
import utils

def load_all_data():
    """Hardcoded function to represent all data
    """
    rdir = '/Users/ztcecere/Data/ProcCellClusters'
    # filenames

    # OP50 + zim:
    zim_op50 = ['Yanno_op50_SF.npz',
                'Yanno_op50_mixMotif.npz',
                'Yanno_op50_pulseBias.npz']
    zim_op50_ids = [1, 0, 1, 0]
    
    # OP50 + npal:
    npal_op50 = ['Yanno_Neuropal.npz']
    npal_op50_ids = [1, 0, 0, 1]

    # nostim
    nostim = ['ztc_nostimclusts.npz',
              'jh_DIACneg5_trial2_bufferclusts.npz',
              'jh_DIACneg7_trial2_bufferclusts.npz',
              'jh_IAAneg4_run1_trial2_bufferclusts.npz',
              'jh_IAAneg6_run1_trial2_bufferclusts.npz']
    nostim_ids = [0, 1, 1, 0]

    # package data into single list:
    cell_clusts = [zim_op50, npal_op50, nostim]
    id_dat = [zim_op50_ids, npal_op50_ids, nostim_ids]

    # flatten out data within each set
    # data and identity data match
    all_dats, all_ids, tlens = [], [], []
    # iter thru sets
    for i, cc in enumerate(cell_clusts):
        sub_dats, sub_ids = [], []
        for fn in cc:
            npz_dat = np.load(os.path.join(rdir,fn))
            for k in npz_dat:
                sub_dats.append(npz_dat[k])
                sub_ids.append(copy.deepcopy(id_dat[i]))
                tlens.append(np.shape(npz_dat[k])[0])
        all_dats.append(sub_dats)
        all_ids.append(sub_ids)

    # remake all ids to be in tseries format:
    all_ids2 = []
    max_tlen = np.amax(np.array(tlens))
    print('max tlen: {0}'.format(max_tlen))
    for i in range(len(all_ids)):
        sub_ids2 = []
        for j in range(len(all_ids[i])):
            cur_tlen = np.shape(all_dats[i][j])[0]
            cid = np.array(all_ids[i][j])
            tile_idz = np.tile(cid[None], (cur_tlen,1))
            twins = np.arange(cur_tlen) / max_tlen
            ids_combos = np.hstack((twins[:,None], tile_idz))
            sub_ids2.append(ids_combos)
        all_ids2.append(sub_ids2)       
    all_ids = all_ids2

    print('number of animals: {0}'.format(len(all_ids)))

    # ensure we don't have any extra data:
    _tdat = []
    for v in all_dats:
        for vi in v:
            _tdat.append(vi)
    dmat = utils.get_all_array_dists(_tdat)
    print('min defined tday distance: {0}'.format(np.nanmin(dmat)))

    print('set sizes')
    for i in range(len(all_dats)):
        print(len(all_dats[i]))
        print(len(all_ids[i]))

    print('T sizes')
    for d in _tdat:
        print(np.shape(d))
    print('Id sizes')
    for i in range(len(all_ids)):
        for j in range(len(all_ids[i])):
            print(np.shape(all_ids[i][j]))

    return all_dats, all_ids
    