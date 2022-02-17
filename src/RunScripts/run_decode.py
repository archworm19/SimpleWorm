"""Run decoder modeling"""
import os
import copy
import numpy as np

import utils
from inp_data import  TrialTimeData

if __name__ == '__main__':
    rdir = '/Users/ztcecere/Data/ProcCellClusters'

    # id logic: [food type dims (2), strain dims (2)]

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

    # flatten out data ~ ensuring time series
    # data and identity data match
    all_dats, all_idz, tlens = [], [], []
    for i, cc in enumerate(cell_clusts):
        for fn in cc:
            npz_dat = np.load(os.path.join(rdir,fn))
            for k in npz_dat:
                all_dats.append(npz_dat[k])
                all_idz.append(copy.deepcopy(id_dat[i]))
                tlens.append(np.shape(npz_dat[k])[0])

    # package into List[TrialTimeData]
    pkg_tdat = []
    max_tlen = np.amax(np.array(tlens))
    print('max tlen: {0}'.format(max_tlen))
    for i in range(len(all_dats)):
        cur_tlen = np.shape(all_dats[i])[0]
        cid = np.array(all_idz[i])
        tile_idz = np.tile(cid[None], (cur_tlen,1))
        twins = np.arange(cur_tlen) / max_tlen
        ids_combos = np.hstack((twins[:,None], tile_idz))
        pkg_tdat.append(TrialTimeData(all_dats[i], ids_combos))

    print('number of anmls: {0}'.format(len(pkg_tdat)))

    # ensure we don't have any extra data:
    _tdat = [TTD.time_data for TTD in pkg_tdat]
    dmat = utils.get_all_array_dists(_tdat)
    print('min defined tday distance: {0}'.format(np.nanmin(dmat)))




