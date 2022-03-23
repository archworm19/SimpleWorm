"""ANML Factory

    set splitting is performed across animals
    if an animal appears in train --> it will
    not appear in test
"""
import dataclasses
from typing import List
import numpy as np
import numpy.random as npr
import Sampler.utils as utils
from Sampler.h_sampler import SmallSampler


@dataclasses.dataclass
class FlatDat:
    """3 related files created by the factory
    time_window_starts_fn: dictates the sampling window"""
    t_file_name: str
    id_file_name: str
    time_window_starts_fn: str
    t_file_shape: List[int]
    id_file_dim0: int
    id_file_dim1: int


class ANMLFactoryMultiSet:
    """Full dataset composed of multiple subsets
    Sampling done on each subset"""
    def __init__(self,
                 data: List[List[np.ndarray]],
                 ident_dat: List[List[np.ndarray]],
                 useable_dat: List[np.ndarray],
                 twindow_size: int,
                 dep_t_mask = np.ndarray,
                 dep_id_mask = np.ndarray,
                 indep_t_mask = np.ndarray,
                 indep_id_mask = np.ndarray,
                 tr_prob = 0.5,
                 rand_seed = 66):
        """Lists = independent sets
        Arrays = different animals
        Sampling is done within each set
        NOTE: np.ndarrays can also be np memmap arrays

        Args:
            data (List[List[np.ndarray]]): [description]
            ident_dat (List[List[np.ndarray]]): [description]
            useable_dat (List[np.ndarray]): boolean array
                of len = len(data) where True
                indicates an anml can be used
            twindow_size (int): [description]
            tr_prob (float, optional): [description]. Defaults to 0.5.
        """
        self.data = data
        self.useable_dat = useable_dat
        self.ident_dat = ident_dat
        self.twindow_size = twindow_size
        self.tr_prob = tr_prob
        self.dep_t_mask = dep_t_mask
        self.dep_id_mask = dep_id_mask
        self.indep_t_mask = indep_t_mask
        self.indep_id_mask = indep_id_mask
        self.gen = npr.default_rng(rand_seed)
        assert(len(self.useable_dat) == len(self.data)), "num set mismatch"
        assert(len(self.data) == len(self.ident_dat)), "tdat id_dat mismatch"
        # TODO: there are more assertions required

    def _split_all_sets(self):
        """Split data subset into train vs. test

        Returns:
            np.ndarray: training; integer array of indices
            np.ndarray: testing; integer array of indices

            NOTE: len(training) + len(testing) = total number
                of anmls across all sets
        """
        offset = 0
        flat_train_inds, flat_test_inds = [], []
        for i, _set in enumerate(self.data):
            # get indices for training vs. test animals:
            train_inds, test_inds = utils.generate_anml_split(self.useable_dat[i],
                                                              self.gen, self.tr_prob)
            flat_train_inds.append(train_inds + offset)
            flat_test_inds.append(test_inds + offset)
            offset += len(self.useable_dat[i])
        return np.hstack(flat_train_inds), np.hstack(flat_test_inds)


    def generate_split(self, save_file_root: str = 'TEMP'):
        """Generate split across animals
        """
        # get training / testing indices within each set
        flat_train_inds, flat_test_inds = self._split_all_sets()

        # initialize the memmap arrays:
        # get sizes:
        train_lens, test_lens = [], []
        count = 0
        for cset in self.data:
            for canml in cset:
                v = len(canml) - self.twindow_size
                if flat_train_inds[count]:
                    train_lens.append(v)
                if flat_test_inds[count]:
                    test_lens.append(v)
        
        train_t_file_shape = [sum(train_lens), self.twindow_size] + np.shape(self.data[0][0][0])
        test_t_file_shape = [sum(test_lens), self.twindow_size] + np.shape(self.data[0][0][0])
        

        # flatten other data:
        flat_dat, flat_id = [], []
        for i, vset in enumerate(self.data):
            flat_dat.extend(vset)
            flat_id.extend(self.ident_dat[i])

        # get data offsets:
        offsets = utils.get_data_offsets(flat_dat)

        # stack data and keep track of windows:
        # TODO: this is exactly what happens above --> factor
        ind_sets = [flat_train_inds, flat_test_inds]
        win_sets = []
        for iset in ind_sets:
            win_subs = []
            for ind in iset:
                offset = offsets[ind]
                # NOTE: overlapping allowed here
                win_subs.append(offset + np.arange(0, len(flat_dat[ind]) - self.twindow_size))
            if len(win_subs) > 0:
                win_sets.append(np.hstack((win_subs)))
            else:
                win_sets.append([])

        full_dat = np.vstack(flat_dat)
        full_id_dat = np.vstack(flat_id)

        return [SmallSampler(full_dat, full_id_dat, win_sets[0], self.twindow_size,
                             self.dep_t_mask, self.dep_id_mask, self.indep_t_mask, self.indep_id_mask),
                SmallSampler(full_dat, full_id_dat, win_sets[1], self.twindow_size,
                             self.dep_t_mask, self.dep_id_mask, self.indep_t_mask, self.indep_id_mask)]
