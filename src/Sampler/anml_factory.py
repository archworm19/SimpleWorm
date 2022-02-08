"""ANML Factory

    set splitting is performed across animals
    if an animal appears in train --> it will
    not appear in test
"""
from typing import List
import numpy as np
import numpy.random as npr
from Sampler.h_sampler import SmallSampler


class ANMLFactory:

    def __init__(self,
                 data: List[np.ndarray],
                 ident_dat: List[np.ndarray],
                 dat_assign: np.ndarray,
                 twindow_size: int,
                 dep_t_mask = np.ndarray,
                 dep_id_mask = np.ndarray,
                 indep_t_mask = np.ndarray,
                 indep_id_mask = np.ndarray,
                 tr_prob = 0.5):
        """[summary]

        Args:
            data (List[np.ndarray]): [description]
            ident_dat (List[np.ndarray]): [description]
            dat_assign (np.ndarray): which arrays in data
                are available for sampling
            twindow_size (int): [description]
            tr_prob (float, optional): [description]. Defaults to 0.5.
        """
        self.data = data
        self.dat_assign = dat_assign
        self.ident_dat = ident_dat
        self.twindow_size = twindow_size
        self.tr_prob = tr_prob
        self.dep_t_mask = dep_t_mask
        self.dep_id_mask = dep_id_mask
        self.indep_t_mask = indep_t_mask
        self.indep_id_mask = indep_id_mask
        assert(len(self.dat_assign) == len(self.data)), "num anml mismatch"

    def _shuffle_sample(self,
                        rand_seed: int,
                        data_len: int,
                        true_perc: float):
        """Shuffle-based sampling

        Args:
            rand_seed (int): rng seeed
            data_len (int): length of data to
                be sampled
            true_perc (float): percent of dataset
                to be marked as true
        
        Returns:
            np.ndarray: data_len boolean array
                representing samples
        """
        gen = npr.default_rng(rand_seed)
        # only consider available data
        inds = np.array([i for i in range(data_len)
                         if self.dat_assign[i]])


        gen.shuffle(inds)
        numtr = int(true_perc * len(inds))
        boolz = np.full((data_len,), False)
        boolz[inds[:numtr]] = True
        return boolz

    def generate_split(self, rand_seed: int):
        """Generate 50/50 split across animals
        using provided random seed

        Args:
            rand_seed (int): random seed for rng
        """
        # sample animals for training
        # True --> train; False --> Cross
        train_anmls = self._shuffle_sample(rand_seed, len(self.data), self.tr_prob)
        # train and test windows ~ can use all overlapping now!
        offset = 0
        train_wins, test_wins = [[]], [[]]
        for i in range(len(self.data)):
            # NOTE: overlapping allowed in this case
            win = offset + np.arange(0, len(self.data[i]) - self.twindow_size)
            offset += len(self.data[i])
            # only use if available to factory
            if self.dat_assign[i]:
                if train_anmls[i]:
                    train_wins.append(win)
                else:
                    test_wins.append(win)
        full_dat = np.vstack([d for d in self.data])
        full_id_dat = np.vstack([idn for idn in self.ident_dat])

        return [SmallSampler(full_dat, full_id_dat, np.hstack(train_wins), self.twindow_size,
                             self.dep_t_mask, self.dep_id_mask, self.indep_t_mask, self.indep_id_mask),
                SmallSampler(full_dat, full_id_dat, np.hstack(test_wins), self.twindow_size,
                             self.dep_t_mask, self.dep_id_mask, self.indep_t_mask, self.indep_id_mask)]

