"""ANML Factory

    set splitting is performed across animals
    if an animal appears in train --> it will
    not appear in test
"""
from typing import List
import numpy as np
import numpy.random as npr
from h_sampler import SmallSampler


class ANMLFactory:

    def __init__(self,
                 data: List[np.ndarray],
                 ident_dat: List[np.ndarray],
                 dat_assign: np.ndarray,
                 twindow_size: int,
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
        assert(len(self.dat_assign) == len(self.data)), "num anml mismatch"

    def generate_split(self, rand_seed: int):
        """Generate 50/50 split across animals
        using provided random seed

        Args:
            rand_seed (int): random seed for rng
        """
        gen = npr.default_rng(rand_seed)
        # sample at top level --> will need to apply dat_assign later
        train_anmls = gen.random(len(self.data)) < self.tr_prob
        # train and test windows ~ can use all overlapping now!
        offset = 0
        train_wins, test_wins = [], []
        for i in range(len(self.data)):
            # skip unavailable
            if not self.dat_assign[i]:
                continue
            # NOTE: overlapping allowed in this case
            win = offset + np.arange(0, len(self.data[i] - self.twindow_size))
            if train_anmls[i]:
                train_wins.append(win)
            else:
                test_wins.append(win)
            offset += len(win)
        full_dat = np.vstack([d for d in self.data])
        full_id_dat = np.vstack([idn for idn in self.ident_dat])
        return [SmallSampler(full_dat, full_id_dat, np.hstack(train_wins), self.twindow_size),
                SmallSampler(full_dat, full_id_dat, np.hstack(test_wins), self.twindow_size)]

