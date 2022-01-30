"""Even Split Factory

    Splits within trial
    TODO: currently unused
"""
from typing import List
import numpy as np
import numpy.random as npr
from h_sampler import SmallSampler

class SSFactoryEvenSplit:
    """Factory that builds Small Samplers
        by splitting data evenly in half
    Typical use case:
    > Call this factory each time you want a different split

    Key: set_indices are not in timewindow format
    --> factory will generate timewindows
    set_bools = booleans ~ True when data is available to this set
    """

    def __init__(self,
                 data: List[np.ndarray],
                 ident_dat: List[np.ndarray],
                 set_bools: List[np.ndarray],
                 twindow_size: int,
                 tr_prob = 0.5):
        self.data = data
        self.ident_dat = ident_dat
        self.set_bools = set_bools
        self.twindow_size = twindow_size
        self.tr_prob = tr_prob

    def _find_legal_windows(self, avail_booli: np.ndarray):
        """Find legal windows ~ entire window is available in booli (from set_bools)

        Args:
            avail_booli (np.ndarray): boolean availability array for one data subset
        
        Returns:
            List[int]: legal window starts
        """
        legal_starts = []
        for i in range(0, len(avail_booli), self.twindow_size):
            if np.sum(avail_booli[i:i+self.twindow_size]) > self.twindow_size - 1:
                legal_starts.append(i)
        return legal_starts

    def generate_split(self, rand_seed: int):
        """Generate 50/50 split of available indices
        using provided random seed

        Args:
            rand_seed (int): random seed for rng
        """
        rng_gen = npr.default_rng(rand_seed)
        full_dat, train_wins, test_wins, full_id_dat = [], [], [], []
        offset = 0
        for i in range(len(self.data)):
            # find all legal windows:
            legal_starts_off = self._find_legal_windows(self.set_bools[i])
            # NOTE: VERY IMPORTANT: offset and random start added in here
            legal_starts = np.array(legal_starts_off) + offset
            # split legal windows across the 2 sets:
            tr_assign = rng_gen.random(len(legal_starts)) < self.tr_prob
            subtr_t0s, subtest_t0s = [], []
            for j, ls in enumerate(legal_starts):
                if tr_assign[j]:
                    subtr_t0s.append(ls)
                else:
                    subtest_t0s.append(ls)
            # update offset:
            offset += len(self.set_bools[i])
            # save all data
            full_dat.append(self.data[i])
            train_wins.append(subtr_t0s)
            test_wins.append(subtest_t0s)
            full_id_dat.append(self.ident_dat[i])
        full_dat = np.vstack(full_dat)
        full_id_dat = np.vstack(full_id_dat)
        return [SmallSampler(full_dat, full_id_dat, np.hstack(train_wins), self.twindow_size),
                SmallSampler(full_dat, full_id_dat, np.hstack(test_wins), self.twindow_size)]

