"""
    Hierarchical Samplers

    NOTE: all samplers designs with timeseries in mind
    --> return shape = N x M x Twin
    Twin = timewindow

    # TODO: better design
    # Wut's the design?
    # > Builder
    # > > Generate sets from hierarchical data
    # > > > Sets = indices into data numpy array
    # > Sampler
    # > > one sampler for each set
    # > > houses data and indices into array
    # > > keeps track of exhausted
"""
from typing import Dict, List
import numpy as np
import numpy.random as npr
import sampler_iface


class SmallSampler(sampler_iface.Sampler):
    """Small Hierarchical sampler
    Small size --> store the whole dataset in memory"""

    def __init__(self,
                 data: np.ndarray,
                 window_starts: np.ndarray,
                 window_size: int):
        """Initialize HSampler

        Args:
            data (np.ndarray): T x ... array
                where T = for time windows
            window_starts: beginning of legal 
                time windows
        """
        self.data = data
        self.window_starts = window_starts
        self.window_size = window_size
        self.next_sample = 0

    def shuffle(self, rng_seed: int = 42):
        """Shuffle 

        Args:
            rng_seed (int): [description]. Defaults to 42.
        """
        dr = npr.default_rng(rng_seed)
        dr.shuffle(self.window_starts)

    def epoch_reset(self):
        """Restart sampling for new epoch
        """
        self.next_sample = 0

    def pull_samples(self, num_samples: int):
        """Pull next num_samples samples

        Args:
            num_samples (int):

        Returns:
            np.ndarray: num_samples x T x ... array
        """
        batch = []
        for i in range(self.next_sample, self.next_sample + num_samples):
            t0 = int(self.window_starts[i])
            batch.append(self.data[t0:t0+self.window_size])
        return np.array(batch)

    def get_sample_size(self):
        return len(self.window_starts)


# TODO: factory interface!
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
                 set_bools: List[np.ndarray],
                 twindow_size: int,
                 tr_prob = 0.5):
        self.data = data
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
            if np.sum(avail_booli[i:i+self.twindow_size]) < 0.5:
                legal_starts.append(i)
        return legal_starts

    def generate_split(self, rand_seed: int):
        """Generate 50/50 split of available indices
        using provided random seed

        Args:
            rand_seed (int): random seed for rng
        """
        rng_gen = npr.default_rng(rand_seed)
        full_dat, train_wins, test_wins = [], [], []
        offset = 0
        for i in range(len(self.data)):
            # pull a starting point
            rt0 = rng_gen.integers(0, self.twindow_size, 1)[0]
            # find all legal windows:
            legal_starts_off = self._find_legal_windows(self.set_bools[i][rt0:])
            # NOTE: VERY IMPORTANT: offset and random start added in here
            legal_starts = legal_starts_off + rt0 + offset
            # split legal windows across the 2 sets:
            tr_assign = rng_gen.random(len(legal_starts)) < self.tr_prob
            subtr_t0s, subtest_t0s = [], []
            for i, ls in enumerate(legal_starts):
                if tr_assign:
                    subtr_t0s.append(ls)
                else:
                    subtest_t0s.append(ls)
            # update offset:
            offset += len(self.set_bools[i])
            # save all data
            full_dat.append(self.data[i])
            train_wins.append(subtr_t0s)
            test_wins.append(subtest_t0s)
        full_dat = np.vstack(full_dat)
        return [SmallSampler(full_dat, np.hstack(train_wins), self.twindow_size),
                SmallSampler(full_dat, np.hstack(test_wins), self.twindow_size)]

            

def build_small_factory(dat: List[np.ndarray],
                           twindow_size: int,
                           rand_seed: int = 42):
    """Build a Small Factory

    dat arrays must be T x ...

    Size assumption: small ~ no need for memory mapping
    
    > Split into train (2/3) and test (1/3) sets
    > build a factory for train
    > build a sampler for test

    Returns:
        SSFactoryEvenSplit: sampler generator for Train sets
        SmallSampler: single sampler for whole of test set
        
    """
    # seeding random number generator makes 
    # process deterministic
    rng_gen = npr.default_rng(rand_seed)

    train_bools, test_bools = [], []
    # iter thru files ~ top-level
    for i in range(len(dat)):
        boolz = np.ones((len(dat[i]))) < 0  # false by default
        for j in range(0, len(dat[i]), twindow_size):
            if rng_gen.random() < .667:
                boolz[j:j+twindow_size] = True
        train_bools.append(boolz)
        test_bools.append(np.logical_not(boolz))
    TrainFactory = SSFactoryEvenSplit(dat, train_bools, twindow_size, 0.5)
    TestFactory = SSFactoryEvenSplit(dat, test_bools, twindow_size, 1.0)
    return TrainFactory, TestFactory.generate_split(0)[0]
