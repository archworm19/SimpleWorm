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

    def __init__(self, data: np.ndarray, set_indices: np.ndarray):
        """Initialize HSampler

        Args:
            data (np.ndarray): N x ... array of data
            indices (np.ndarray): array of ptrs to data (< len N)
                to data
        """
        self.data = data
        self.set_indices = set_indices
        self.next_sample = 0

    def shuffle(self, rng_seed: int = 42):
        """Shuffle 

        Args:
            rng_seed (int): [description]. Defaults to 42.
        """
        dr = npr.default_rng(rng_seed)
        dr.shuffle(self.set_indices)

    def epoch_reset(self):
        """Restart sampling for new epoch
        """
        self.next_sample = 0

    def pull_samples(self, num_samples: int):
        """Pull next num_samples samples

        Args:
            num_samples (int):

        Returns:
            np.ndarray: next set of data ...
        """
        sel_inds = self.set_indices[self.next_sample:self.next_sample+num_samples]
        self.next_sample += num_samples
        return self.data[sel_inds]

    def get_data_shape(self):
        return np.shape(self.data)


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
                 twindow_size: int):
        self.data = data
        self.set_bools = set_bools
        self.twindow_size = twindow_size

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
        for i in range(len(self.data)):
            # pull a starting point
            rt0 = rng_gen.integers(0, twindow_size, 1)[0]
            # find all legal windows:
            legal_starts_off = self._find_legal_windows(self.set_bools[i][rt0:])
            legal_starts = legal_starts_off + rt0
            # split legal windows across the 2 sets:
            # TODO: finish this!




def build_from_wormwindows(npz_dat: Dict,
                           twindow_size: int,
                           train_percentage: float,
                           rand_seed: int = 42):
    """build_from_wormwindows
    npz_dat assumption:
    > each Dict value is a numpy array
    AND this is the top level of hierarchy
    > Each Dict is a T x z array where T
    is the number of timepoints

    Size assumption: small ~ no need for memory mapping
    
    Procedure:
    > Breaks up into non-overalapping time windows

    Note: for out-of-bootstrapping --> call this func
        for each out-of-boot sample

    Args:
        npz_dat (Dict): Dict where values are
            numpy arrays
    
    Returns:
        SmallHSampler: Training set
        SmallHSampler: Test Set
        
    """
    # seeding random number generator makes 
    # process deterministic
    rng_gen = npr.default_rng(rand_seed)
    full_dat, train_inds, test_inds = [], [], []
    # iter thru files ~ top-level
    kz = list(npz_dat.keys())
    offset = 0
    for k in kz:
        # random start of first window:
        rt0 = rng_gen.integers(0, twindow_size, 1)[0]
        # update groups:
        dat_i = npz_dat[k]
        ctrain_inds, ctest_inds = [], []
        for j in range(rt0, len(dat_i), twindow_size):
            if rng_gen.random() < train_percentage:
                ctrain_inds.extend([offset + z for z in range(j, j+twindow_size)])
            else:
                ctest_inds.extend([offset + z for z in range(j, j+twindow_size)])
        # save train/test
        train_inds.append(ctrain_inds)
        test_inds.append(ctest_inds)
        # save data:
        full_dat.append(npz_dat[k])
        # update offset for set completion:
        offset += np.shape(npz_dat[k])[0]
    Tr = SmallHSampler(np.vstack(full_dat), np.hstack(train_inds))
    Te = SmallHSampler(np.vstack(full_dat), np.hstack(test_inds))
    return Tr, Te
