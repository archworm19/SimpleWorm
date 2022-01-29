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
from typing import List
import numpy as np
import numpy.random as npr
import sampler_iface


class SmallSampler(sampler_iface.Sampler):
    """Small Hierarchical sampler
    Small size --> store the whole dataset in memory"""

    def __init__(self,
                 data: np.ndarray,
                 ident_dat: np.ndarray,
                 window_starts: np.ndarray,
                 window_size: int):
        """Initialize HSampler

        Args:
            data (np.ndarray): T x ... array
                where T = for time windows
                data assumed to be timeseries
            ident_dat (np.ndarray) T x q array of
                identity indicators. Identity indicators
                are different fdom data because they are
                assumed to NOT be a timeseries
                Typical example: q = 1 and all entries
                indicate source animal
            window_starts: beginning of legal 
                time windows
        """
        self.data = data
        self.ident_dat = ident_dat
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
        """Pull next [num_samples] samples

        Args:
            num_samples (int):

        Returns:
            np.ndarray: num_samples x T x ... array
                timeseries data
            np.ndarray: num_samples x q array
                identity indicator data
                Ex: what worm is this coming from?
        """
        batch, ident_batch = [], []
        for i in range(self.next_sample, min(self.next_sample + num_samples,
                                             len(self.window_starts))):
            t0 = int(self.window_starts[i])
            batch.append(self.data[t0:t0+self.window_size])
            ident_batch.append(self.ident_dat[t0])
        # update sample ptr:
        self.next_sample += num_samples
        return np.array(batch), np.array(ident_batch)

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
            # pull a starting point
            rt0 = rng_gen.integers(0, self.twindow_size, 1)[0]
            # find all legal windows:
            legal_starts_off = self._find_legal_windows(self.set_bools[i][rt0:])
            # NOTE: VERY IMPORTANT: offset and random start added in here
            legal_starts = legal_starts_off + rt0 + offset
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


def _get_max_dat_len(dat: List[np.ndarray]):
    """Helper function to find maximum data length
    across arrays
    Returns: int
    """
    ml = -1
    for d in dat:
        if len(d) > ml:
            ml = len(d)
    return ml


def _assert_dim_match(dat: List[np.ndarray]):
    """Assert that all dims match after dim0
    for arrays within dat List
    """
    if len(dat) > 1:
        v0 = np.array(np.shape(dat[0])[1:])
        for d in dat[1:]:
            vi = np.array(np.shape(d)[1:])
            assert(np.sum(v0 != vi) < 1), "dim match assertion failure"
          

def build_small_factory(dat: List[np.ndarray],
                        twindow_size: int,
                        block_size: int,
                        rand_seed: int = 42):
    """Build a Small Factory

    dat arrays must be T x ...
    
    Creates 2 identity variables
    > relative window start (divide by max data length)
    > array (animal) identity ~ which array (within List[np.ndarray])
    sample came from

    Size assumption: small ~ no need for memory mapping
    
    > Split into train (2/3) and test (1/3) sets
    > build a factory for train
    > build a sampler for test

    Arguments:
        dat (List[np.ndarray]): list of T x ... arrays
            assumed to be timeseries with time along axis 0
        twindow_size (int): sampling window size
            = Time window size models will act on
        block_size (int): number of contiguous twindows that
            go into training vs. test. This does not
            affect model windows.
        rand_seed (int): rng seed for train vs. test splitting

    Returns:
        SSFactoryEvenSplit: sampler generator for Train sets
        SmallSampler: single sampler for whole of test set
        
    """
    _assert_dim_match(dat)

    # get max data length for length normalization:
    max_dlen = _get_max_dat_len(dat)

    # seeding random number generator makes 
    # process deterministic
    rng_gen = npr.default_rng(rand_seed)

    # tbwin = timewindow size * block size
    # central time unit for train vs. test split
    tbwin = twindow_size * block_size

    train_bools, test_bools = [], []
    ident_dat = []
    # iter thru files ~ top-level
    for i in range(len(dat)):
        boolz = np.ones((len(dat[i]))) < 0  # false by default
        for j in range(0, len(dat[i]), tbwin):
            if rng_gen.random() < .667:
                boolz[j:j+tbwin] = True
        train_bools.append(boolz)
        test_bools.append(np.logical_not(boolz))
        # identity data:
        wid = i * np.ones((len(boolz)))
        trang = np.arange(len(boolz)) / max_dlen
        ident_dat.append(np.vstack((wid, trang)).T)
    TrainFactory = SSFactoryEvenSplit(dat, ident_dat, train_bools, twindow_size, 0.5)
    TestFactory = SSFactoryEvenSplit(dat, ident_dat, test_bools, twindow_size, 1.0)
    return TrainFactory, TestFactory.generate_split(0)[0]


def _search_ident(ident: np.ndarray,
                  query_patrn: np.ndarray):
    """Testing function
    Looking for duplicate identities

    Args:
        ident (np.ndarray): N x m array
        query_patrn (np.ndarray): len m array
    """
    m = np.shape(ident)[1]
    b = query_patrn[None] == ident
    return np.where(np.sum(b, axis=1) >= m)[0]


if __name__ == "__main__":
    # test samplers with fake data
    d1 = 1 * np.ones((300, 3))
    d2 = 2 * np.ones((400, 3))
    train_factory, test_sampler = build_small_factory([d1, d2], 6, 4, 42)
    train_sampler, cross_sampler = train_factory.generate_split(1)
    identz = []
    for v in [train_sampler, cross_sampler, test_sampler]:
        (_, ident) = v.pull_samples(1000)
        identz.append(ident)
    collisions = []
    for i in range(len(identz)):
        idi = identz[i]
        for j in range(np.shape(idi)[0]):
            for k in range(len(identz)):
                if k == i:
                    continue
                idk = identz[k]
                collisions.append(_search_ident(idk, idi[j,:]))
    print('number of available ids: {0}'.format(str(len(collisions))))
    print('collisions: {0}'.format(str(np.array(collisions))))
    
    # Q: 
