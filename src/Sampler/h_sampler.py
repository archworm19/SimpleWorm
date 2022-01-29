"""
    Hierarchical Samplers

    NOTE: all samplers designs with timeseries in mind
    --> return shape = N x M x Twin
    Twin = timewindow

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

    # TODO: mappings; flattening and unflattening

    def flatten_samples(self,
                        dat_tseries: np.ndarray,
                        dat_ids: np.ndarray):
        """flatten samples
        Takes in structured timeseries and identity data
        --> flattens it to N x M array

        Args:
            dat_tseries (np.ndarray): num_samples x T x ... array
            dat_ids (np.ndarray):
        """
        sample_size = np.shape(dat_tseries)[0]
        return np.hstack((np.reshape(dat_tseries, (sample_size, -1)),
                          dat_ids))
    
    def unflatten_samples(self, dat_flat: np.ndarray):
        """Reshape flatten data back to original shape
        Inverse operation of flatten_sampels

        Args:
            dat_flat (np.ndarray): N x M array
                where N = number of samples
        
        Returns:
            np.ndarray: time series data in original
                data shape
            np.ndarray: id data in original data shape
        """
        # first: break up between tseries and ids:
        t_shape = [self.window_size] + list(np.shape(self.data)[1:])
        num_dim_t = int(np.prod(t_shape))
        t_dat = dat_flat[:, :num_dim_t]
        id_dat = dat_flat[:, num_dim_t:]
        # reshape time series data to get back to og size
        re_t_dat = np.reshape(t_dat, [-1] + t_shape)
        return re_t_dat, id_dat

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


# TODO: factory for animal splitting
# much simpler
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


# TODO: anml factory
def build_anml_factory(dat: List[np.ndarray],
                        twindow_size: int,
                        rand_seed: int = 42):
    """animal factory
    break up train-test by animals
    Assumes: each array in dat is for different animal
    Assumes: 2/3 train - test split (makes sense assuming 
        cross-validation in train)

    Args:
        dat (List[np.ndarray]): animal data
        twindow_size (int): sampling twindow size 
            used by sampler
        rand_seed (int, optional): Defaults to 42.
    """
    # get max data length for length normalization:
    max_dlen = _get_max_dat_len(dat)

    gen = npr.default_rng(rand_seed)
    train_anmls = gen.random(len(dat)) < 0.667

    # TODO: build identity data:
    ident_dat = []
    for i, d in enumerate(dat):
        ids = i * np.ones((len(d)))
        tz = np.arange(len(d)) / max_dlen

    train_factory = ANMLFactory(dat,
                                ident_dat,
                                train_anmls,
                                twindow_size)
    test_factory = ANMLFactory(dat,
                                ident_dat,
                                np.logical_not(train_anmls),
                                twindow_size)
    return train_factory, test_factory


def build_small_factory(dat: List[np.ndarray],
                        twindow_size: int,
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

    train_bools, test_bools = [], []
    ident_dat = []
    # iter thru files ~ top-level
    for i in range(len(dat)):
        # pull rt0 here ~ random offsets
        rt0 = rng_gen.integers(twindow_size)
        boolz = np.ones((len(dat[i]))) < 0  # false by default
        for j in range(rt0, len(dat[i]), twindow_size):
            if rng_gen.random() < .667:
                boolz[j:j+twindow_size] = True
        train_bools.append(boolz)
        test_bools.append(np.logical_not(boolz))
        # identity data:
        wid = i * np.ones((len(boolz)))
        trang = (rt0 + np.arange(len(boolz))) / max_dlen
        ident_dat.append(np.vstack((wid, trang)).T)
    TrainFactory = SSFactoryEvenSplit(dat, ident_dat, train_bools, twindow_size, 0.5)
    TestFactory = SSFactoryEvenSplit(dat, ident_dat, test_bools, twindow_size, 1.0)
    return TrainFactory, TestFactory.generate_split(0)[0]


def _color_idents(dat_ids: List[np.ndarray],
                  t_ids: List[np.ndarray]):
    """Testing function to ensure no sampling
        overlap
    Assumes: t_id entries are normalized
    """
    # compress all entries to single dim:
    newts, colours = [], []
    for i in range(len(dat_ids)):
        newts.append(dat_ids[i] + t_ids[i])
        colours.append(i * np.ones((len(newts[-1]))))
    newts = np.hstack(newts)
    sinds = np.argsort(newts)
    newts = newts[sinds]
    colours = np.hstack(colours)[sinds]
    # collision check:
    assert(len(np.unique(newts)) == len(newts)), "collision failure"
    colour_ar = np.vstack((newts, colours)).T
    print(colour_ar)
    print(np.shape(colour_ar))


if __name__ == "__main__":
    # test samplers with fake data
    d1 = 1 * np.ones((300, 3))
    d2 = 2 * np.ones((400, 3))
    train_factory, test_sampler = build_small_factory([d1, d2], 6, 42)
    train_sampler, cross_sampler = train_factory.generate_split(1)
    idz, tz = [], []
    for v in [train_sampler, cross_sampler, test_sampler]:
        (dtseries, ident) = v.pull_samples(1000)
        print(ident)
        idz.append(ident[:, 0])
        tz.append(ident[:, 1])
        # TESTING: flattening and unflattening
        flatz = v.flatten_samples(dtseries, ident)
        udtseries, uident = v.unflatten_samples(flatz)
        assert(np.sum(dtseries != udtseries) < 1), "unflatten check1"
        assert(np.sum(ident != uident) < 1), "unflatten check1"
    _color_idents(idz, tz)
