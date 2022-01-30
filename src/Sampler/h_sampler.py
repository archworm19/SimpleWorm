"""
    Hierarchical Samplers

    NOTE: all samplers designs with timeseries in mind
    --> return shape = N x M x Twin
    Twin = timewindow

    TODO/DESIGN ISSUE
    > Where are experiments implemented?
    > > dependent vs. independent samples
    > Does it have to be implemented at sampler level?
    > Or: can sampler be oblivious?
    TODO/IDEA:
    > timeseries vs. id data must be stored separately
    > > cuz of their different nature
    > specify independent / dependent variables
    > using an array or dict!!!
    > Thus: sampler can be agnostic
    > Factory? can be agnostic ~ just pass list/dict thru
    > builders will specify experiments!!

    TODO: move factories to separate file

"""
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
