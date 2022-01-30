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
                 window_size: int,
                 dep_t_mask = np.ndarray,
                 dep_id_mask = np.ndarray,
                 indep_t_mask = np.ndarray,
                 indep_id_mask = np.ndarray):
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
            window_size (np.ndarray): size of each timewindow
            masks (np.ndarray): boolean masks for selection of
                independent/dependent dims for timeseries and
                id data
        """
        self.data = data
        self.ident_dat = ident_dat
        self.window_starts = window_starts
        self.window_size = window_size
        self.dep_t_mask = dep_t_mask
        self.dep_id_mask = dep_id_mask
        self.indep_t_mask = indep_t_mask
        self.indep_id_mask = indep_id_mask

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

    def _fselection(self, ar_sample: np.ndarray, mask: np.ndarray):
        """ar_sample has multiple samples. Mask does not
            account for num_sample natively
            --> tile mask and select ar_sample subset
            KEY: returns a slice into ar_sample"""
        num_sample = np.shape(ar_sample)[0]
        nm_tile = [num_sample] + [1 for z in np.shape(mask)]
        mask_tile = np.tile(mask[None], nm_tile)
        return ar_sample[mask_tile]

    def flatten_samples(self,
                        dat_tseries: np.ndarray,
                        dat_ids: np.ndarray):
        """flatten samples
        Takes in structured timeseries and identity batch data
        > spits it into independent and dependent data
        > flattens both to [num_sample] x m_i

        Args:
            dat_tseries (np.ndarray): num_samples x T x m array
            dat_ids (np.ndarray):

        Returns:
            np.ndarray: independent flattened data
            np.ndarray: dependent flattened data
        """
        sample_size = np.shape(dat_tseries)[0]
        # order data
        raw = [dat_tseries, dat_ids]
        mindraw = [self.indep_t_mask, self.indep_id_mask]
        mdepraw = [self.dep_t_mask, self.dep_id_mask]
        # apply masks
        indep = [self._fselection(raw[i], mindraw[i]) for i in range(len(raw))]
        dep = [self._fselection(raw[i], mdepraw[i]) for i in range(len(raw))]
        # flatten and stack
        for v in [indep, dep]:
            v = np.hstack([np.reshape(v, (sample_size, -1))])
        return indep, dep
    
    # TODO: needs reworking with masks
    def unflatten_samples(self,
                          dat_flat: np.ndarray,
                          indep: bool = True):
        """Reshape flatten data back to original shape
        Inverse operation of flatten_sampels

        Args:
            dat_flat (np.ndarray): N x M array
                where N = number of samples
            indep (bool): whether this is independent (True)
                or dependent sample
        
        Returns:
            np.ndarray: time series data in original
                data shape
            np.ndarray: id data in original data shape
        """
        # TODO: figure out what masks to use
        if indep:
            umasks = [self.indep_t_mask, self.indep_id_mask]
        else:
            umasks = [self.dep_t_mask, self.dep_id_mask]

        num_sample = np.shape(dat_flat)[0]
        parent_shape = [[num_sample, self.window_size] + np.shape(self.data[0])[1:],
                        [num_sample] + np.shape(self.ident_dat[0])[1:]]

        # first: break up between tseries and ids:
        dim0 = int(np.sum(umasks[0]))
        t_dat = dat_flat[:, :dim0]
        id_dat = dat_flat[:, dim0:]
        fdatz = [t_dat, id_dat]

        # TODO: invert the masks
        big_dats = []
        for i in range(len(umasks)):
            stru = np.nan * np.ones(parent_shape[i])
            strusub = self._fselection(stru)
            strusub = fdatz
            big_dats.append(strusub)
        return big_dats[0], big_dats[1]

    def get_sample_size(self):
        return len(self.window_starts)
