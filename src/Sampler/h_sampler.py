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
                 window_size: int,
                 dep_t_mask: np.ndarray,
                 dep_id_mask: np.ndarray,
                 indep_t_mask: np.ndarray,
                 indep_id_mask: np.ndarray):
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

    def _fselection(self, num_sample: int, mask: np.ndarray):
        """ar_sample has multiple samples. Mask does not
            account for num_sample natively
            --> tile mask for mult samples
            --> flatten
            KEY: returns reshaped mask"""
        nm_tile = [num_sample] + [1 for _ in list(np.shape(mask))]
        mask_tile = np.tile(mask[None], nm_tile)
        return np.reshape(mask_tile, (num_sample, -1))

    def _flatten_helper(self,
                        sample_size: int,
                        dats: List[np.ndarray],
                        masks: List[np.ndarray]):
        # TODO: docstring
        # flatten data + apply flattened masks
        fdat_l = []
        for i, d in enumerate(dats):
            dfla = np.reshape(d, (sample_size, -1))
            mfla = self._fselection(sample_size, masks[i])
            vre = np.reshape(dfla[mfla], (sample_size, -1))
            fdat_l.append(vre)
        return np.hstack(fdat_l)

    def flatten_samples(self,
                        dat_tseries: np.ndarray,
                        dat_ids: np.ndarray):
        """flatten samples
        Takes in structured timeseries and identity batch data
        > spits it into independent and dependent data
        > flattens both to [num_sample] x m_i

        Args:
            dat_tseries (np.ndarray): num_samples x T x ... array
            dat_ids (np.ndarray): num_sampels x M array

        Returns:
            np.ndarray: independent flattened data
            np.ndarray: dependent flattened data
        """
        sample_size = np.shape(dat_tseries)[0]
        # order data
        raw = [dat_tseries, dat_ids, dat_tseries, dat_ids]
        maskz = [self.indep_t_mask, self.indep_id_mask,
                 self.dep_t_mask, self.dep_id_mask]

        indep2 = self._flatten_helper(sample_size,
                                      raw[:2],
                                      maskz[:2])
        dep2 = self._flatten_helper(sample_size,
                                      raw[2:],
                                      maskz[2:])
        return indep2, dep2
    
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

        # approach: initialize unflattened data in correct
        # shape with nans = unflat
        # > flatten and mask unflat --> assign to flattened data
        unflat_t = np.nan * np.ones(([num_sample, self.window_size] + np.shape(self.data[0])[1:]))
        unflat_id = np.nan * np.ones(([num_sample] + np.shape(self.ident_dat[0])[1:]))
        unflat_l = [unflat_t, unflat_id]

        for i, uma in enumerate(umasks):
            # reshape/flatten mask --> num_sample x m
            mask2 = self._fselection(num_sample, uma)
            mask_len = np.shape(mask2)[1]
            funf = np.reshape(unflat_l[i], (num_sample, -1))
            funf[mask2] = dat_flat[:,:mask_len]

        return unflat_t, unflat_id

    def get_sample_size(self):
        return len(self.window_starts)
