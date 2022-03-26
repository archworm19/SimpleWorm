""""  
    Flattener and Unflattener
    > Operates on numpy (or memmap stuff)
    > Doesn't handle sampling
    > Operates on a batch (set of samples)
"""
from typing import List
import numpy as np


class Flattener:
    """Handles flattening and unflattening for a
    given experiment"""

    def __init__(self,
                 dep_t_mask: np.ndarray,
                 dep_id_mask: np.ndarray,
                 indep_t_mask: np.ndarray,
                 indep_id_mask: np.ndarray):
        """Initialize Flattener

        Args:
            masks (np.ndarray): boolean masks for selection of
                independent/dependent dims for timeseries and
                id data

                t masks = T x ...
                    T = time window size
                    rest of dims should match base data
                id masks = len M
        """
        self.dep_t_mask = dep_t_mask
        self.dep_id_mask = dep_id_mask
        self.indep_t_mask = indep_t_mask
        self.indep_id_mask = indep_id_mask

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
                NOTE: both can be np memmap arrays too

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
        # figure out what masks to use
        if indep:
            umasks = [self.indep_t_mask, self.indep_id_mask]
        else:
            umasks = [self.dep_t_mask, self.dep_id_mask]

        num_sample = np.shape(dat_flat)[0]

        # approach: 
        # > init nan data in unflattened shape ~ unflat
        # > full flatten everything, including masks
        # > reshape unflat data

        # approach: initialize unflattened data in correct
        # shape with nans = unflat
        # > flatten and mask unflat --> assign to flattened data
        unflat_t = np.nan * np.ones(([num_sample, self.window_size] + list(np.shape(self.data)[1:])))
        unflat_id = np.nan * np.ones(([num_sample] + list(np.shape(self.ident_dat)[1:])))
        unflat_l = [unflat_t, unflat_id]
        split_pt = int(np.sum(umasks[0]))
        flat_l = [dat_flat[:, :split_pt], dat_flat[:, split_pt:]]

        # iter thru data types (tseries, ids)
        fin_unflat = []
        for i, umask in enumerate(umasks):

            # case where there are no useable variables
            if np.sum(umask) < .5:
                fin_unflat.append(unflat_l[i])
                continue

            # reshape ~ completely flatten
            # 1. unflat data -> funf
            # 2. mask -> fmask
            # 3. flat data -> ff
            og_shape = np.shape(unflat_l[i])
            funf = np.reshape(unflat_l[i], (-1,))
            fmask = np.reshape(self._fselection(num_sample, umask), (-1,))
            ff = np.reshape(flat_l[i], (-1,))
            funf[fmask] = ff
            fin_unflat.append(np.reshape(funf, og_shape))
        return fin_unflat[0], fin_unflat[1]
