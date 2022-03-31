""" 
    Drawer
    > Draws specified raw data
    > Specifies legal data + keeps track of history

    NOTE: just have a time sampler for now
    ... could easily be adapted to an interface

"""
from typing import List
import numpy as np
from Sampler import file_reps


class TDrawer:

    def __init__(self, set_root: file_reps.FileSet,
                 twindow_size: int):
        """Iniitialization

        Args:
            set_root (file_reps.FileSet): root file set object
            twindow_size (int): timewindow size
            t_sample_prob (float, optional): sampling probability
                for tpts within a file. Defaults to 1..
        """
        self.set_root = set_root
        self.twindow_size = twindow_size
        self.filez = file_reps.get_files(self.set_root)
        # initalize:
        self.t0s, self.fidns, self.total = self._init_file_map(self.filez)
        # singleton formulation
        N = len(self.t0s)
        self.tdat = [None] * N
        self.iddat = [None] * N
        self.t0dat = [None] * N

    def _get_useable_t0s(self, t0s: np.ndarray, ar_len: int):
        """Not every t0 in a file rep is useable
        Some specify windows that overhang the end of the dataset
        This file returns a boolean array specifying which can 
        be used given the twindow_size

        Args:
            t0s (np.ndarray): t0 samples for given file
            ar_len (int): array len
                Ex: len(t_dat[i])

        Returns:
            np.ndarray: boolean array of len = len(t0s)
        """

        return t0s <= ar_len - self.twindow_size

    def _init_file_map(self, filez: List[file_reps.SingleFile]):
        """Initialize the internal file map
        NOTE: should close memmap files at end of exe (gc)

        Returns: as a list
            np.ndarray: starting t0 of each file
                Ex: if file0 has 100 legal samples -->
                first 2 elems of t0s = [0, 100, ...]
            List[Any]: ids for each file
            int: total number of legal sampels
        """
        t0s, file_idns = [], []
        ct0 = 0
        for cfile in filez:
            t0s.append(ct0)
            t_dat, _id_dat, t0_samples = file_reps.open_file(cfile)
            file_idns.append(cfile.idn)

            # useable t0_samples = where twindow won't overrun sample
            use_t0s = self._get_useable_t0s(t0_samples, len(t_dat))

            # increment ct0 to next file
            ct0 += int(np.sum(use_t0s))

        return np.array(t0s), file_idns, ct0

    def get_available_samples(self):
        """Get number of available samples
        over all files/whole set

        Returns:
            int: number of available samples
        """
        return self.total
    
    def draw_sample(self, idx: int):
        """Draw a single sample
        NOTE: handles singleton loading
        NOTE: by operating on self.t0s, this
            function implicitly operates only on
            useable t0 samples
        TODO: there's probably a more efficient version of this sampling

        Args:
            idx (int): index across files

        Returns:
            np.ndarray: time data sample
                Twindow_size x ...
            np.ndarray: id data sample
                len M array 
        """
        if idx >= self.total:
            return None
        # figure out file
        # = last file with starting t0 <= idx
        targ_file_idx = np.where(self.t0s <= idx)[0][-1]
        # figure out t0 within file:
        t0_offset = idx - self.t0s[targ_file_idx]
        # singleton handling:
        if self.tdat[targ_file_idx] is None:
            t_dat, id_dat, sample_t0s = file_reps.open_file(self.filez[targ_file_idx])
            self.tdat[targ_file_idx] = t_dat
            self.iddat[targ_file_idx] = id_dat
            self.t0dat[targ_file_idx] = sample_t0s

        # sample in time --> tile t:
        # raw_t = twindow_size x ...
        fin_t0 = self.t0dat[targ_file_idx][t0_offset]
        raw_t = self.tdat[targ_file_idx][fin_t0:fin_t0+self.twindow_size]
        raw_id = self.iddat[targ_file_idx][fin_t0]
        return raw_t, raw_id

    def draw_samples(self, idxs: List[int]):
        """Draw a number of samples
        > just a wrapper on draw_samples

        Args:
            idxs (List[int]): target indices across files

        Returns:
            np.ndarray: time sample
                num_sample x Twindow_size x ...
            np.ndarray: identity sample
                num_sample x M
        """
        t_samp, id_samp = [], []
        for idx in idxs:
            ti, idi = self.draw_sample(idx)
            t_samp.append(ti)
            id_samp.append(idi)
        return np.array(t_samp), np.array(id_samp)
