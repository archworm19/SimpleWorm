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
        self.set_root = set_root
        self.twindow_size = twindow_size
        self.filez = file_reps.get_files(self.set_root)
        # initalize:
        self.t0s, self.fidns, self.total = self._init_file_map(self.filez)
        # singleton formulation
        N = len(self.t0s)
        self.tdat = [None] * N
        self.iddat = [None] * N

    def _init_file_map(self, filez: List[file_reps.SingleFile]):
        """Initialize the internal file map

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
            t_dat, _id_dat = file_reps.open_file(cfile)
            file_idns.append(cfile.idn)
            # increment ct0 to next file
            ct0 += np.shape(t_dat)[0] - self.twindow_size
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
            t_dat, id_dat = file_reps.open_file(self.filez[targ_file_idx])
            self.tdat[targ_file_idx] = t_dat
            self.iddat[targ_file_idx] = id_dat
        # sample in time --> tile t:
        # raw_t = twindow_size x ...
        raw_t = self.tdat[targ_file_idx][t0_offset:self.twindow_size]
        raw_id = self.iddat[targ_file_idx][t0_offset]
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
