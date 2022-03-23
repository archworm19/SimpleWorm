""" 
    Thin Sampler
    > Wraps file_reps.FileSets
    > Instantiated with 

"""
import numpy as np
import numpy.random as npr
from Sampler.file_reps import FileSet, map_idx_to_file


class ThinSampler:
    """Small Hierarchical sampler
    Thin = a wrapper around memmaped files (FileSet)"""

    def __init__(self,
                 data_tree: FileSet,
                 window_starts: np.ndarray,
                 window_size: int):
        """Initialize

        Args:
            data_tree (np.ndarray): hierarchical data
                = tree of FileSet objects
            window_starts: pointers into the data_tree
                = specify where accessible samples are located
                can be numpy array of numpy memmap
                elems =
                    [set_idxs..., file_idx, t0]
            window_size (np.ndarray): size of each timewindow
        """
        self.data_tree = data_tree
        self.window_starts = window_starts
        self.window_size = window_size
        self.w_ptrs = np.arange(len(self.window_starts))
        self.next_sample = 0

    def shuffle(self, rng_seed: int = 42):
        """Shuffle 

        Args:
            rng_seed (int): [description]. Defaults to 42.
        """
        dr = npr.default_rng(rng_seed)
        dr.shuffle(self.w_ptrs)

    def epoch_reset(self):
        """Restart sampling for new epoch
        """
        self.next_sample = 0

    def _decode_winstart(self, win_start):
        """elems = [set_idxs..., file_idx, t0]
        
        Returns:
            List[int]: set_idx
            int: file_idx
            int: t0
        """
        return win_start[:-2], win_start[-2], win_start[-1]

    def pull_single_sample(self):
        """Pull next sample

        Args:
            num_samples (int):

        Returns:
            SingleFile: files containing the sample
            int: window start time for beginning of window
                within file

        """
        wptr = self.w_ptrs[self.next_sample]
        cwin = self.window_starts[wptr]
        set_idxs, file_idx, t0 = self._decode_winstart(cwin)
        target_file = map_idx_to_file(self.data_tree, set_idxs, file_idx)
        # incrementing is key!
        self.next_sample += 1
        return target_file, t0

    def pull_samples(self, sample_size: int):
        """
        Returns:
            List[SingleFile]: references to sample files
            List[int]: corresponding t0s    
        """
        file_refs, t0s = [], []
        for _ in sample_size:
            target_file, t0 = self.pull_single_sample()
            file_refs.append(target_file)
            t0s.append(t0)
        return file_refs, t0s
