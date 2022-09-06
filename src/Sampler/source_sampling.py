"""Sample within a source"""
import abc
import numpy as np
from typing import Dict, List


class SourceSampler(abc.ABC):

    def source_sample(self,
                      src_dat: Dict[str, np.ndarray],
                      train_mode: bool,
                      set_path: List[int]) -> np.ndarray:
        """sample from a source

        Args:
            src_dat (Dict[str, np.ndarray]): source data
            train_mode (bool): if True --> train set
                TODO: might want >2 sets
            set_path (List[int]): path in the data group/set
                tree to this data source ~ uniquely specifies data source

        Returns:
            np.ndarray: indices to sample
        """
        pass


class SamplerAll(SourceSampler):

    def __init__(self):
        self.build = True

    def source_sample(self,
                      src_dat: Dict[str, np.ndarray],
                      train_mode: bool,
                      set_path: List[int]) -> np.ndarray:
        return np.arange(np.shape(src_dat)[0])


class SamplerAlternate(SourceSampler):

    def __init__(self, chunk_size: int, chunk_0bit: bool):
        self.chunk_size = chunk_size
        self.chunk_0bit = chunk_0bit

    def source_sample(self,
                      src_dat: Dict[str, np.ndarray],
                      train_mode: bool,
                      set_path: List[int]) -> np.ndarray:
        if train_mode:
            use_bit = self.chunk_0bit
        else:
            use_bit = not self.chunk_0bit
        N = np.shape(src_dat)[0]
        ret_inds = []
        for i in range(0, N, self.chunk_size):
            if use_bit:
                ret_inds.append(np.arange(i, min(i+self.chunk_size, N)))
            use_bit = not use_bit
        return np.concatenate(ret_inds, axis=0)
