"""Sample within a source"""
import abc
import numpy as np
from typing import Dict, List


# In Memory IM interface


class DataSourceIM(abc.ABC):

    def get_numpy_data(self,
                       target_columns: List[str],
                       train_mode: bool) -> Dict[str, np.ndarray]:
        # names numpy arrays
        pass


class DataSourceFull(DataSourceIM):
    # just get the full source

    def __init__(self, dat: Dict[str, np.ndarray]):
        self.dat = dat

    def get_numpy_data(self,
                       target_columns: List[str],
                       train_mode: bool) -> Dict[str, np.ndarray]:
        return {tc: self.dat[tc] for tc in target_columns}


class DataSourceAlternating(DataSourceIM):
    # sources alternating blocks (blocks in terms of 0 axis)

    def __init__(self,
                 dat: Dict[str, np.ndarray],
                 chunk_size: int,
                 chunk0_bit: bool):
        self.dat = dat
        self.chunk_size = chunk_size
        self.chunk0_bit = chunk0_bit

    def _get_indices(self, train_mode: bool, num_samples: int):
        if train_mode:
            use_bit = self.chunk0_bit
        else:
            use_bit = not self.chunk0_bit
        ret_inds = []
        for i in range(0, num_samples, self.chunk_size):
            if use_bit:
                ret_inds.append(np.arange(i, min(i + self.chunk_size, num_samples)))
            use_bit = not use_bit
        return np.concatenate(ret_inds, axis=0)

    def get_numpy_data(self,
                       target_columns: List[str],
                       train_mode: bool) -> Dict[str, np.ndarray]:
        num_samples = np.shape(self.dat[target_columns[0]])[0]
        inds = self._get_indices(train_mode, num_samples)
        return {tc: self.dat[tc][inds] for tc in target_columns}
