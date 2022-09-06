"""Sample within a source"""
import abc
import numpy as np
from typing import Dict


class DataSource(abc.ABC):

    def get_numpy_data(self) -> Dict[str, np.ndarray]:
        # names numpy arrays
        pass


class DataSourceInMem(DataSource):

    def __init__(self, dat: Dict[str, np.ndarray]):
        self.dat = dat

    def get_numpy_data(self) -> Dict[str, np.ndarray]:
        return self.dat
