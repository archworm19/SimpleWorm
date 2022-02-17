"""Input data class"""
from dataclasses import dataclass
import numpy as np

@dataclass
class TrialTimeData:
    """Time series data for a single trial"""

    # time series data
    # shape = T x ...
    time_data: np.ndarray

    # window identity data
    # shape = T x M array
    win_id_data: np.ndarray

