"""
    Sampling interface
"""
import abc
import numpy as np

class SamplerIface(abc.ABC):

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
            np.ndarray: starting timepoints of each
                timewindow
                len num_samples
        """
        pass

    def epoch_reset(self):
        """Restart sampling for new epoch
        """
        pass

    def shuffle(self, rng_seed: int = 42):
        """Shuffle 

        Args:
            rng_seed (int): [description]. Defaults to 42.
        """
        pass

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
        pass

    def get_sample_size(self):
        pass

    def get_twindow_size(self):
        pass


