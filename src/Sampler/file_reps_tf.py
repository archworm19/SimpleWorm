"""Tensorflow file representations ~ implement FileWrapper interface for tf"""
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple
from Sampler.file_sets import FileWrapper
from Sampler.utils.tfrecords_utils import open_tfr, convert_tfdset_numpy

class FileWrapperTF(FileWrapper):

    def __init__(self, file_name: str, dtype_map: Dict,
                    sample_inds: np.ndarray = None):
        """Open tf records files --> convert to tfrecords dataset

        Args:
            file_name (str): name of tfrecords file
            dtype_map (Dict): mapping from variable names to variable dtypes.
                variable names assumed to match variable names
                used in saving the tfrecords files (the keys in np_map)
            sample_inds (np.ndarray): optional initial sampling indices
                if not set, all samples will be set to useable

        Returns:
            tf.MapDataset: the dataset made up of tf tensors
        """
        self.file_name = file_name
        self.dtype_map = dtype_map
        dset = open_tfr([self.file_name], self.dtype_map)
        v = [1 for _ in dset]
        self.T = len(v)
        if sample_inds is None:
            # set all t0s to useable
            self.t0_inds = np.arange(v)
        else:
            self.t0_inds = sample_inds

    def clone(self):
        """copy file info into new object
        Underlying data is NOT copied (shallow)
        
        Returns:
            FileWrapperTF
        """
        return FileWrapperTF(self.file_name, self.dtype_map)

    def get_samples(self):
        """Get the available samples as index array
        
        Returns:
            np.ndarray: indices of available samples
        """
        return self.t0_inds

    def get_data_len(self):
        """Get total length of the data 
            = number of samples available before any sampling done
        
        Returns:
            int: data length
        """
        return self.T

    def sample(self, sample_inds: np.ndarray):
        """draw samples = restrict access to only allow sampling
        from sample_inds

        Returns:
            copy of FileWrapper ~ underlying data should not change
                Except for samples rep
        """
        return FileWrapperTF(self.file_name, self.dtype_map, sample_inds)

    def check_nan(self, locations: Tuple[Tuple[int]],
                    target_tensor: int):
        """Check if there are any nans in specified locations

        Args:
            locations (Tuple[Tuple[int]]): locations to check
                specified in numpy indexing format = 
                0th tuple for 0th axis, 1st tuple for 1st axis, etc.

        Returns:
            bool: true if any nans
        """
        dset = open_tfr([self.file_name], self.dtype_map)
        # convert to numpy:
        np_dset = convert_tfdset_numpy(dset, target_tensor)
        return np.any(np.isnan(np_dset[locations]))
