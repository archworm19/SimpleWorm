"""Numpy Memmap File Representations"""
import numpy as np
from typing import List
from Sampler.file_sets import FileWrapper


class FileWrapperNP(FileWrapper):

    def __init__(self, idn: int,
                 t_file_name: str, id_file_name: str, t0_sample_file_name: str,
                 t_file_shape: List[int], id_file_shape: List[int], t0_file_shape: List[int],
                 dtypes: str, t0_sample_dtype: str):
        self.idn = idn
        self.t_file_name = t_file_name
        self.id_file_name = id_file_name  # file name for id data
        self.t0_sample_file_name = t0_sample_file_name # file name for t0 samples
        self.t_file_shape = t_file_shape # shape of numpy array in memmap file
        self.id_file_shape = id_file_shape # ""
        self.t0_file_shape = t0_file_shape # number of t0s sampled for the current file
        self.dtypes = dtypes
        self.t0_sample_dtype = t0_sample_dtype

    def clone(self):
        # required by FileWrapper interface
        # cheap operation cuz just filenames and metadata:
        return FileWrapperNP(self.idn,
                      self.t_file_name, self.id_file_name, self.t0_sample_file_name,
                      self.t_file_shape, self.id_file_shape, self.t0_file_shape,
                      self.dtypes, self.t0_sample_dtype)

    def open_file(self):
        """Open numpy memmap file
        Opening file should have very little memory hit
        ... there will be some memory cost to doing operations
        on returned elements via caching

        Args:
            target_file (SingleFile):

        Returns:
            np.ndarray: time data
                memmap file that can be treated like numpy array
                T x d1 x d2 x ...
            np.ndarray: id data
                memmap file
                T x M
            np.ndarray: indices of sampled t0s within file
                len num_samples array
        """
        t_dat = np.memmap(self.t_file_name, dtype=self.dtypes,
                        mode='r+', shape=self.t_file_shape)
        id_dat = np.memmap(self.id_file_name, dtype=self.dtypes,
                        mode='r+', shape=self.id_file_shape)
        t0_samples = np.memmap(self.t0_sample_file_name, dtype=self.t0_sample_dtype,
                            mode='r+', shape=self.t0_file_shape)
        return t_dat, id_dat, t0_samples

    def sample(self, sample_inds: np.ndarray):
        """Sample t0s within a file
        How? reassigns the t0 fields of a given file to point
        to t0_samples ~ handles the creation of memmap file
        NOTE: does NOT alter the original file

        Args:
            t0_samples (np.ndarray): sample indices within file

        Returns:
            FileWrapperNP: copy of file with t0 fields altered
        """
        t0_file_name = self.t0_sample_file_name + "_t0sample"
        t0_dat = np.memmap(t0_file_name, dtype=sample_inds.dtype,
                        mode='w+', shape=np.shape(sample_inds))
        t0_dat[:] = sample_inds[:]
        return FileWrapperNP(self.idn,
                        self.t_file_name,
                        self.id_file_name,
                        t0_file_name,
                        self.t_file_shape,
                        self.id_file_shape,
                        np.shape(sample_inds),
                        self.dtypes,
                        sample_inds.dtype)
    
    def get_samples(self):
        """Get the available samples as index array
        
        Returns:
            np.ndarray: indices of available sampels"""
        _, _, sample_inds = self.open_file()
        return sample_inds


def save_file(file_id: int, file_root: str,
              t_ar: np.ndarray, id_ar: np.ndarray, t0_samples: np.ndarray):
    """Create numpy memmap files from numpy arrays
    --> return SingleFile that points to these files
    NOTE: the created memmaps will be garbage collected
    --> need to reload if want to use

    Args:
        file_id (int):
        file_root (str): file_root that will be incorporated
            into the filenames
        t_ar (np.ndarray): timeseries array
        id_ar (np.ndarray): identity array
        t0_samples (np.ndarray): indices of sampled t0s

    Returns:
        FileWrapperNP
    """
    assert(t_ar.dtype == id_ar.dtype), "arrays must have same type"
    t_file_name = "{0}_{1}_{2}".format(file_root, str(file_id), "t_file.dat")
    id_file_name = "{0}_{1}_{2}".format(file_root, str(file_id), "id_file.dat")
    t0_file_name = "{0}_{1}_{2}".format(file_root, str(file_id), "t0_samples.dat")
    # make the file --> copy data in
    t_dat = np.memmap(t_file_name, dtype=t_ar.dtype,
                      mode='w+', shape=np.shape(t_ar))
    id_dat = np.memmap(id_file_name, dtype=id_ar.dtype,
                       mode='w+', shape=np.shape(id_ar))
    t0_dat = np.memmap(t0_file_name, dtype=t0_samples.dtype,
                       mode='w+', shape=np.shape(t0_samples))
    t_dat[:] = t_ar[:]
    id_dat[:] = id_ar[:]
    t0_dat[:] = t0_samples[:]
    return FileWrapperNP(file_id,
                      t_file_name,
                      id_file_name,
                      t0_file_name,
                      np.shape(t_ar),
                      np.shape(id_ar),
                      np.shape(t0_samples),
                      t_ar.dtype,
                      t0_samples.dtype)
