"""
    File Wrapper Classes
    Handles hierarchical mapping

    NOTE: assumes file follows SingleFile
    ... could easily be an interface

"""
import dataclasses
from typing import List
import numpy as np
import numpy.random as npr


# NOTE: should maybe just be a generic SingleFile
# --> this would be a single version of it... easy fix if needed
@dataclasses.dataclass
class SingleFile:
    """Assumed shapes
    t_file: T x d1 x d2 x ...
    id_file: T x M
        T = number of time points
    """
    idn: int
    t_file_name: str  # file name for time data
    id_file_name: str  # file name for id data
    t0_sample_file_name: str  # file name for t0 samples
    t_file_shape: List[int]  # shape of numpy array in memmap file
    id_file_shape: List[int]  # ""
    t0_file_shape: List[int]  # number of t0s sampled for the current file
    dtypes: str
    t0_sample_dtype: str


@dataclasses.dataclass
class FileSet:
    sub_sets: List  # List[FileSet]
    files: List[SingleFile]


def clone_single_file(old_file: SingleFile):
    # cheap operation cuz just filenames and metadata:
    return SingleFile(old_file.idn,
                      old_file.t_file_name, old_file.id_file_name, old_file.t0_sample_file_name,
                      old_file.t_file_shape, old_file.id_file_shape, old_file.t0_file_shape,
                      old_file.dtypes, old_file.t0_sample_dtype)


def map_idx_to_file(root_set: FileSet, set_idx: List[int], file_idx: int):
    """map idx representation to a specific file

    Args:
        set_idx (List[int]): indices specifying hierarchy
            index 0 = top-level, etc.
        file_idx (int): file index
    """
    fset = root_set
    # iter thru subsets
    for sidx in set_idx:
        fset = fset.sub_sets[sidx]
    # get file:
    return fset.files[file_idx]


def open_file(target_file: SingleFile):
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
    t_dat = np.memmap(target_file.t_file_name, dtype=target_file.dtypes,
                      mode='r+', shape=target_file.t_file_shape)
    id_dat = np.memmap(target_file.id_file_name, dtype=target_file.dtypes,
                       mode='r+', shape=target_file.id_file_shape)
    t0_samples = np.memmap(target_file.t0_sample_file_name, dtype=target_file.t0_sample_dtype,
                           mode='r+', shape=target_file.t0_file_shape)
    return t_dat, id_dat, t0_samples


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
        SingleFile: 
    """
    assert(t_ar.dtype == id_ar.dtype), "arrays must have same type"
    t_file_name = "{0}_{1}".format(file_root, "t_file.dat")
    id_file_name = "{0}_{1}".format(file_root, "id_file.dat")
    t0_file_name = "{0}_{1}".format(file_root, "t0_samples.dat")
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
    return SingleFile(file_id,
                      t_file_name,
                      id_file_name,
                      t0_file_name,
                      np.shape(t_ar),
                      np.shape(id_ar),
                      np.shape(t0_samples),
                      t_ar.dtype,
                      t0_samples.dtype)


def sample_file(file: SingleFile,
                t0_samples: np.ndarray):
    """Sample t0s within a file
    How? reassigns the t0 fields of a given file to point
    to t0_samples ~ handles the creation of memmap file
    NOTE: does NOT alter the original file

    Args:
        file (SingleFile): file to sample
        t0_samples (np.ndarray): sample indices within file

    Returns:
        SingleFile: copy of file with t0 fields altered
    """
    t0_file_name = file.t0_sample_file_name + "_t0sample"
    t0_dat = np.memmap(t0_file_name, dtype=t0_samples.dtype,
                       mode='w+', shape=np.shape(t0_samples))
    t0_dat[:] = t0_samples[:]
    return SingleFile(file.idn,
                      file.t_file_name,
                      file.id_file_name,
                      t0_file_name,
                      file.t_file_shape,
                      file.id_file_shape,
                      np.shape(t0_samples),
                      file.dtypes,
                      t0_samples.dtype)


def sample_file_subset(file: SingleFile,
                       sample_prob: float,
                       rng: npr.Generator):
    """Sample t0s within a file
    Taking a subset of the current t0s

    Args:
        file (SingleFile): target file
        sample_prob (float): sample probability
        rng (npr.Generator):

    Returns:
        SingleFile: copy of file with t0 fields altered
    """
    _, _, t0_samples = open_file(file)
    rz = rng.random(len(t0_samples)) <= sample_prob
    new_t0s = t0_samples[rz]
    return sample_file(file, new_t0s)


def _get_depths_helper(cset: FileSet,
                       cdepth: int):
    if len(cset.sub_sets) == 0:
        return [cdepth]
    ret_depths = []
    for child in cset.sub_sets:
        ret_depths.extend(_get_depths_helper(child, cdepth+1))
    return ret_depths


def get_depths(root_set: FileSet):
    """Get distance to each leaf of the tree

    Args:
        root_set (FileSet):

    Returns:
        List[int]
    """
    return _get_depths_helper(root_set, 0)


def get_files(cset: FileSet):
    """Get all file leaves

    Args:
        cset (FileSet): current set
            should typically be the root of the set tree

    Returns:
        List[SingleFile]
    """
    if len(cset.sub_sets) == 0:
        return cset.files
    ret_files = []
    for child in cset.sub_sets:
        ret_files.extend(get_files(child))
    return ret_files
