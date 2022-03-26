"""
    File Wrapper Classes
    Handles hierarchical mapping

    NOTE: assumes file follows SingleFile
    ... could easily be an interface

"""
import dataclasses
from typing import List
import numpy as np


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
    t_file_shape: List[int]  # shape of numpy array in memmap file
    id_file_shape: List[int]  # ""
    dtypes: str


@dataclasses.dataclass
class FileSet:
    sub_sets: List  # List[FileSet]
    files: List[SingleFile]


def clone_single_file(old_file: SingleFile):
    # cheap operation cuz just filenames and metadata:
    return SingleFile(old_file.idn, old_file.t_file_name,
                      old_file.id_file_name, old_file.t_file_shape,
                      old_file.id_file_shape, old_file.dtypes)


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
    """
    t_dat = np.memmap(target_file.t_file_name, dtype=target_file.dtypes,
                      mode='r+', shape=target_file.t_file_shape)
    id_dat = np.memmap(target_file.id_file_name, dtype=target_file.dtypes,
                       mode='r+', shape=target_file.id_file_shape)
    return t_dat, id_dat


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
