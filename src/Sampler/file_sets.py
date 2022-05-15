"""Store sets of files in a tree
    Agnostic to underlying file implementations"""
import abc
import dataclasses
import numpy as np
import numpy.random as npr
from typing import List, Tuple


class FileWrapper(abc.ABC):
    """Wraps a single file
    Allows other set operations to be agnostic towards
    file type"""
    def clone(self):
        """copy file info into new object
        Underlying data is NOT copied (shallow)"""
        pass

    def get_samples(self):
        """Get the available samples as index array
        
        Returns:
            np.ndarray: indices of available sampels
        """
        pass

    def get_data_len(self):
        """Get total length of the data 
            = number of samples available before any sampling done
        
        Returns:
            int: data length
        """
        pass

    def sample(self, sample_inds: np.ndarray):
        """draw samples = restrict access to only allow sampling
        from sample_inds

        Returns:
            copy of FileWrapper ~ underlying data should not change
                Except for samples rep
        """
        pass

    def check_nan(self, locations: Tuple[Tuple[int]]):
        """Check if there are any nans in specified locations

        Args:
            locations (Tuple[Tuple[int]]): locations to check
                specified in numpy indexing format = 
                0th tuple for 0th axis, 1st tuple for 1st axis, etc.

        Returns:
            bool: true if any nans
        """
        pass


@dataclasses.dataclass
class FileSet:
    sub_sets: List  # List[FileSet]
    files: List[FileWrapper]


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


def sample_file_subset(file_wrapper: FileWrapper,
                       sample_prob: float,
                       rng: npr.Generator):
    """Sample t0s within a file
    Taking a subset of the current t0s

    Args:
        file_wrapper (FileWrapper): target file
        sample_prob (float): sample probability
        rng (npr.Generator):

    Returns:
        FileWrapper: copy of file with t0 fields altered
    """
    t0_samples = file_wrapper.get_samples()
    rz = rng.random(len(t0_samples)) <= sample_prob
    new_t0s = t0_samples[rz]
    return file_wrapper.sample(new_t0s)


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


def _get_files(cset: FileSet,
               cpath: List[int]):
    # helper func
    if len(cset.sub_sets) == 0:
        return [cpath], [cset.files]
    ret_paths, ret_subs = [], []
    for i, sub_set in enumerate(cset.sub_sets):
        rpath, rsubs = _get_files(sub_set,
                                  cpath + [i])
        ret_paths.extend(rpath)
        ret_subs.extend(rsubs)
    return ret_paths, ret_subs


def get_files_struct(cset: FileSet):
    """Get all file leaves in structured format

    Args:
        cset (FileSet): current set
            should typically be the root of the set tree

    Returns:
        List[List[int]]: paths
        List[List[FileWrapper]]: filez
    """
    return _get_files(cset, [])


def get_files(cset: FileSet):
    """Get files in flattened format

    Args:
        cset (FileSet): current set
            should typically be the root of the set tree
    
    Returns:
        List[FileWrapper]: filez
    """
    _, struct_files = get_files_struct(cset)
    filez = []
    for fs in struct_files:
        filez.extend(fs)
    return filez


def _get_num_sets_lvl(cset: FileSet, lvl: int,
                      cpath: List[int]):
    # helper func
    if lvl == 0:
        return [cpath], [len(cset.sub_sets)]
    ret_paths, ret_subs = [], []
    for i, sub_set in enumerate(cset.sub_sets):
        rpath, rsubs = _get_num_sets_lvl(sub_set, lvl-1,
                                         cpath + [i])
        ret_paths.extend(rpath)
        ret_subs.extend(rsubs)
    return ret_paths, ret_subs


def get_num_sets_lvl(cset: FileSet, lvl: int):
    """Get the number of subsets at the given level

    Args:
        cset (FileSet): root of set sys
        lvl (int): current set
            should typically be the root of the set tree
        
    Returns:
        List[List[int]]: path representation
        List[int]: number of substs for each set at
            the given level
            Corresponds to path rep
    """
    return _get_num_sets_lvl(cset, lvl, [])
