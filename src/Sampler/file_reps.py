"""
    File Wrapper Classes
    Handles hierarchical mapping
"""
import dataclasses
from typing import List


# TODO: I still don't really have a design figured out
# ... I guess I would need some sort of conversion system for the sub_processes
# What I'm worried about
# 1. passing in memmap stuff to sub-processes = wierd behavior with caching...
# 2. Reloading the memmap file for every sample will be crazy slow... right?
# ... the point of memmap is that we treat it as array and can take advantage of caching
# ...
# Soln? 
# > Pass in unitialized Sampler to each child process
# > Each child process calls [Sampler].data_initialize
# > > this Sampler loads all interesting data from all files --> operates on these


@dataclasses.dataclass
class SingleFile:
    set_id: int
    t_file_name: str  # file name for time data
    id_file_name: str  # file name for id data
    t_file_shape: List[int]  # shape of numpy array in memmap file
    id_file_shape: List[int]  # ""


class FileSet:

    def __init__(self, sub_sets: List,
                 files: List[SingleFile] = None):
        """File Set can have either child sets (children) or
        a set of files (leaf)

        Args:
            sub_sets (List): List[FileSet]
            files (List[SingleFile]): None unless leaf
        """
        self.sub_sets = sub_sets
        self.files = files
    
    def get_file(self, file_ind: int):
        """Get file if available

        Args:
            file_ind (int): file index

        Returns:
            SingleFile: 
        """
        if self.files is not None:
            return self.files[file_ind]
        return None
    
    def get_subset(self, subset_ind: int):
        return self.sub_sets[subset_ind]


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
        fset = fset.get_subset(sidx)
    # get file:
    return fset.get_file(file_idx)