"""Tensorflow Utils
"""
from typing import List
import numpy as np
from Sampler.file_sets import FileSet
from Sampler.file_reps_tf import FileWrapperTF
from Sampler.utils.tfrecords_utils import write_numpy_to_tfr


def build_file_wrapper(x: List[np.ndarray], y: np.ndarray, data_weight: np.ndarray,
                        fn: str):
    """Build a tfrecords file for a single animal
    > Shallow wrapper on write_numpy_to_tfr
    > 0th axis of all arrays must be T (must match)

    Args:
        x (List[np.ndarray]): input data 
        y (np.ndarray): target data
        data_weights (np.ndarray): weights for each sample, model combo
        fn (str): tfrecords file name

    Returns
        FileWrapperTF: tensorflow file wrapper
    """
    # check 0th axis
    T = len(y)
    for xi in (x + [data_weight]):
        assert(len(xi) == T), "0th axis must be time axis for all arrays"
    
    # make the tf records file
    np_map = {"x_{0}".format(i): x[i] for i in range(len(x))}
    np_map["y"] = y
    np_map["data_weight"] = data_weight
    write_numpy_to_tfr(fn, np_map)

    # make the file wrapper:
    dtype_map = {k: np_map[k].dtype for k in np_map}
    return FileWrapperTF(fn, dtype_map)


def build_file_set(x: List[List[np.ndarray]], y: List[np.ndarray], data_weight: List[np.ndarray],
                            base_fn: str):
    """Build a file set of tensorflow filewrappers

    Args:
        x (List[List[np.ndarray]]): input data 
        y (List[np.ndarray]): target data
        data_weights (List[np.ndarray]): weights for each sample, model combo
        base_fn (str): base filename --> used to generate filenames for each animal
    """
    for v in [x, y, data_weight]:
        assert(len(v) == len(x)), "mismatch in the number of animals"

    file_wraps = []
    for i in range(len(x)):
        fn = "{0}_{1}".format(base_fn, str(i))
        fw = build_file_wrapper(x[i], y[i], data_weight[i], fn)
        file_wraps.append(fw)
    return FileSet([], file_wraps)


def combine_file_sets(file_sets: List[FileSet]):
    """Create a new FileSet with file_sets as children

    Args:
        file_sets (List[FileSet]): 
    """
    return FileSet(file_sets, [])
