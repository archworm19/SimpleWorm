"""Build tensorflow dataset: tf.data.Dataset"""
import abc
import numpy as np
import tensorflow as tf
from typing import Dict, List
from Sampler.set_sampling import DataGroup, gather_leaves


# tensorflow transformations


class TFTransform(abc.ABC):

    def transform(self, dat: Dict[str, np.ndarray]) -> Dict[str, tf.Tensor]:
        pass


class TFTransformConstant(TFTransform):
    # constant transform to tensor (and enforce single type)

    def __init__(self, dtype: tf.Dtype):
        self.dtype = dtype

    def transform(self, dat: Dict[str, np.ndarray]) -> Dict[str, tf.Tensor]:
        return {k: tf.constant(dat[k], self.dtype)
                for k in dat}


# dataset builders


def _repack_dicts(dat: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    ret_dict = {}
    for k in dat[0]:
        all_k = [di[k] for di in dat]
        assert(len(all_k) == len(dat)), "source missing key"
        ret_dict[k] = np.concatenate(all_k, axis=0)
    return ret_dict


def build_tfdset_inmemory(sampled_dataset_root: DataGroup,
                          tf_transform: TFTransform,
                          target_columns: List[str],
                          train_mode: bool):
    # returns tf.data.Dataset
    lvs = gather_leaves(sampled_dataset_root)
    dat = [lv.get_numpy_data(target_columns, train_mode) for lv in lvs]
    dat = _repack_dicts(dat)
    dat = tf_transform.transform(dat)
    return tf.data.Dataset.from_tensor_slices(dat)
