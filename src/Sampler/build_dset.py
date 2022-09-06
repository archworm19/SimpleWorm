"""Build tensorflow dataset: tf.data.Dataset"""
import abc
import numpy as np
import tensorflow as tf
from typing import Dict
from Sampler.set_sampling import DataGroup
from Sampler.source_sampling import SourceSampler


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


def build_tfdset_inmemory(sampled_dataset_root: DataGroup,
                          src_sampler: SourceSampler,
                          tf_transform: TFTransform):
    # TODO
