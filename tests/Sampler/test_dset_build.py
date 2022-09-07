"""Let's build some datasets"""
import numpy as np
import tensorflow as tf
from Sampler.build_dset import build_tfdset_inmemory, TFTransformConstant
from Sampler.data_source import DataSourceAlternating
from Sampler.set_sampling import DataGroup


def _get_fake_data():
    A = np.array([[1, 1] for _ in range(10)])
    DS1 = DataSourceAlternating({"0": np.vstack([A] * 5),
                                "1": np.vstack([A * 2] * 5)},
                                10, True)
    DS2 = DataSourceAlternating({"0": np.vstack([A * 3] * 5),
                                "1": np.vstack([A * 4] * 5)},
                                10, True)
    DS3 = DataSourceAlternating({"0": np.vstack([A * 5] * 5),
                                "1": np.vstack([A * 6] * 5)},
                                10, True)
    DS4 = DataSourceAlternating({"0": np.vstack([A * 7] * 5),
                                "1": np.vstack([A * 8] * 5)},
                                10, True)
    DG1 = DataGroup([DS1, DS2], [])
    DG2 = DataGroup([DS3, DS4], [])
    DG = DataGroup(None, [DG1, DG2])
    return DG


def test_inmem_builder():
    DG = _get_fake_data()
    dset_train = build_tfdset_inmemory(DG,
                                       TFTransformConstant(tf.float32),
                                       ["0", "1"],
                                       True)
    dbatch = dset_train.batch(30)

    cnt = 1
    for v in dbatch:
        assert(tf.reduce_all(v["0"] == cnt * tf.ones([30,2])).numpy())
        assert(tf.reduce_all(v["1"] == (cnt + 1) * tf.ones([30,2])).numpy())
        cnt += 2


if __name__ == "__main__":
    test_inmem_builder()
