"""Unit Tests for Dataset Operations"""
import tensorflow as tf
import numpy as np
from RunScripts.dataset_ops import (set_nans_to_val, sample_field_conditional,
                                    split_by_value, get_anml_windows)


def _build_small_dataset():
    v1 = np.array([[1., np.nan],
                   [3., 4.]])
    v2 = np.array([5., 6.])
    d = {"v1": v1, "v2": v2}
    dset = tf.data.Dataset.from_tensor_slices(d)
    return dset


def _build_anml_dataset():
    # more useful for windowing experiments
    v1 = np.array([0, 1, 2, 3, 4, 5, 6])
    anml = np.array([0, 0, 0, 0, 1, 1, 1])
    d = {"v1": v1, "anml": anml}
    return tf.data.Dataset.from_tensor_slices(d)


def test_nan_reset():
    dset = _build_small_dataset()
    # testing nan resetting
    dset_nonan = set_nans_to_val(dset, "v1", 2.)
    target = [tf.constant([1., 2.], tf.float64),
              tf.constant([3., 4.], tf.float64)]
    for i, v in enumerate(dset_nonan):
        assert tf.math.reduce_all(v["v1"] == target[i])


def test_field_cond_sample():
    dset = _build_small_dataset()
    # test element selection:
    # > dataset output should be constant for single salt
    # > can differ across salts
    for salt in range(10):
        # try sampling multiple times --> should get same results within salt
        dset_filter1 = sample_field_conditional(dset, "v2", 2, salt)
        dset_filter2 = sample_field_conditional(dset, "v2", 2, salt)
        for v in zip(dset_filter1, dset_filter2):
            assert tf.math.reduce_all(v[0]["v2"] == v[1]["v2"])


def test_split_by_value():
    dset = _build_small_dataset()
    # test tradeoff:
    # factor = 1. --> should be able to separate for some salt
    split = False
    for salt in range(10):
        dset_trade = split_by_value(dset, "v2", 1., salt=salt)
        bitz = tf.concat([v["BIT"] for v in dset_trade], axis=0)
        if tf.math.reduce_any(bitz != bitz[0]):
            split = True
            break
    assert split
    # factor = 0.1 + offset at 1. --> should always group together
    # NOTE: should also test that you can get either bit
    const = True
    save_bitz = []
    for salt in range(10):
        dset_trade = split_by_value(dset, "v2", 0.1, offset=1., salt=salt)
        bitz = tf.concat([v["BIT"] for v in dset_trade], axis=0)
        const = tf.math.reduce_all(bitz == bitz[0]).numpy()
        save_bitz.append(bitz[0])
    assert const
    set_bitz = set([v.numpy() for v in save_bitz])
    assert set_bitz == {0, 1}


def test_anml_windows():
    dset = _build_anml_dataset()
    dset_win = get_anml_windows(dset, 3, "anml", 1)
    target_v1 = tf.constant([[0, 1, 2],
                             [1, 2, 3],
                             [4, 5, 6]], dtype=tf.int64)
    target_v2 = tf.constant([[0] * 3,
                             [0] * 3,
                             [1] * 3], dtype=tf.int64)
    for i, v in enumerate(dset_win):
        assert tf.math.reduce_all(v["v1"] == target_v1[i])
        assert tf.math.reduce_all(v["anml"] == target_v2[i])


if __name__ == "__main__":
    test_nan_reset()
    test_field_cond_sample()
    test_split_by_value()
    test_anml_windows()

    # window testing:
    # TODO: need better dataset for testing animal filter
    # get_anml_windows(dset, 2, "", shift=1)
