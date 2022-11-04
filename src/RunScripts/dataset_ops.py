"""Useful tensorflow Dataset operations"""
import tensorflow as tf
from tensorflow.keras.layers import Hashing


def set_nans_to_val(dset: tf.data.Dataset, field_name: str, val: float):
    """set nans in given field to some value
    NOTE: assumes field is numeric type

    Args:
        dset (tf.data.Dataset): root dataset
        field_name (str): name of target field within dataset
        val (float): new value
    """
    def set_func(x):
        # copy over other fields:
        x2 = {k: x[k] for k in x if k != field_name}
        # tensor op friendly soln:
        nan_bools = tf.math.is_nan(x[field_name])
        nan_inds = tf.where(nan_bools)
        good_inds = tf.where(tf.math.logical_not(nan_bools))
        uv_const = tf.constant([val], dtype=x[field_name].dtype)
        update_vals = tf.tile(uv_const, [tf.shape(nan_inds)[0]])
        ret_shape = tf.cast(tf.shape(x[field_name]), nan_inds.dtype)
        tf.scatter_nd(nan_inds, update_vals, ret_shape)
        v2 = (tf.scatter_nd(good_inds, tf.gather_nd(x[field_name], good_inds), ret_shape) +
              tf.scatter_nd(nan_inds, update_vals, ret_shape))
        x2[field_name] = v2
        return x2
    return dset.map(set_func)


def sample_field_conditional(dset: tf.data.Dataset, field_name: str, num_bins: int,
                             salt: int = 42):
    """sample from dataset
    where sampling is conditioned on the value in specified field
    Ex: target field as 2 values: A, B
    > if A is selected for set --> all entries with value A will
    be passed thru

    Args:
        dset (tf.data.Dataset): input dataset
        field_name (str): target field name
        num_bins (int): probability of selection = (1 / num_bins)
        salt (int): acts as random seed
    """
    hash_layer = Hashing(num_bins, salt=salt)
    def select_elem(x):
        s = tf.strings.reduce_join(tf.strings.as_string(x[field_name]),
                                   separator=",")
        return tf.math.reduce_all(hash_layer(s) == 0)
    return dset.filter(select_elem)


def split_by_value(dset: tf.data.Dataset, field_name: str, factor: float,
                   bit_field_name: str = "BIT", salt: int = 42):
    """adds BIT field that indicates whether sample is selected or not
    NOTE: requires that target field be float type

    Args:
        dset (tf.data.Dataset): dataset
        field_name (str): target field name
        factor (float):
            eval_value = round(factor * reduce_sum(input_val))
    """
    hash_layer = Hashing(2, salt=salt)
    def add_bit(x):
        x2 = {k: x[k] for k in x}
        v = tf.math.round(factor * tf.math.reduce_sum(x[field_name]))
        s = tf.strings.as_string(v)
        x2[bit_field_name] = hash_layer(s)
        return x2
    return dset.map(add_bit)



if __name__ == "__main__":
    # TODO: move these to tests
    import numpy as np
    v1 = np.array([[1., np.nan],
                   [3., 4.]])
    v2 = np.array([5., 6.])
    d = {"v1": v1, "v2": v2}
    dset = tf.data.Dataset.from_tensor_slices(d)

    # testing nan resetting
    dset_nonan = set_nans_to_val(dset, "v1", 2.)
    for v in dset_nonan:
        print(v)

    # test element selection:
    # > dataset output should be constant for single salt
    # > can differ across salts
    for salt in [0, 10, 42]:
        print('salt test')
        for _ in range(2):
            dset_filter = sample_field_conditional(dset, "v2", 2, salt)
            for v in dset_filter:
                print(v)

    # test tradeoff:
    # factor = 1. --> should be able to separate
    print("TRADEOFF")
    for salt in [0, 10, 42]:
        dset_trade = split_by_value(dset, "v2", 1., salt=salt)
        for v in dset_trade:
            print(v)
    # factor = 0.1 --> should always group together
    # TODO: FAILURE at 0.1 (wrks at 0.0)
    print("TRADEOFF GROUP")
    for salt in [0, 10, 42]:
        dset_trade = split_by_value(dset, "v2", 0.0, salt=salt)
        for v in dset_trade:
            print(v)
