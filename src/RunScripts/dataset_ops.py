"""Useful tensorflow Dataset operations
    NOTE: all of these operations assume elements are dictionaries

"""
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


def split_by_value(dset: tf.data.Dataset, field_name: str,
                   factor: float, offset: float = 0.0,
                   bit_field_name: str = "BIT", salt: int = 42):
    """adds BIT field that indicates whether sample is selected or not
    NOTE: requires that target field be float type

    Args:
        dset (tf.data.Dataset): dataset
        field_name (str): target field name
        factor (float):
        offset (float):
            eval_value = round(factor * (reduce_sum(input_val) + offset))
    """
    hash_layer = Hashing(2, salt=salt)
    def add_bit(x):
        x2 = {k: x[k] for k in x}
        v = tf.math.round(factor * (offset + tf.math.reduce_sum(x[field_name])))
        s = tf.strings.as_string(v)
        x2[bit_field_name] = hash_layer(s)
        return x2
    return dset.map(add_bit)


def get_anml_windows(dset: tf.data.Dataset, win_size: int,
                     anml_id_field: str, shift: int = 1):
    """transforms data into windowed format
        where windows are only allowed within animals

    Args:
        dset (tf.data.Dataset): dataset. Must be in order!
            Assumes: elements are dicts
        win_size (int): window size
        anml_id_field (str): field that identifies animals
            NOTE: for now, assumes this field is numeric
            ... uses diff under hood
        shift (int): how far to step between windows
            Ex: if 1 --> all windows will be 'shifted' one step
    """
    # window --> each element is k, v where v = dataset for given key
    win_dset = dset.window(win_size, shift=shift,
                           drop_remainder=True)

    # flat each dict elem
    # --> k, v where v = dataset with one element (batch combined)
    def dict_batch(x):
        # assumes x = k, v where v = dataset
        x2 = {k: x[k].batch(win_size) for k in x}
        # NOTE: zip transforms from dict of datasets
        # to dataset with dict elements (KEY)
        return tf.data.Dataset.zip(x2)
    # flat_map flattens nested dataset to non-nested dataset
    flat_dset = win_dset.flat_map(dict_batch)

    # filter anml:
    def anml_filter(x):
        v = x[anml_id_field]
        v_diff = v[1:] - v[:-1]
        return tf.math.reduce_all(v_diff == 0)
    return flat_dset.filter(anml_filter)


def select_idx(dset: tf.data.Dataset, field_name: str, idx: int):
    # select idx for 0th axis for a single field
    def sel(x):
        x2 = {k: x[k] for k in x if k != field_name}
        x2[field_name] = x[field_name][idx]
        return x2
    return dset.map(sel)
