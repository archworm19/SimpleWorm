"""Useful tensorflow Dataset operations"""
import tensorflow as tf


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
    return dset.map(lambda x: set_func(x))


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
