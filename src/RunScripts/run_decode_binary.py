"""Binary Decoding Problem + Soft Forest
    = Predict probability that stimulus is On vs Off"""

import tensorflow as tf
import os
import numpy as np
import numpy.random as npr
import pylab as plt
from functools import partial
from RunScripts.data_loader import get_datasets
from RunScripts.dataset_ops import set_nans_to_val, sample_field_conditional, get_anml_windows


# TODO: this crap needs testing!
# NOTE: assumed to operate on windowed data
def t_select(x, factor: float = 10., offset: float = 0.):
    val = tf.round(x["t"][0] * factor)
    mod_val = tf.math.floormod(val + offset, tf.constant(2., val.dtype))
    return tf.math.reduce_all(mod_val == 0)


if __name__ == "__main__":

    TWINDOW_SIZE = 16

    # return each type as a separate dataset
    # fields: 1. cell_clusters (T x num_clust matrix),
    #         2. normalized time = t
    #         3. anml_spec (vector) ~ experiment specification
    #         4. set_name (str)
    #         5. anml_id (int) ~ unique animal id
    dsets = get_datasets()
    print("number of datasets: " + str(len(dsets)))

    # set nans to 0
    dsets = [set_nans_to_val(ds, "cell_clusters", 0.) for ds in dsets]

    # TODO: anml id unique across sets or within sets?

    # split train/cross vs. test by animal
    dsets = [sample_field_conditional(ds, "anml_id", 3) for ds in dsets]
    test_dsets = [ds.filter(lambda x: x["BIT"]) for ds in dsets]
    trcv_dsets = [ds.filter(lambda x: not x["BIT"]) for ds in dsets]

    # TODO/TESTING
    for ds in test_dsets:
        idz = []
        for v in ds:
            if str(v["anml_id"].numpy()) not in idz:
                idz.append(str(v["anml_id"].numpy()))
        print(idz)

    # train-cross splits and processing:
    for salt in range(3):
        # split train vs cross:
        trcv2 = [sample_field_conditional(ds, "anml_id", 2, salt=salt) for ds in dsets]
        train_dsets = [ds.filter(lambda x: x["BIT"]) for ds in dsets]
        cross_dsets = [ds.filter(lambda x: not x["BIT"]) for ds in dsets]

        # get timewindows:
        train_wins = [get_anml_windows(ds, TWINDOW_SIZE, "anml_id") for ds in train_dsets]
        cross_wins = [get_anml_windows(ds, TWINDOW_SIZE, "anml_id") for ds in cross_dsets]

        # NOTE/KEY: time splitting system
        # > timewindows used in train should NOT be used in cross (even for different animals)
        train_wins = [ds.filter(partial(t_select, offset=salt)) for ds in train_wins]
        cross_wins = [ds.filter(partial(t_select, offset=salt + 1)) for ds in cross_wins]

        # TODO: does t splitting system work?
        plt.figure()
        for v in train_wins[0]:
            x = v["t"].numpy()
            plt.plot(x, [-1] * len(x))
        for v in cross_wins[0]:
            x = v["t"].numpy()
            plt.plot(x, [1] * len(x))
        plt.show()