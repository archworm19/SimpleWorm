"""Proper Design for this?

  > reader and writer functions ~ operate on single animal
  > > Writer
  > > > Input
  > > > > file name
  > > > > Dict: mapping from tensor name to tensor (all tensors have time/samples on 0th axis)
  > > > Output
  > > > > Nothing
  > > Reader
  > > > Input
  > > > > tfrecords file name(s)  TODO: can we read multiple files as if they were one?
  > > > > Dict: mapping from tensor name to tensor dtype

"""
import numpy as np
import tensorflow as tf
from typing import Dict

def write_numpy_to_tfr(file_name: str, np_map: Dict[str, np.ndarray]):
    """Write numpy tensors to tfrecords file

    Args:
        file_name (str): file name to tfrecords file
        np_map (Dict[str, np.ndarray]): mapping from tensor name
          to tensor
          Assumes: every tensor is organized such that the 0
            axis is the sampling axis (typically, different timepoints)
    """
    # check that all tensors are the same length:
    Ts = [np.shape(np_map[k])[0] for k in np_map]
    assert(all([Tsi == Ts[0] for Tsi in Ts]))

    with tf.io.TFRecordWriter(file_name) as file_writer:
        for i in range(Ts[0]):
            feats = {}
            for k in np_map:
                # tensor convert
                z_tf = tf.convert_to_tensor(np_map[k][i])
                # serialize:
                z_ser = tf.io.serialize_tensor(z_tf)
                # pkg into a bytes feature:
                # NOTE: numpy() call gets the hex rep in byte form ~ effectively removes other fields
                z_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[z_ser.numpy()]))
                # save feature:
                feats[k] = z_feat
    
            # pkg example:
            ex = tf.train.Example(features=tf.train.Features(feature = feats))
            file_writer.write(ex.SerializeToString())


if __name__ == "__main__":
    import numpy.random as npr
    # TODO/TESTING:
    file_name = 'sample.tfr'
    np_map = {"x": npr.rand(10,5,2),
              "y": npr.rand(10, 8)}
    dtype_map = {k: np_map[k].dtype for k in np_map}

    # write samples to sample.tfr
    write_numpy_to_tfr(file_name, np_map)
    
    # TODO: everything below here should be packaged into a function

    # load the tfrecords
    tfr_dset = tf.data.TFRecordDataset(file_name)

    # make the parse mapping:
    parse_dict = {k: tf.io.FixedLenFeature([], tf.string) for k in np_map}
    def parse_elem(elem):
        ex_message = tf.io.parse_single_example(elem, parse_dict)
        feats = [tf.io.parse_tensor(ex_message[k], out_type=dtype_map[k]) for k in parse_dict]
        return feats

    # use map to get dataset:
    dataset = tfr_dset.map(parse_elem)
    for v in dataset:
        print(v)
