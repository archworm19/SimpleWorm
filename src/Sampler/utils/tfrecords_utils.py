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
import os.path
import numpy as np
import tensorflow as tf
from typing import List, Dict

def write_numpy_to_tfr(file_name: str, np_map: Dict[str, np.ndarray]):
    """Write numpy tensors to tfrecords file

    Args:
        file_name (str): file name to tfrecords file
        np_map (Dict[str, np.ndarray]): mapping from tensor name
          to tensor
          Assumes: every tensor is organized such that the 0
            axis is the sampling axis (typically, different timepoints)
    """
    # assert that filename does not exist
    assert(not os.path.exists(file_name)), "file already exists"

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


def open_tfr(file_names: List[str], dtype_map: Dict):
    """Open tf records files --> convert to tfrecords dataset

    Args:
        file_names (List[str]): list of filenames.
            Every file is assumed to use the same data schema.
        dtype_map (Dict): mapping from variable names to variable dtypes.
            variable names assumed to match variable names
            used in saving the tfrecords files (the keys in np_map)

    Returns:
        tf.MapDataset: the dataset made up of tf tensors
    """
    # make the parse mapping:
    parse_dict = {k: tf.io.FixedLenFeature([], tf.string) for k in dtype_map}

    # open the datasets:
    tfr_dset = tf.data.TFRecordDataset(file_names)

    def parse_elem(elem):
        ex_message = tf.io.parse_single_example(elem, parse_dict)
        feats = [tf.io.parse_tensor(ex_message[k], out_type=dtype_map[k]) for k in parse_dict]
        return feats

    # use map to get dataset:
    dataset = tfr_dset.map(parse_elem)
    return dataset


def convert_tfdset_numpy(tfdset, target_idx: int):
    """Convert tensorflow dataset to single numpy array
        > stack the samples into single array
    
    Args:
        tfsets: tensorflow dataset
        target_idx: which tensor to pull from
    
    Returns:
        np.ndarray: T x example_dims"""
    rawv = [v[target_idx].numpy() for v in tfdset]
    return np.array(rawv)
    

if __name__ == "__main__":
    import numpy.random as npr
    # initial testing without nans
    file_name = 'sample.tfr'
    file_name2 = 'sample2.tfr'
    np_map = {"x": npr.rand(10,5,2),
              "y": npr.rand(10, 8)}
    dtype_map = {k: np_map[k].dtype for k in np_map}

    # write samples to sample.tfr
    write_numpy_to_tfr(file_name, np_map)
    write_numpy_to_tfr(file_name2, np_map)

    # load up the datasets:
    dataset = open_tfr([file_name, file_name2], dtype_map)
    for i, (x, y) in enumerate(dataset):
        print('sample{0}'.format(str(i)))
        print(x.shape)
        print(y.shape)

    # can we shuffle the dataset? yup ~ can do all normal ops
    dataset.shuffle(1)

    # does tensorflow even allow nans? yes
    file_name = 'sample_nan.tfr'
    np_map = {"x": npr.rand(10, 5, 2),
              "y": npr.rand(10, 8)}
    dtype_map = {k: np_map[k].dtype for k in np_map}
    # set the nan
    np_map["x"][1,1,1] = np.nan
    write_numpy_to_tfr(file_name, np_map)
    # load up the datasets:
    dataset = open_tfr([file_name, file_name2], dtype_map)
    for i, (x, y) in enumerate(dataset):
        if i == 1:
            print('sample{0}'.format(str(i)))
            print(x)
    
    print(convert_tfdset_numpy(dataset, 0))
