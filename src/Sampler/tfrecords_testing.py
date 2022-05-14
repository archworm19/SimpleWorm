import numpy as np
import tensorflow as tf

example_path = "sample.tfr"

with tf.io.TFRecordWriter(example_path) as file_writer:
  for _ in range(4):
    x, y, x2 = np.random.rand(10, 5), np.random.rand(3, 3), np.random.rand(2, 3, 4)


    # How to serialize tensors?
    # From tensorflow docs ->
    # convert to tensor: tf.convert_to_tensor(array)
    # serialize tensor: tf.io.serialize_tensor(tensor)

    # tensor conversion + serialization:
    feats = {}
    for z, z_name in zip([x, y, x2], ['x', 'y', 'x2']):
      # tensor convert
      z_tf = tf.convert_to_tensor(z)
      # serialize:
      z_ser = tf.io.serialize_tensor(z_tf)
      # pkg into a bytes feature:
      # NOTE: numpy() call gets the hex rep in byte form ~ effectively removes other fields
      z_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[z_ser.numpy()]))
      # save feature:
      feats[z_name] = z_feat
    
    # pkg example:
    ex = tf.train.Example(features=tf.train.Features(feature = feats))
    file_writer.write(ex.SerializeToString())

# TODO: how to read?
tfr_dset = tf.data.TFRecordDataset(example_path)

# make the parse mapping:
def parse_elem(elem):
  parse_dict = {'x': tf.io.FixedLenFeature([], tf.string),
                'y': tf.io.FixedLenFeature([], tf.string),
                'x2': tf.io.FixedLenFeature([], tf.string),}
  ex_message = tf.io.parse_single_example(elem, parse_dict)
  feats = [tf.io.parse_tensor(ex_message[k], out_type=tf.float64) for k in parse_dict]
  return feats

# use map to get dataset:
dataset = tfr_dset.map(parse_elem)
for v in dataset:
  print(v)