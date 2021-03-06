# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import pdb
import tensorflow as tf
import pickle as pkl
from nets import dataset_utils
from pre_processing import glob2
# The URL where the Flowers data can be downloaded.
_DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'

# The number of images in the validation set.
_NUM_VALIDATION = 350

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  # flower_root = os.path.join(dataset_dir, 'flower_photos')
  # flower_root = dataset_dir
  directories = []
  class_names = []
  all_classes = []
  # for filename in os.listdir(flower_root):
  #   path = os.path.join(flower_root, filename)
  #   if os.path.isdir(path):
  #     directories.append(path)
  #     class_names.append(filename)

  # photo_filenames = []
  # for directory in directories:
  #   for filename in os.listdir(directory):
  #     path = os.path.join(directory, filename)
  #     photo_filenames.append(path)
  jpg_imgs = glob2.glob(dataset_dir+'/**/*.jpg')
  capital_jpg_imgs = glob2.glob(dataset_dir+'/**/*.JPG')
  photo_filenames = jpg_imgs + capital_jpg_imgs
  # pdb.set_trace()
  photo_filenames = filter(lambda x:x.split('/')[-1].split('_')[0]!='U',photo_filenames)
  for i in xrange(len(photo_filenames)):
    img_name = photo_filenames[i]
    name = img_name.split('/')[-1]
  # pdb.set_trace()
  # try:
    class_name = name.split('_')[0] 
    if class_name not in all_classes:
      all_classes.append(class_name)
    # row_name = name.split('_')[0][1]
    # column_name = name.split('_')[0][0]
    # if ord(row_name) not in all_classes:
    #   all_classes.append(ord(row_name))
    # if ord(column_name) not in all_classes:
    #   all_classes.append(ord(column_name))
    # if ord(row_name)>97 or ord(column_name)>97:
    #   pdb.set_trace() 
  # except:
  #   pass
  # with open('path_to_plate.pkl','w') as f:
    # pkl.dump(photo_filenames, f)
  # pdb.set_trace()
  # class_names = [str(unichr(i)) for i in sorted(all_classes)]
  
  ##Make classes for plate
  all_classes = [str(i) for i in range(10)]+['BLANK']
  
  return photo_filenames, sorted(all_classes)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'stamps_plate_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir, output_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            output_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
            height, width = image_reader.read_image_dims(sess, image_data)
            # class_name = os.path.basename(os.path.dirname(filenames[i]))
            ##Convert to class name for row_column
            class_name = filenames[i].split('/')[-1].split('_')[0]
            # class_name = str(int(class_name))
            if int(class_name)<100:
                all_class_plate = ['BLANK'] + list(class_name)
                # pdb.set_trace()
            else:
                all_class_plate = list(class_name)

            class_id_0 = class_names_to_ids[all_class_plate[0]]
            class_id_1 = class_names_to_ids[all_class_plate[1]]
            class_id_2 = class_names_to_ids[all_class_plate[2]]
            
            example = dataset_utils.image_to_tfexample(
                image_data, 'jpg', height, width, class_id_0, class_id_1, class_id_2)
                
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, 'flower_photos')
  tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(dataset_dir, output_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  if _dataset_exists(output_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  # dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
  photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))
  
  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  NUM_VALIDATION = int(len(photo_filenames)*0.3)
  # pdb.set_trace()
  training_filenames = photo_filenames[NUM_VALIDATION:]
  validation_filenames = photo_filenames[:NUM_VALIDATION]

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, class_names_to_ids, dataset_dir, output_dir)
  _convert_dataset('validation', validation_filenames, class_names_to_ids, dataset_dir, output_dir)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, output_dir)
  pdb.set_trace()
  # _clean_up_temporary_files(dataset_dir, output_dir)
  print('\nFinished converting the Stamps dataset!')

