# %matplotlib inline

from __future__ import division

import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np
import pdb
from nets import dataset_factory
from nets import nets_factory
from nets import preprocessing_factory
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import monitored_session


tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/home/thn2079/git/stamp_project/save_evaluation_results_plate_June14/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 2,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'stamp', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'resnet_v2_50', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

checkpoint_path = '/home/thn2079/train_logs_plate_June14/model.ckpt-57019'
slim = tf.contrib.slim
image_filename = '/shared/kgcoe-research/mil/stamp_stamp/data/plate/100_BK_15255.jpg'
def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    # [image, label] = provider.get(['image', 'label'])
    [image, label_0, label_1, label_2] = provider.get(['image', 'label_0', 'label_1', 'label_2'])
    # label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
    ####################
    # Define the model #
    ####################
    # pdb.set_trace()
    image_filename_placeholder = tf.placeholder(tf.string)
    image_tensor = tf.read_file(image_filename_placeholder)
    image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
    image_tensor = image_preprocessing_fn(image_tensor, eval_image_size, eval_image_size)
    image_batch_tensor = tf.expand_dims(image_tensor, axis=0)
    logits, endpoints = network_fn(image_batch_tensor)
    predictions_0 = tf.argmax(logits[:,:,0], 1)
    predictions_1 = tf.argmax(logits[:,:,1], 1)
    predictions_2 = tf.argmax(logits[:,:,2], 1)
    if FLAGS.moving_average_decay:
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, tf_global_step)
        variables_to_restore = variable_averages.variables_to_restore(
            slim.get_model_variables())
        variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
        variables_to_restore = slim.get_variables_to_restore()

    initializer = tf.local_variables_initializer()
    # saver = tf.train.Saver()
    saver = tf_saver.Saver(variables_to_restore)

    with tf.Session() as sess:
        sess.run(initializer)
        saver.restore(sess, checkpoint_path)
        # pdb.set_trace()

        predictions_0, predictions_1, predictions_2 = sess.run([predictions_0, predictions_1, predictions_2], feed_dict={image_filename_placeholder: image_filename})
    # print logits_pred
    pdb.set_trace()


if __name__ == '__main__':
  tf.app.run()

