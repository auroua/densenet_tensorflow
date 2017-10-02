import os
import tensorflow as tf
import logging
import gzip
import sys
import time
import numpy


flags = tf.app.flags

flags.DEFINE_string('mnist_path', '/home/aurora/workspaces/data/mnist', 'the mnist_datasets zip files path')
flags.DEFINE_string('tfrecords_path', '/home/aurora/workspaces/data/tfrecords_data/mnist_dataset', 'generated mnist_datasets files path')
flags.DEFINE_integer('mnist_size', 28, 'the mnist_datasets image height and width')
flags.DEFINE_integer('mnist_channel', 1, 'the mnist_datasets image channel size')
flags.DEFINE_integer('pixel_depth', 255, 'the mnist_datasets image channel size')
flags.DEFINE_integer('num_labels', 10, 'the mnist_datasets image channel size')
flags.DEFINE_integer('validate_size', 5000, 'the mnist_datasets image channel size')
FLAGS = flags.FLAGS

logging.basicConfig(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

data_files = ['train-images-idx3-ubyte.gz',  't10k-images-idx3-ubyte.gz']
label_files = ['train-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
data_size = [60000, 10000]


def get_mnist_data(filename):
    file_path = os.path.join(FLAGS.mnist_path, filename)
    if not tf.gfile.Exists(file_path):
        raise ValueError('file path %s does not exists' % file_path)
    with tf.gfile.GFile(file_path) as fid:
        logging.info('the file size is %d' % fid.size())
    return file_path


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(FLAGS.mnist_size * FLAGS.mnist_size * num_images * FLAGS.mnist_channel)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    # data = (data - (FLAGS.pixel_depth / 2.0)) / FLAGS.pixel_depth
    data = numpy.multiply(data, 1.0/255.0)
    data = data.reshape(num_images, FLAGS.mnist_size, FLAGS.mnist_size, FLAGS.mnist_channel)
    return data


def extract_data_tfrecords(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(FLAGS.mnist_size * FLAGS.mnist_size * num_images * FLAGS.mnist_channel)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, FLAGS.mnist_size, FLAGS.mnist_size, FLAGS.mnist_channel)
    return data


def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    return labels


def generate_data():
    datasets = []
    labels = []
    for index, file in enumerate(data_files):
        data_file_path = get_mnist_data(file)
        label_file_path = get_mnist_data(label_files[index])
        train_data = extract_data_tfrecords(data_file_path, data_size[index])
        label_data = extract_labels(label_file_path, data_size[index])
        datasets.append(train_data)
        labels.append(label_data)
    validation_data = datasets[0][:FLAGS.validate_size, :]
    validation_labels = labels[0][:FLAGS.validate_size]
    train_data = datasets[0][FLAGS.validate_size:, :]
    train_labels = labels[0][FLAGS.validate_size:]
    logging.info(train_data.shape)
    logging.info(train_labels.shape)
    logging.info(validation_data.shape)
    logging.info(validation_labels.shape)
    logging.info(datasets[1].shape)
    logging.info(labels[1].shape)
    return train_data, train_labels, validation_data, validation_labels, datasets[1], labels[1]


def dense_to_one_hot(labels):
    labels_array = numpy.zeros((labels.shape[0], 10), dtype=numpy.int32)
    labels_array[range(labels.shape[0]), labels] = 1
    return labels_array


if __name__ == '__main__':
    # test method get_mnist_data
    file_path = get_mnist_data('train-images-idx3-ubyte.gz')
    # test extract_data
    data = extract_data_tfrecords(file_path, 60000)
    logging.info(data.dtype)
    # generate_data method test
    _, labels, _, _, _, _ = generate_data()
    dense_to_one_hot(labels)