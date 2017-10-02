import tensorflow as tf
from scipy.io import loadmat
import os

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/aurora/workspaces/data/svhn/format_2', 'path to svhn dataset')
flags.DEFINE_string('output_path', '/home/aurora/workspaces/data/tfrecords_data/svhn_dataset/', 'svhn output path')
FLAGS = flags.FLAGS


def data_set(data_dir, name, num_sample_size=10000):
    filename = os.path.join(data_dir, name + '_32x32.mat')
    if not os.path.isfile(filename):
        raise ValueError('Please supply a the file')
    datadict = loadmat(filename)
    train_x = datadict['X']
    train_x = train_x.transpose((3, 0, 1, 2))
    print(train_x.shape)
    train_y = datadict['y'].flatten()
    train_y[train_y == 10] = 0
    train_x = train_x[:num_sample_size]
    train_y = train_y[:num_sample_size]
    return train_x, train_y


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecords(images, labels, fileName):
    num_examples, rows, cols, depth = images.shape
    paths = os.path.join(FLAGS.output_path, fileName)
    print('Writing', paths)
    writer = tf.python_io.TFRecordWriter(paths)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    train_x, train_y = data_set(FLAGS.data_dir, 'train', 73257)
    test_x, test_y = data_set(FLAGS.data_dir, 'test', 26032)
    extra_x, extra_y = data_set(FLAGS.data_dir, 'extra', 531131)
    convert_to_tfrecords(train_x, train_y, 'train.tfrecords')
    convert_to_tfrecords(test_x, test_y, 'test.tfrecords')
    convert_to_tfrecords(extra_x, extra_y, 'extra.tfrecords')
