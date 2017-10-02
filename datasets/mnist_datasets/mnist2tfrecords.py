import mnist_utils as utils
import tensorflow as tf
import os

flags = tf.app.flags
FLAGS = flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def generate_tfrecords(data, label, name):
    """Converts a dataset to tfrecords."""

    filename = os.path.join(FLAGS.tfrecords_path, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(data.shape[0]):
        image_raw = data[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(FLAGS.mnist_size),
            'width': _int64_feature(FLAGS.mnist_size),
            'depth': _int64_feature(FLAGS.mnist_channel),
            'label': _int64_feature(int(label[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    print(FLAGS.mnist_size)
    train_data, train_label, val_data, val_label, test_data, test_label = utils.generate_data()
    generate_tfrecords(train_data, train_label, 'train')
    generate_tfrecords(val_data, val_label, 'validation')
    generate_tfrecords(test_data, test_label, 'test')
