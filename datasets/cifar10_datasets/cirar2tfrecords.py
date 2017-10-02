import tensorlayer as tl
import tensorflow as tf
import numpy as np
import os

flags = tf.app.flags
flags.DEFINE_string('cifar_path', '/home/aurora/workspaces/data/cifar10/', 'path to cifar10 datasets')
flags.DEFINE_string('cifar_output_path', '/home/aurora/workspaces/data/tfrecords_data/cifar_dataset/', 'the output path to cifar 10 tfrecords')
FLAGS = flags.FLAGS


def data_to_tfrecord(images, labels, filename):
    print("Converting data into %s ..." % filename)
    output_path = os.path.join(FLAGS.cifar_output_path, filename)
    writer = tf.python_io.TFRecordWriter(output_path)
    for index, img in enumerate(images):
        img_raw = img.tobytes()
        # tl.visualize.frame(np.asarray(img, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        label = labels[index]
        # image = np.fromstring(img_raw, np.float32)
        # image = image.reshape([32, 32, 3])
        # tl.visualize.frame(np.asarray(image, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1333)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()


def read_and_decode(filename, is_train=None):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                       })
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [32, 32, 3])
    # img = tf.cast(img, tf.float32) #* (1. / 255) - 0.5
    if is_train == True:
        # 1. Randomly crop a [height, width] section of the image.
        img = tf.random_crop(img, [24, 24, 3])
        # 2. Randomly flip the image horizontally.
        img = tf.image.random_flip_left_right(img)
        # 3. Randomly change brightness.
        img = tf.image.random_brightness(img, max_delta=63)
        # 4. Randomly change contrast.
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        # 5. Subtract off the mean and divide by the variance of the pixels.
        try: # TF 0.12+
            img = tf.image.per_image_standardization(img)
        except: # earlier TF versions
            img = tf.image.per_image_whitening(img)

    elif is_train == False:
        # 1. Crop the central [height, width] of the image.
        img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)
        # 2. Subtract off the mean and divide by the variance of the pixels.
        try: # TF 0.12+
            img = tf.image.per_image_standardization(img)
        except: # earlier TF versions
            img = tf.image.per_image_whitening(img)
    elif is_train == None:
        img = img

    label = tf.cast(features['label'], tf.int32)
    return img, label


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3),
                                                                     plotable=False, path=FLAGS.cifar_path)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)

    print('X_train.shape', X_train.shape)  # (50000, 32, 32, 3)
    print('y_train.shape', y_train.shape)  # (50000,)
    print('X_test.shape', X_test.shape)  # (10000, 32, 32, 3)
    print('y_test.shape', y_test.shape)  # (10000,)
    print('X %s   y %s' % (X_test.dtype, y_test.dtype))
    data_to_tfrecord(X_train, y_train, 'train.tfrecords')
    data_to_tfrecord(X_test, y_test, 'test.tfrecords')

    # read_and_decode('/home/aurora/workspaces/data/tfrecords_data/cifar_dataset/train.tfrecords')