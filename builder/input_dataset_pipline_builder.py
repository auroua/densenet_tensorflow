import tensorflow as tf
import input_config_dataset_pb2
from google.protobuf import text_format
from dataset_input_pipline import transformation_func_train, transformation_func_test, generate_input_pipline
from functools import partial


def _parse_function_train(example_proto, random_crop, crop_size, image_channel, random_flip, random_change_brightness,
                 random_change_contrast, normalization_type, brightness, contrast_lower, contrast_upper):
  features = {"image_raw": tf.FixedLenFeature((), tf.string, default_value=""),
              "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features)
  img = tf.decode_raw(parsed_features['image_raw'], tf.float32)
  img = tf.reshape(img, [32, 32, 3])
  img = transformation_func_train(img, random_crop, crop_size, image_channel, random_flip, random_change_brightness,
                 random_change_contrast, normalization_type, brightness, contrast_lower, contrast_upper)
  return img, parsed_features["label"]


def _parse_function_test(example_proto, random_crop, crop_size, normalization_type):
  features = {"image_raw": tf.FixedLenFeature((), tf.string, default_value=""),
              "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features)
  img = tf.decode_raw(parsed_features['image_raw'], tf.float32)
  img = tf.reshape(img, [32, 32, 3])
  img = transformation_func_test(img, random_crop=random_crop, crop_size=crop_size, normalization_type=normalization_type)
  return img, parsed_features["label"]


def builder(config_path):
    input_config = input_config_dataset_pb2.InputDatasetPipline()
    with tf.gfile.GFile(config_path) as fid:
        text_format.Merge(fid.read(), input_config)
    _parse_function_train_part = partial(_parse_function_train, random_crop=input_config.random_crop, crop_size=input_config.crop_size,
                                            image_channel=input_config.image_channel , random_flip=input_config.random_flip,
                                            random_change_brightness=input_config.random_change_brightness,
                                            random_change_contrast=input_config.random_change_contrast,
                                            normalization_type=input_config.normalization_type, brightness=input_config.brightness,
                                            contrast_lower=input_config.contrast_lower,
                                            contrast_upper=input_config.contrast_upper)
    _parse_function_test_part = partial(_parse_function_test, random_crop=input_config.random_crop, crop_size=input_config.crop_size,
                                      normalization_type=input_config.normalization_type)
    next_element, training_init_op, test_init_op = generate_input_pipline(input_config.file_location, input_config.tfrecords,
                                        input_config.batch_size, test_batch_size=input_config.test_batch_size,
                                        capacity=input_config.capacity, _parse_function_train=_parse_function_train_part,
                                                            _parse_function_test=_parse_function_test_part)
    return next_element, training_init_op, test_init_op, input_config


if __name__ == '__main__':
    next_element, training_init_op, test_init_op, _ = builder('/home/aurora/workspaces/PycharmProjects/object_detection_'
                           'models/config_files/cifar10_input_dataset.config')
    sess = tf.Session()
    # Run 3 epochs in which the training dataset is traversed, followed by the
    # validation dataset.
    for i in range(3):
        # Initialize an iterator over the training dataset.
        sess.run(training_init_op)
        counter = 0
        while True:
            try:
                images, label = sess.run(next_element)
                counter += 1
            except tf.errors.OutOfRangeError:
                print('End of training data in loop %d, totoal loop is %d' %(i, counter))
                break

        sess.run(test_init_op)
        counter = 0
        while True:
            try:
                images, label = sess.run(next_element)
                counter += 1
            except tf.errors.OutOfRangeError:
                print('End of testing data in loop %d, total loop is %d' %(i, counter))
                break