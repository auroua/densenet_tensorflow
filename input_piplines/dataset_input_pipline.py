import tensorflow as tf
import os
from tensorflow.contrib.data import Iterator


def transformation_func_train(image, random_crop, crop_size, image_channel, random_flip, random_change_brightness,
                 random_change_contrast, normalization_type, brightness, contrast_lower, contrast_upper):
    if random_crop:
        image = tf.random_crop(image, [crop_size, crop_size, image_channel])
    if random_flip:
        image = tf.image.random_flip_left_right(image)
    if random_change_brightness:
        image = tf.image.random_brightness(image, brightness)
    if random_change_contrast:
        image = tf.image.random_contrast(image, contrast_lower, contrast_upper)
    if normalization_type == 'by_channels':
        image = tf.image.per_image_standardization(image)
    return image


def transformation_func_test(image, random_crop, crop_size, normalization_type):
    if random_crop:
        image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
    if normalization_type == 'by_channels':
        image = tf.image.per_image_standardization(image)
    return image


def generate_input_pipline(tfrecords_path, filenames, batch_size, test_batch_size, capacity, _parse_function_train, _parse_function_test):
    '''

    :param sess:
    :param tfrecords_path:
    :param filenames:
    :param epochs:
    :param batch_size:
    :param capacity:
    :param _parse_function: tf dataset api parse function, you should provide this function by yourself
    :return:
    '''
    file_paths = [os.path.join(tfrecords_path, name) for name in filenames]

    train_dataset = tf.contrib.data.TFRecordDataset(file_paths[0])
    train_dataset = train_dataset.map(_parse_function_train)
    train_dataset = train_dataset.repeat(1)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.shuffle(buffer_size=capacity)

    test_dataset = tf.contrib.data.TFRecordDataset(file_paths[1])
    test_dataset = test_dataset.map(_parse_function_test)
    test_dataset = test_dataset.repeat(1)
    test_dataset = test_dataset.batch(test_batch_size)

    iterator = Iterator.from_structure(train_dataset.output_types,
                                       train_dataset.output_shapes)
    next_element = iterator.get_next()
    training_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)
    return next_element, training_init_op, test_init_op
