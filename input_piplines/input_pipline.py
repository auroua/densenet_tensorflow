import tensorflow as tf
import tensorlayer as tl
import os
import numpy as np


class Input(object):
    def __init__(self, base_path, file_lists, epochs, batch_size, capacity, min_after_dequeue, num_threads, image_shape,
                 dataset_name, phase, random_crop, crop_size, random_flip, random_change_brightness,
                 random_change_contrast, whitening, brightness, contrast_lower, contrast_upper):
        self.base_path = base_path
        self.files = file_lists
        self.epochs = epochs
        self.batch_size = batch_size
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.num_threads = num_threads
        self.image_shape = image_shape
        self.dataset_name = dataset_name
        self.phase = phase
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.random_flip = random_flip
        self.random_change_brightness = random_change_brightness
        self.random_change_contrast = random_change_contrast
        self.normalization_type = whitening
        self.brightness = brightness
        self.contrast_lower = contrast_lower
        self.contrast_upper = contrast_upper

    def build_pipline(self):
        file_names = [os.path.join(self.base_path, filename) for filename in self.files]
        filename_queue = tf.train.string_input_producer(file_names, num_epochs=self.epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            }
        )
        if self.dataset_name == 'mnist':
            # decode image and label from tfrecords
            image = tf.decode_raw(features['image_raw'], tf.uint8)
            image.set_shape([self.image_shape])
            image = tf.cast(image, tf.float32) * (1./255)
            label = tf.cast(features['label'], tf.int32)
        elif self.dataset_name == 'cifar10':
            # decode image and label from tfrecords
            image = tf.decode_raw(features['image_raw'], tf.float32)
            image = tf.reshape(image, [32, 32, 3])
            label = tf.cast(features['label'], tf.int32)
        elif self.dataset_name == 'svhn':
            image = tf.decode_raw(features['image_raw'], tf.uint8)
            image = tf.reshape(image, [32, 32, 3])
            image = tf.cast(image, tf.float32) * (1./255)
            label = tf.cast(features['label'], tf.int32)

        if self.phase == 'training':
            if self.random_crop:
                image = tf.random_crop(image, [self.crop_size, self.crop_size, 3])
            if self.random_flip:
                image = tf.image.random_flip_left_right(image)
            if self.random_change_brightness:
                image = tf.image.random_brightness(image, self.brightness)
            if self.random_change_contrast:
                image = tf.image.random_contrast(image, self.contrast_lower, self.contrast_upper)
            if self.normalization_type == 'by_channels':
                image = tf.image.per_image_standardization(image)
                # image_mean = tf.reduce_mean(image, axis=[0, 1], keep_dims=True)
                # image_std = tf.reduce_mean(tf.square(image-image_mean), axis=[0, 1], keep_dims=True)
                # image = (image-image_mean)/image_std
        else:
            if self.random_crop:
                image = tf.image.resize_image_with_crop_or_pad(image, self.crop_size, self.crop_size)
            if self.normalization_type == 'by_channels':
                image = tf.image.per_image_standardization(image)

        images_batch, label_batch = tf.train.shuffle_batch(
            [image, label], batch_size=self.batch_size, num_threads=self.num_threads,
            capacity=self.capacity,
            min_after_dequeue=self.min_after_dequeue
        )
        return images_batch, label_batch


    def build_test_pipline(self):
        file_names = [os.path.join(self.base_path, filename) for filename in self.files]
        filename_queue = tf.train.string_input_producer(file_names, num_epochs=self.epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            }
        )
        if self.dataset_name == 'mnist':
            # decode image and label from tfrecords
            image = tf.decode_raw(features['image_raw'], tf.uint8)
            image.set_shape([self.image_shape])
            image = tf.cast(image, tf.float32) * (1./255)
            label = tf.cast(features['label'], tf.int32)
        elif self.dataset_name == 'cifar10':
            # decode image and label from tfrecords
            image = tf.decode_raw(features['image_raw'], tf.float32)
            image = tf.reshape(image, [32, 32, 3])
            label = tf.cast(features['label'], tf.int32)
        elif self.dataset_name == 'svhn':
            image = tf.decode_raw(features['image_raw'], tf.uint8)
            image = tf.reshape(image, [32, 32, 3])
            image = tf.cast(image, tf.float32) * (1./255)
            label = tf.cast(features['label'], tf.int32)

        if self.random_crop:
            image = tf.image.resize_image_with_crop_or_pad(image, self.crop_size, self.crop_size)
        if self.normalization_type == 'by_channels':
            image = tf.image.per_image_standardization(image)
            # image_mean = tf.reduce_mean(image, axis=[0, 1], keep_dims=True)
            # image_std = tf.reduce_mean(tf.square(image - image_mean), axis=[0, 1], keep_dims=True)
            # image = (image - image_mean) / image_std

        images_batch, label_batch = tf.train.batch(
            [image, label], batch_size=self.batch_size, num_threads=self.num_threads,
            capacity=self.capacity
        )
        return images_batch, label_batch


if __name__ == '__main__':
    # test class
    input_pipline = Input('/home/aurora/workspaces/data/tfrecords_data/mnist_dataset', ['test.tfrecords'],
                          1, 100, 10000, 5000, 4, 784)
    images, labels = input_pipline.build_pipline()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(threads)
    print(tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS))
    with tf.control_dependencies([images, labels]):
        train_op = tf.no_op()

    try:
        step = 0
        while not coord.should_stop():
            sess.run(train_op)
            step += 1
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (input_pipline.epochs, step))
    finally:
        coord.request_stop()
    coord.join()
    sess.close()