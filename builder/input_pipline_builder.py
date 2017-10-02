import tensorflow as tf
from input_pipline import Input
import input_config_pb2
from google.protobuf import text_format


def builder(config_path):
    input_config = input_config_pb2.InputPipline()
    with tf.gfile.GFile(config_path) as fid:
        text_format.Merge(fid.read(), input_config)
    input_pipline = Input(base_path=input_config.file_location, file_lists=input_config.tfrecords, epochs=input_config.epochs,
                          batch_size=input_config.batch_size, capacity=input_config.capacity, min_after_dequeue=input_config.min_after_dequeue,
                          num_threads=input_config.num_threads, image_shape=input_config.image_shape, dataset_name=input_config.dataset_name,
                          phase=input_config.phase, random_crop=input_config.random_crop, crop_size=input_config.crop_size, random_flip=input_config.random_flip,
                          random_change_brightness=input_config.random_change_brightness, random_change_contrast=input_config.random_change_contrast,
                          whitening=input_config.normalization_type, brightness=input_config.brightness, contrast_lower=input_config.contrast_lower,
                          contrast_upper=input_config.contrast_upper)
    return input_pipline, input_config