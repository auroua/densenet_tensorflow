import tensorflow as tf
import densenet_pb2
from google.protobuf import text_format
from dense_net_utils import stack_blocks
import numpy as np
import logging
from net_utils import get_initializer, get_shape, count_trainable_params, activation_summary


def builder(inputs, config_path, phase, scope_name, reuse=False):
    net_config = densenet_pb2.DenseNet()
    with tf.gfile.GFile(config_path) as fid:
        text_format.Merge(fid.read(), net_config)
        assert len(net_config.block_size) == net_config.dense_blocks, 'config error'
    inputs_shape = get_shape(inputs)
    logging.info(inputs_shape)
    with tf.variable_scope(scope_name, reuse=reuse):
        with tf.variable_scope('head_block'):
            head_weights = tf.get_variable('weights', shape=[net_config.head_kernel_size, net_config.head_kernel_size, inputs_shape[-1], net_config.k_size*2]
                                    , dtype=tf.float32, initializer=get_initializer(net_config.head_block_weight_initializer)())
            net = tf.nn.conv2d(inputs, head_weights, strides=[1, net_config.head_stride_size, net_config.head_stride_size, 1], padding='SAME')
            if net_config.head_pool_kernel_size!=0 and net_config.head_pool_stride_size!=0:
                net = tf.nn.max_pool(net, ksize=[1,  net_config.head_pool_kernel_size, net_config.head_pool_kernel_size, 1],
                                 strides=[1, net_config.head_pool_stride_size, net_config.head_pool_stride_size, 1], padding='SAME')
        net = tf.contrib.layers.batch_norm(net, is_training=phase == 'train', scale=True, updates_collections=None)
        net = tf.nn.relu(net)
        activation_summary(net)
        logging.info('########################')
        logging.info(net.shape)
        net = stack_blocks(inputs=net, blocks=net_config.block_size, k_size=net_config.k_size,
                        boottleneck=net_config.densenet_b, densenet_c=net_config.densenet_c, theta=net_config.theta,
                        bn_relu=net_config.transition_layers_bn_relu, bias_add=net_config.bias_add,
                        dropout_rate=net_config.dropout_rate, dropout=net_config.dropout, phase=phase,
                        weights_initializer=get_initializer(net_config.dense_block_weights_initializer),
                        intern_rate=net_config.intern_rate)
        net = tf.contrib.layers.batch_norm(net, is_training=phase == 'train', scale=True, updates_collections=None)
        net = tf.nn.relu(net)
        activation_summary(net)
        with tf.variable_scope('pooling_block', reuse=reuse):
            net = tf.nn.avg_pool(net, ksize=[1, net.get_shape()[-2], net.get_shape()[-2], 1],
                                     strides=[1,net.get_shape()[-2], net.get_shape()[-2], 1], padding='VALID')
            net = tf.squeeze(net, axis=[1, 2])
            output_weights = tf.get_variable('fully_weights', shape=[net.shape[-1], net_config.output_num_class], dtype=tf.float32,
                                                 initializer= get_initializer(net_config.pool_block_weights_initializer)())
            output_bias = tf.get_variable('fully_bias', shape=[net_config.output_num_class], dtype=tf.float32, initializer=tf.zeros_initializer())
            logit = tf.matmul(net, output_weights) + output_bias

    logging.info('=================================network info===================================')
    count_trainable_params()
    return logit, net_config


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # np_inputs = np.random.normal(size=(1, 32, 32, 3))
    # np_inputs = np_inputs.astype(dtype=np.float32)
    # t_np_inputs = tf.convert_to_tensor(np_inputs, name='np2tensor')
    sess = tf.Session()
    # net, net_c = builder(t_np_inputs, '/home/aurora/workspaces/PycharmProjects/object_detection_models/config_files/'
    #                            'densenet/densenet_cifar10_bc.config', phase='train', scope_name='densenet_cifar10')
    # net_test, net_test_c = builder(t_np_inputs, '/home/aurora/workspaces/PycharmProjects/object_detection_models/config_files/'
    #                            'densenet/densenet_cifar10_bc.config', phase='test', scope_name='densenet_cifar10', reuse=True)
    # sess.run(tf.global_variables_initializer())
    # outputs = sess.run(net)

    # image net input information
    np_inputs_imagenet = np.random.normal(size=(1, 224, 224, 3))
    np_inputs_imagenet = np_inputs_imagenet.astype(dtype=np.float32)
    t_np_inputs_imagenet = tf.convert_to_tensor(np_inputs_imagenet, name='np2tensor')
    net, net_c = builder(t_np_inputs_imagenet, '/home/aurora/workspaces/PycharmProjects/object_detection_models/config_files/densenet/'
                                               'densenet_cifar10_250_24_bc.config', phase='train', scope_name='densenet_imagenet')
    net_test, net_test_c = builder(t_np_inputs_imagenet,
                                   '/home/aurora/workspaces/PycharmProjects/object_detection_models/config_files/'
                                   'densenet/densenet_cifar10_250_24_bc.config', phase='test', scope_name='densenet_imagenet',
                                   reuse=True)
    sess.run(tf.global_variables_initializer())
    outputs = sess.run(net)
