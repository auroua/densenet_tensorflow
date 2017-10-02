import tensorflow as tf
import logging


def count_trainable_params():
    total_parameters = 0
    logging.info('######################################')
    for variable in tf.trainable_variables():
        logging.info(variable.op.name)
        logging.info(str(variable.get_shape()))
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total training params: %.1fM" % (total_parameters / 1e6))


def activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = x.op.name
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def add_variables_to_summary():
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)


def add_variables_grad_to_summary(grads):
    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)


def get_initializer(init_type):
    if init_type == 'variance_scaling_initializer':
        return tf.contrib.layers.variance_scaling_initializer
    elif init_type == 'xavier_initializer':
        return tf.contrib.layers.xavier_initializer
    elif init_type == 'xavier_initializer_conv2d':
        return tf.contrib.layers.xavier_initializer_conv2d
    elif init_type == 'zeros_initializer':
        return tf.zeros_initializer


def get_shape(tensors):
    static_shapes = tensors.get_shape().as_list()
    dynamic_shape = tf.unstack(tf.shape(tensors))
    dims = [s[1] if s[0] is None else s[0] for s in zip(static_shapes, dynamic_shape)]
    return dims


def batch_normalization(tensor, training=False, epsilon=0.001, momentum=0.9, fused_batch_norm=False, name=None):
    """Performs batch normalization on given 4-D tensor.

    The features are assumed to be in NHWC format. Note that you need to
    run UPDATE_OPS in order for this function to perform correctly, e.g.:

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      train_op = optimizer.minimize(loss)

    Based on: https://arxiv.org/abs/1502.03167
    """
    with tf.variable_scope(name, default_name='batch_normal'):
        shapes = get_shape(tensor)
        channels = shapes[-1]
        axes = list(range(len(shapes)-1))

        beta = tf.get_variable('beta', channels, initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma', channels, initializer=tf.zeros_initializer())

        avg_mean = tf.get_variable('avg_mean', channels, initializer=tf.zeros_initializer(), trainable=False)
        avg_variance = tf.get_variable('avg_variance', channels, initializer=tf.ones_initializer(), trainable=False)

        if training:
            if fused_batch_norm:
                mean, variance = None, None
            else:
                mean, variance = tf.nn.moments(tensor, axes=axes)
        else:
            mean, variance = avg_mean, avg_variance

        if fused_batch_norm:
            tensor, mean, variance = tf.nn.fused_batch_norm(tensor, scale=gamma, offset=beta, mean=mean,
                                                variance=variance, epsilon=epsilon, is_training=training)
        else:
            tensor = tf.nn.batch_normalization(tensor, mean, variance, beta, gamma, epsilon)
        if training:
            update_mean = tf.assign(avg_mean, avg_mean * momentum + mean * (1.0 - momentum))
            update_variance = tf.assign(avg_variance, avg_variance * momentum + variance * (1.0 - momentum))
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_variance)
        return tensor
