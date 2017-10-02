import tensorflow as tf
import numpy as np
import logging
from net_utils import activation_summary, get_shape


def bottlenecks_b(k_size, scope, index, weight_initializer, intern_rate, bias=False, dropout=False, dropout_rate=0.5, phase='train', reuse=False):
    logging.info('#########################  bottlenecks_b  '+scope+'___'+str(index)+'   #############################')
    logging.info(tf.get_collection(scope+phase))
    inputs = tf.concat(tf.get_collection(scope+phase), axis=-1)
    shapes = get_shape(inputs)
    logging.info('input feature size is '+str(shapes))

    with tf.variable_scope('group'+str(index), reuse=False):
        # 1x1 conv2d weight and bias
        weights = tf.get_variable('weights1', shape=[1, 1, shapes[-1], intern_rate*k_size], dtype=tf.float32,
                            initializer=weight_initializer())
        logging.info(weights.op.name)

        # 3x3 conv2d weight and bias
        weights3x3 = tf.get_variable('weights3', dtype=tf.float32, shape=[3, 3, intern_rate*k_size, k_size],
                            initializer=weight_initializer())
        logging.info(weights3x3.op.name)

        # 1x1 conv layer
        # net = batch_normalization(inputs, training=phase == 'train')
        net = tf.contrib.layers.batch_norm(inputs, is_training=phase == 'train', scale=True, updates_collections=None)
        net = tf.nn.relu(net)
        activation_summary(net)
        net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding='SAME', name='conv2d_1')
        if bias:
            bias = tf.get_variable('bias1', shape=(k_size,), dtype=tf.float32,
                                   initializer=tf.zeros_initializer())
            net = tf.nn.bias_add(net, bias, name='bias_add_1')
        if dropout:
            if phase == 'train':
                net = tf.nn.dropout(net, keep_prob=dropout_rate)
            else:
                net = tf.nn.dropout(net, keep_prob=1.0)

        # net = batch_normalization(net, training=phase == 'train')
        net = tf.contrib.layers.batch_norm(net, is_training=phase == 'train', scale=True, updates_collections=None)
        net = tf.nn.relu(net)
        activation_summary(net)
        # 3x3 conv layer
        net = tf.nn.conv2d(net, weights3x3, strides=[1, 1, 1, 1], padding='SAME', name='conv2d_3')
        if bias:
            bias3x3 = tf.get_variable('bias3', dtype=tf.float32, shape=(k_size,),
                                      initializer=tf.zeros_initializer())
            net = tf.nn.bias_add(net, bias3x3, name='outputs')
        if dropout:
            if phase == 'train':
                net = tf.nn.dropout(net, keep_prob=dropout_rate)
            else:
                net = tf.nn.dropout(net, keep_prob=1.0)

        logging.info(net.op.name)
        tf.add_to_collection(scope+phase, net)
    return net


def bottlenecks(k_size, scope, index, weight_initializer, bias=False, dropout=False, dropout_rate=0.5, phase='train', reuse=False):
    logging.info('#########################  bottlenecks '+scope+'___'+str(index)+'   #############################')
    logging.info(tf.get_collection(scope+phase))
    # regularizer_term = tf.contrib.layers.l2_regularizer(scale=regularizer)
    inputs = tf.concat(tf.get_collection(scope+phase), axis=-1)
    shapes = get_shape(inputs)
    logging.info('input feature size is ' + str(shapes))

    with tf.variable_scope('group'+str(index), reuse=False):
        # 3x3 conv2d weight and bias
        weights3x3 = tf.get_variable('weights3', dtype=tf.float32, shape=[3, 3, shapes[-1], k_size],
                        initializer=weight_initializer())
        logging.info(weights3x3.op.name)

        # net = batch_normalization(inputs, training=phase == 'train')
        net = tf.contrib.layers.batch_norm(inputs, is_training=phase == 'train', scale=True, updates_collections=None)
        net = tf.nn.relu(net)
        activation_summary(net)
        # 3x3 conv layer
        net = tf.nn.conv2d(net, weights3x3, strides=[1, 1, 1, 1], padding='SAME', name='conv2d_3')
        if bias:
            bias3x3 = tf.get_variable('bias3', dtype=tf.float32, shape=(k_size,),
                                      initializer=tf.zeros_initializer())
            net = tf.nn.bias_add(net, bias3x3, name='outputs')
        if dropout:
            if phase == 'train':
                net = tf.nn.dropout(net, keep_prob=dropout_rate)
            else:
                net = tf.nn.dropout(net, keep_prob=1.0)
        logging.info(net.op.name)
        tf.add_to_collection(scope+phase, net)
    return net


def dense_block(net, block_nums, k_size, bottleneck, bias, scope, weight_initializer, intern_rate, dropout=False, dropout_rate=0.5, phase='train'):
    tf.add_to_collection(scope+phase, net)
    with tf.variable_scope(scope):
        for i in range(block_nums):
            if bottleneck:
                outputs = bottlenecks_b(k_size=k_size, scope=scope, index=i, bias=bias, dropout=dropout, dropout_rate=dropout_rate,
                                        phase=phase, weight_initializer=weight_initializer, intern_rate=intern_rate)
            else:
                outputs = bottlenecks(k_size=k_size, scope=scope, index=i, bias=bias, dropout=dropout, dropout_rate=dropout_rate,
                                        phase=phase, weight_initializer=weight_initializer)
    return outputs


def transition_layers(inputs, densenet_c, theta, bn_relu, bias, scope, phase, weight_initializer):
    logging.info('#########################   ' + scope + '  #############################')
    if densenet_c:
        assert (theta != 0 and theta != 1), 'theta and densenet_c is not consistance'
    else:
        assert theta == 1, 'theta should equals 1'
    _shapes = get_shape(inputs)
    logging.info(_shapes)
    output_features = _shapes[-1] if not densenet_c else np.floor(theta*_shapes[-1])
    logging.info('transition_layers densenet compression val is ' +str(densenet_c)+' output feature maps '+str(output_features))
    with tf.variable_scope(scope):
        weights = tf.get_variable('weights', shape=[1, 1, _shapes[-1], output_features], dtype=tf.float32,
                           initializer=weight_initializer())

        if bn_relu:
            # outputs = batch_normalization(inputs, training=phase == 'train')
            outputs = tf.contrib.layers.batch_norm(inputs, is_training=phase == 'train', scale=True, updates_collections=None)
            outputs = tf.nn.relu(outputs)
            activation_summary(outputs)
        else:
            outputs = inputs
        outputs = tf.nn.conv2d(outputs, weights, strides=[1, 1, 1, 1], padding='SAME', name='conv2d')
        if bias:
            bias = tf.get_variable('bias', shape=[output_features], dtype=tf.float32,
                                   initializer=tf.zeros_initializer())
            outputs = tf.nn.bias_add(outputs, bias, name='bias_add')
        outputs = tf.nn.avg_pool(outputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='outputs')
        logging.info('outputs shape equals: ')
        logging.info(get_shape(outputs))
    return outputs


def stack_blocks(inputs, blocks, k_size, boottleneck, densenet_c, theta, bn_relu, bias_add, dropout, dropout_rate, phase,
                 weights_initializer, intern_rate):
    for index, block_info in enumerate(blocks):
        outputs = dense_block(inputs, block_nums=block_info, k_size=k_size, bottleneck=boottleneck, dropout=dropout,
                              dropout_rate=dropout_rate, phase=phase, scope='dense_block'+str(index), bias=bias_add,
                              weight_initializer=weights_initializer, intern_rate=intern_rate)
        if not len(blocks)-1 == index:
            inputs = tf.concat(tf.get_collection('dense_block'+str(index)+phase), axis=-1)
            outputs = transition_layers(inputs, densenet_c=densenet_c, theta=theta, bn_relu=bn_relu,
              scope='trainsition_layers_'+str(index), bias=bias_add, phase=phase,weight_initializer=weights_initializer)
            inputs = outputs
    outputs = tf.concat(tf.get_collection('dense_block'+str(len(blocks)-1)+phase), axis=-1)
    return outputs


def train(logit, labels, learning_rate, nesterov_momentum):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))
    optimizer = tf.train.MomentumOptimizer(learning_rate, nesterov_momentum, use_nesterov=True)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss)
    return loss, train_op


def train_weight_deacy(logit, labels, learning_rate, nesterov_momentum, weight_deacy):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if ('beta' not in var.op.name) and ('gamma' not in var.op.name)])
    total_loss = loss + l2_loss*weight_deacy
    optimizer = tf.train.MomentumOptimizer(learning_rate, nesterov_momentum, use_nesterov=True)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(total_loss)
        grads = optimizer.compute_gradients(total_loss)
    return loss, train_op, grads


def train_weight_deacy_with_batch_normal(logit, labels, learning_rate, nesterov_momentum, weight_deacy):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    total_loss = loss + l2_loss*weight_deacy
    optimizer = tf.train.MomentumOptimizer(learning_rate, nesterov_momentum, use_nesterov=True)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(total_loss)
        grads = optimizer.compute_gradients(total_loss)
    return loss, train_op, grads


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # blocks = [(12, 6), (12, 12), (12, 24), (12, 16)]
    blocks = [(12, 3), (12, 6)]

    # test _get_shape method
    # inputs = tf.placeholder(name='inputs', dtype=tf.float32, shape=[None, 3])
    # b = tf.add(inputs, 1)
    np_inputs = np.random.normal(size=(1, 224, 224, 3))
    np_inputs = np_inputs.astype(dtype=np.float32)
    t_np_inputs = tf.convert_to_tensor(np_inputs, name='np2tensor')
    sess = tf.Session()
    net = dense_block(t_np_inputs, 6, 12, 'dense_block1')
    # net = stack_blocks(blocks, t_np_inputs)
    sess.run(tf.global_variables_initializer())
    sess.run(net)
