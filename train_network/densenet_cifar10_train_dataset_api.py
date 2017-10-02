import add_path
import tensorflow as tf
from dense_net_utils import train_weight_deacy_with_batch_normal
import input_dataset_pipline_builder, densenet_builder
import logging
import time
from datetime import datetime
import os
import train_config_pb2
from google.protobuf import text_format

flags = tf.app.flags
flags.DEFINE_string('train_config_path', '../config_files/train_parameter.config', 'training config file path')
FLAGS = flags.FLAGS


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train_config = train_config_pb2.train_parameters()
    with tf.gfile.GFile(FLAGS.train_config_path) as fid:
        text_format.Merge(fid.read(), train_config)
    next_element, training_init_op, test_init_op, input_config = input_dataset_pipline_builder.builder(
            train_config.input_config_path)
    lr = tf.placeholder(name='leraning_rate', shape=[], dtype=tf.float32)
    images = tf.placeholder(dtype=tf.float32, shape=[None, input_config.image_height, input_config.image_width,
                                                     input_config.image_channel], name='input_image')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='input_label')
    accuracy_test = tf.placeholder(dtype=tf.float32, shape=[], name='test_dataset_accuracy')

    logit, net_config = densenet_builder.builder(inputs=images, scope_name='densenet_cifar10', phase='train',
                                                 config_path=train_config.densenet_config_path)
    predict = tf.nn.softmax(logit)
    loss, train_op, grads = train_weight_deacy_with_batch_normal(logit, labels, lr, 0.9, net_config.weight_deacy)
    loss_summary = tf.summary.scalar('loss_val', loss)
    lr_summary = tf.summary.scalar('learning_rate', lr)
    acc_summary = tf.summary.scalar('test_accuracy', accuracy_test)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict, axis=1), labels), tf.float32))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    summary = tf.summary.FileWriter(graph=sess.graph, logdir=train_config.summary_save_path)
    # summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    train_start_time = time.time()
    learning_rate_val = train_config.learning_rate
    step = 0
    for i in range(train_config.epoch):
        if i == 150:
            learning_rate_val = 0.01
        elif i == 225:
            learning_rate_val = 0.001

        # Initialize an iterator over the training dataset.
        sess.run(training_init_op)
        while True:
            try:
                batch_images, batch_labels = sess.run(next_element)
                start_time = time.time()
                _, loss_val = sess.run([train_op, loss], feed_dict={images: batch_images, labels: batch_labels,
                                                                    lr: learning_rate_val})
                duration = time.time() - start_time
                step += 1
                if step % 10 == 0:
                    num_examples_per_step = input_config.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)
                    format_str = ('%s: epoch %d, step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), i, step, loss_val,
                                        examples_per_sec, sec_per_batch))

                if step % train_config.summary_step == 0:
                    loss_summary_val, lr_summary_val = sess.run([loss_summary, lr_summary], feed_dict={images: batch_images, labels: batch_labels,
                                                                  lr: learning_rate_val})
                    summary.add_summary(loss_summary_val, step)
                    summary.add_summary(lr_summary_val, step)

                if step % train_config.test_save_step == 0:
                    checkpoint_path = os.path.join(train_config.model_save_path, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
            except tf.errors.OutOfRangeError:
                break

        sess.run(test_init_op)
        counter = 0
        total_accuracy = 0.0
        while True:
            try:
                batch_images_test, batch_labels_test = sess.run(next_element)
                accuracy_val = sess.run(accuracy, feed_dict={images: batch_images_test, labels: batch_labels_test})
                counter += 1
                total_accuracy += accuracy_val
                # print(total_accuracy)
            except tf.errors.OutOfRangeError:
                test_accuracy_val = total_accuracy/counter
                test_accuracy_val_summary = sess.run(acc_summary, feed_dict={accuracy_test:test_accuracy_val})
                summary.add_summary(test_accuracy_val_summary, i)
                print('test data accuracy in epoch %d is %.2f' % (i, total_accuracy/counter))
                break
