import input_pipline_builder
import tensorflow as tf


if __name__ == '__main__':
    input_pipline, input_config = input_pipline_builder.builder('/home/aurora/workspaces/PycharmProjects/object_detection_models/'
                                                  'config_files/cifar10_input.config')
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