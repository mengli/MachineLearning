"""A convolutional neural network for CIFAR-10 classification.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cifar
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial, 'weights')


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, 'biases')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(layer_name, input, in_dim, in_ch, out_dim, pooling=False):
    with tf.name_scope(layer_name):
        W_conv = weight_variable([in_dim, in_dim, in_ch, out_dim])
        b_conv = bias_variable([out_dim])
        tf.summary.histogram(layer_name + "/weights", W_conv)
        tf.summary.histogram(layer_name + "/biases", b_conv)
        if pooling:
            return max_pool_2x2(tf.nn.relu(conv2d(input, W_conv) + b_conv))
        else:
            return tf.nn.relu(conv2d(input, W_conv) + b_conv)
            # tf.summary.image("conv_layer1/images", tf.transpose(W_conv1, [3, 0, 1, 2]), max_outputs=8)


def fc_layer(layer_name, input, in_dim, out_dim, activation=True):
    with tf.name_scope(layer_name):
        W_fc = weight_variable([in_dim, out_dim])
        b_fc = bias_variable([out_dim])
        tf.summary.histogram(layer_name + "/weights", W_fc)
        tf.summary.histogram(layer_name + "/biases", b_fc)
        if activation:
            return tf.nn.relu(tf.matmul(input, W_fc) + b_fc)
        else:
            return tf.matmul(input, W_fc) + b_fc


def main(_):
    cifar10 = cifar.Cifar()
    cifar10.ReadDataSets(one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 3072])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.transpose(tf.reshape(x, [-1, 3, 32, 32]), [0, 2, 3, 1])

    h_pool1 = conv_layer("conv_layer1", x_image, 5, 3, 32, True)
    h_pool2 = conv_layer("conv_layer2", h_pool1, 5, 32, 64, True)
    h_conv3 = conv_layer("conv_layer3", h_pool2, 5, 64, 64)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 8 * 8 * 64])

    h_fc1 = fc_layer('fc_layer1', h_conv3_flat, 8 * 8 * 64, 384)
    h_fc2 = fc_layer('fc_layer2', h_fc1, 384, 192)
    y_conv = fc_layer('output', h_fc2, 192, 10, False)

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.0005
    learning_rate = tf.train.exponential_decay(
        starter_learning_rate, global_step, 100, 0.1, staircase=True)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    tf.summary.scalar('loss', cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train', sess.graph)

    sess.run(tf.global_variables_initializer())

    for i in range(100):
        batch = cifar10.train.next_batch(128)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: cifar10.test.images, y_: cifar10.test.labels})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1]})
        train_writer.add_summary(summary, i)

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: cifar10.test.images, y_: cifar10.test.labels}))


if __name__ == '__main__':
    tf.app.run(main=main)
