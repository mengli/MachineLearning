"""A convolutional neural network for CIFAR-10 classification.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils import cifar
from utils.utils import put_kernels_on_grid

EPOCH = 36000
BATCH_SIZE = 128


def weight_variable_with_decay(shape, wd):
    initial = tf.truncated_normal(shape, stddev=0.05, dtype=tf.float32)
    var = tf.Variable(initial, 'weights')
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return var


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, 'biases')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(layer_name, input, in_dim, in_ch, out_dim, out_size, summary_conv=False):
    with tf.name_scope(layer_name):
        # Initialize weights and bias
        W_conv = weight_variable_with_decay([in_dim, in_dim, in_ch, out_dim], 0.004)
        b_conv = bias_variable([out_dim])

        # Log weights and bias
        tf.summary.histogram("weights", W_conv)
        tf.summary.histogram("biases", b_conv)

        # Draw weights in 8x8 grid for the first conv layer
        if summary_conv:
            kernel_grid = put_kernels_on_grid(W_conv, (8, 8))
            tf.summary.image("kernel", kernel_grid, max_outputs=1)

        # Draw conv activation in 8x8 grid
        activation = tf.nn.bias_add(conv2d(input, W_conv), b_conv)
        # Only draw the activation for the first image in a batch
        activation_sample = tf.slice(activation, [0, 0, 0, 0], [1, out_size, out_size, out_dim])
        activation_grid = put_kernels_on_grid(tf.transpose(activation_sample, [1, 2, 0, 3]), (8, 8))
        tf.summary.image("conv/activatins", activation_grid, max_outputs=1)

        # Draw relu activation in 8x8 grid
        activation = tf.nn.relu(activation)
        # Only draw the activation for the first image in a batch
        activation_sample = tf.slice(activation, [0, 0, 0, 0], [1, out_size, out_size, out_dim])
        activation_grid = put_kernels_on_grid(tf.transpose(activation_sample, [1, 2, 0, 3]), (8, 8))
        tf.summary.image("relu/activatins", activation_grid, max_outputs=1)

        # 2x2 max pooling
        pool = max_pool_2x2(activation)

        return tf.nn.lrn(pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')


def fc_layer(layer_name, input, in_dim, out_dim, activation=True):
    with tf.name_scope(layer_name):
        # Initialize weights and bias
        W_fc = weight_variable_with_decay([in_dim, out_dim], 0.004)
        b_fc = bias_variable([out_dim])

        # Log weights and bias
        tf.summary.histogram("weights", W_fc)
        tf.summary.histogram("biases", b_fc)

        # Shouldn't only apply activation function for the last fc layer
        if activation:
            return tf.nn.relu(tf.nn.bias_add(tf.matmul(input, W_fc), b_fc))
        else:
            return tf.nn.bias_add(tf.matmul(input, W_fc), b_fc)


def loss(logits, labels):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
    tf.add_to_collection('losses', cross_entropy)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    tf.summary.scalar('loss', total_loss)
    return total_loss


def learning_rate(global_step):
    starter_learning_rate = 0.001
    learning_rate_1 = tf.train.exponential_decay(
        starter_learning_rate, global_step, EPOCH * 0.2, 0.1, staircase=True)
    learning_rate_2 = tf.train.exponential_decay(
        learning_rate_1, global_step, EPOCH * 0.4, 0.5, staircase=True)
    decayed_learning_rate = tf.train.exponential_decay(
        learning_rate_2, global_step, EPOCH * 0.6, 0.8, staircase=True)
    tf.summary.scalar('learning_rate', decayed_learning_rate)
    return decayed_learning_rate


def main(_):
    cifar10 = cifar.Cifar()
    cifar10.ReadDataSets(one_hot=True)

    keep_prob = tf.placeholder(tf.float32)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 3, 32, 32])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.transpose(x, [0, 2, 3, 1])

    tf.summary.image("images", x_image, max_outputs=1)

    h_pool1 = conv_layer("conv_layer1", x_image, 5, 3, 64, 32, summary_conv=True)
    h_pool2 = conv_layer("conv_layer2", h_pool1, 5, 64, 64, 16)

    h_conv3_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])

    h_fc1 = fc_layer('fc_layer1', h_conv3_flat, 8 * 8 * 64, 384, activation=True)
    h_fc2 = fc_layer('fc_layer2', h_fc1, 384, 192, activation=True)
    y_conv = fc_layer('fc_layer3', h_fc2, 192, 10, activation=False)

    global_step = tf.Variable(0, trainable=False)
    lr = learning_rate(global_step)

    total_loss = loss(y_conv, y_)
    optimizer = tf.train.AdamOptimizer(lr)
    grads_and_vars = optimizer.compute_gradients(total_loss)
    with tf.name_scope("conv_layer1_grad"):
        kernel_grad_grid = put_kernels_on_grid(grads_and_vars[0][0], (8, 8))
        tf.summary.image("weight_grad", kernel_grad_grid, max_outputs=1)

    train_step = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train', sess.graph)

    sess.run(tf.global_variables_initializer())

    for i in range(EPOCH):
        batch = cifar10.train.next_batch(BATCH_SIZE)
        if i % 100 == 0:
            test_accuracy = accuracy.eval(feed_dict={x: cifar10.test.images, y_: cifar10.test.labels})
            print("step %d, test accuracy %g" % (i, test_accuracy))
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1]})
        train_writer.add_summary(summary, i)

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: cifar10.test.images, y_: cifar10.test.labels}))


if __name__ == '__main__':
    tf.app.run(main=main)
