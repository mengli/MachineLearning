"""A convolutional neural network for CIFAR-10 classification.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cifar
import tensorflow as tf

EPOCH = 5000
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


def put_kernels_on_grid(kernel, (grid_Y, grid_X), pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
    '''
    # pad X and Y
    x1 = tf.pad(kernel, tf.constant([[pad, 0], [pad, 0], [0, 0], [0, 0]]))

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + pad
    X = kernel.get_shape()[1] + pad

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, 3]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, 3]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 1]
    x_min = tf.reduce_min(x7)
    x_max = tf.reduce_max(x7)
    x8 = (x7 - x_min) / (x_max - x_min)

    return x8


def conv_layer(layer_name, input, in_dim, in_ch, out_dim, summary_conv=False):
    with tf.name_scope(layer_name):
        W_conv = weight_variable_with_decay([in_dim, in_dim, in_ch, out_dim], 0.004)
        b_conv = bias_variable([out_dim])
        tf.summary.histogram("weights", W_conv)
        tf.summary.histogram("biases", b_conv)
        if summary_conv:
            kernel_grid = put_kernels_on_grid(W_conv, (8, 8))
            tf.summary.image("kernel", kernel_grid, max_outputs=1)
            #tf.summary.image("kernel", tf.transpose(W_conv, [3, 0, 1, 2]), max_outputs=64)
        activation = tf.nn.bias_add(conv2d(input, W_conv), b_conv)
        activation_dim = tf.shape(activation)[1]
        activation_sample = tf.slice(activation, [0, 0, 0, 0], [1, activation_dim, activation_dim, out_dim])
        tf.summary.image("activatins", tf.transpose(activation_sample, [3, 1, 2, 0]), max_outputs=out_dim)
        pool = max_pool_2x2(tf.nn.relu(activation))
        return tf.nn.lrn(pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')


def fc_layer(layer_name, input, in_dim, out_dim, activation=True):
    with tf.name_scope(layer_name):
        W_fc = weight_variable_with_decay([in_dim, out_dim], 0.004)
        b_fc = bias_variable([out_dim])
        tf.summary.histogram("weights", W_fc)
        tf.summary.histogram("biases", b_fc)
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

    h_pool1 = conv_layer("conv_layer1", x_image, 5, 3, 64, summary_conv=True)
    h_pool2 = conv_layer("conv_layer2", h_pool1, 5, 64, 64)
    #h_pool3 = conv_layer("conv_layer3", h_pool2, 5, 128, 256)
    #h_pool4 = conv_layer("conv_layer4", h_pool3, 5, 256, 512)
    #h_pool5 = conv_layer("conv_layer5", h_pool4, 5, 512, 512)

    h_conv3_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])

    h_fc1 = fc_layer('fc_layer1', h_conv3_flat, 8 * 8 * 64, 384, activation=True)
    h_fc2 = fc_layer('fc_layer2', h_fc1, 384, 192, activation=True)
    y_conv = fc_layer('fc_layer3', h_fc2, 192, 10, activation=False)

    global_step = tf.Variable(0, trainable=False)
    lr = learning_rate(global_step)

    total_loss = loss(y_conv, y_)
    #train_step = tf.train.AdamOptimizer(lr).minimize(total_loss, global_step=global_step)
    optimizer = tf.train.AdamOptimizer(lr)
    grads_and_vars = optimizer.compute_gradients(total_loss)
    with tf.name_scope("conv_layer1_grad"):
        kernel_grad_grid = put_kernels_on_grid(grads_and_vars[0][0], (8, 8))
        tf.summary.image("weight_grad", kernel_grad_grid, max_outputs=1)
        #tf.summary.image("weight_grad", tf.transpose(grads_and_vars[0][0], [3, 0, 1, 2]), max_outputs=64)

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
            #train_batch = cifar10.train.next_batch(BATCH_SIZE)
            #train_accuracy = accuracy.eval(feed_dict={x: train_batch[0], y_: train_batch[1]})
            #print("step %d, training accuracy %g" % (i, train_accuracy))
            test_accuracy = accuracy.eval(feed_dict={x: cifar10.test.images, y_: cifar10.test.labels})
            print("step %d, test accuracy %g" % (i, test_accuracy))
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1]})
        train_writer.add_summary(summary, i)

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: cifar10.test.images, y_: cifar10.test.labels}))


if __name__ == '__main__':
    tf.app.run(main=main)
