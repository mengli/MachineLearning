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
    initial = tf.truncated_normal(shape, stddev=0.05)
    var = tf.Variable(initial, 'weights')
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return var


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, 'biases')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(layer_name, input, in_dim, in_ch, out_dim, keep_prob=None, pooling=False, summary_conv=False):
    with tf.name_scope(layer_name):
        W_conv = weight_variable_with_decay([in_dim, in_dim, in_ch, out_dim], 0.0005)
        b_conv = bias_variable([out_dim])
        tf.summary.histogram("weights", W_conv)
        tf.summary.histogram("biases", b_conv)
        if summary_conv:
            # scale weights to [0 255] and convert to uint8
            #W_min = tf.reduce_min(W_conv)
            #W_max = tf.reduce_max(W_conv)
            #weights_0_to_1 = (W_conv - W_min) / (W_max - W_min)
            #weights_0_to_255_uint8 = tf.image.convert_image_dtype(weights_0_to_1, dtype=tf.uint8)
            tf.summary.image("weights", tf.transpose(W_conv, [3, 0, 1, 2]), max_outputs=12)
        if keep_prob != None:
            input = tf.nn.dropout(input, keep_prob)
        if pooling:
            return max_pool_2x2(tf.nn.relu(conv2d(input, W_conv) + b_conv))
        else:
            return tf.nn.relu(conv2d(input, W_conv) + b_conv)
            # tf.summary.image("conv_layer1/images", tf.transpose(W_conv1, [3, 0, 1, 2]), max_outputs=8)


def fc_layer(layer_name, input, in_dim, out_dim, keep_prob=None, activation=True):
    with tf.name_scope(layer_name):
        W_fc = weight_variable_with_decay([in_dim, out_dim], 0.0005)
        b_fc = bias_variable([out_dim])
        tf.summary.histogram("weights", W_fc)
        tf.summary.histogram("biases", b_fc)
        if keep_prob != None:
            input = tf.nn.dropout(input, keep_prob)
        if activation:
            return tf.nn.relu(tf.matmul(input, W_fc) + b_fc)
        else:
            return tf.matmul(input, W_fc) + b_fc


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
        learning_rate_2, global_step, EPOCH * 0.6, 0.5, staircase=True)
    tf.summary.scalar('learning_rate', decayed_learning_rate)
    return decayed_learning_rate


def main(_):
    cifar10 = cifar.Cifar()
    cifar10.ReadDataSets(one_hot=True)

    keep_prob = tf.placeholder(tf.float32)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 3072])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.transpose(tf.reshape(x, [-1, 3, 32, 32]), [0, 2, 3, 1])

    h_pool1 = conv_layer("conv_layer1", x_image, 5, 3, 64, pooling=True, summary_conv=True)
    h_pool2 = conv_layer("conv_layer2", h_pool1, 5, 64, 128, keep_prob=keep_prob, pooling=True)
    h_pool3 = conv_layer("conv_layer3", h_pool2, 5, 128, 256, keep_prob=keep_prob, pooling=True)
    h_pool4 = conv_layer("conv_layer4", h_pool3, 5, 256, 512, keep_prob=keep_prob, pooling=True)
    h_pool5 = conv_layer("conv_layer5", h_pool4, 5, 512, 512, keep_prob=keep_prob, pooling=True)

    h_conv3_flat = tf.reshape(h_pool5, [-1, 512])

    h_fc1 = fc_layer('fc_layer1', h_conv3_flat, 512, 512, keep_prob=keep_prob, activation=True)
    h_fc2 = fc_layer('fc_layer2', h_fc1, 512, 512, keep_prob=keep_prob, activation=True)
    y_conv = fc_layer('output', h_fc2, 512, 10, keep_prob=keep_prob, activation=False)

    global_step = tf.Variable(0, trainable=False)
    lr = learning_rate(global_step)

    total_loss = loss(y_conv, y_)
    train_step = tf.train.AdamOptimizer(lr).minimize(total_loss, global_step=global_step)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train', sess.graph)

    sess.run(tf.global_variables_initializer())

    for i in range(EPOCH):
        batch = cifar10.train.next_batch(BATCH_SIZE)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: cifar10.test.images, y_: cifar10.test.labels, keep_prob: 1})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.85})
        train_writer.add_summary(summary, i)

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: cifar10.test.images, y_: cifar10.test.labels, keep_prob: 1}))


if __name__ == '__main__':
    tf.app.run(main=main)
