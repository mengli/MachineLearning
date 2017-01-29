"""A Variational Autoencoders for CIFAR-10.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.utils import put_kernels_on_grid

EPOCH = 5000
BATCH_SIZE = 64
LATENT_VAR_NUM = 128
FLAGS = None


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, name=name, stddev=0.02, dtype=tf.float32)
    return tf.Variable(initial, name)


def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape, name=name, dtype=tf.float32)
    return tf.Variable(initial, name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(layer_name, input, in_dim, in_ch, out_dim):
    with tf.name_scope(layer_name):
        W_conv = weight_variable([in_dim, in_dim, in_ch, out_dim], "weights")
        b_conv = bias_variable([out_dim], "bias")
        tf.summary.histogram("weights", W_conv)
        tf.summary.histogram("biases", b_conv)
        activation = tf.nn.bias_add(conv2d(input, W_conv), b_conv)
        return lrelu(activation)


def conv_transpose(layer_name, x, in_ch, out_dim, output_shape):
    with tf.variable_scope(layer_name):
        W_conv = weight_variable([5, 5, out_dim, in_ch], "weights")
        return tf.nn.conv2d_transpose(x, W_conv, output_shape=output_shape, strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(layer_name, input, in_dim, out_dim):
    with tf.name_scope(layer_name):
        W_fc = weight_variable([in_dim, out_dim], "weights")
        b_fc = bias_variable([out_dim], "bias")
        return tf.nn.bias_add(tf.matmul(input, W_fc), b_fc)

# leaky reLu unit
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


# encoder
def recognition(input_images):
    with tf.variable_scope("recognition"):
        conv1 = conv_layer("conv_layer1", input_images, 5, 1, 16)
        conv2 = conv_layer("conv_layer2", conv1, 5, 16, 32)

        conv2_flat = tf.reshape(conv2, [-1, 7 * 7 * 32])
        tf.summary.histogram("conv2_flat", conv2_flat)

        z_mean = fc_layer("w_mean", conv2_flat, 7 * 7 * 32, LATENT_VAR_NUM)
        z_stddev = fc_layer("w_stddev", conv2_flat, 7 * 7 * 32, LATENT_VAR_NUM)

        tf.summary.histogram("z_mean", z_mean)
        tf.summary.histogram("z_stddev", z_stddev)

        return z_mean, z_stddev

# decoder
def generation(z):
    with tf.variable_scope("generation"):
        z_develop = fc_layer('z_matrix', z, LATENT_VAR_NUM, 7 * 7 * 32)
        z_matrix = tf.nn.relu(tf.reshape(z_develop, [BATCH_SIZE, 7, 7, 32]))
        tf.summary.histogram('z_matrix', z_matrix)
        h1 = tf.nn.relu(conv_transpose("g_h1", z_matrix, 32, 16, [BATCH_SIZE, 14, 14, 16]))
        tf.summary.histogram('h1', h1)
        h2 = tf.nn.sigmoid(conv_transpose("g_h2", h1, 16, 1, [BATCH_SIZE, 28, 28, 1]))
        tf.summary.histogram('h2', h2)

        kernel_grad_grid = put_kernels_on_grid(tf.transpose(h2, [1, 2, 3, 0]), (8, 8))
        tf.summary.image("gen_images", kernel_grad_grid, max_outputs=1)

        return h2


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
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    image_grid = put_kernels_on_grid(tf.transpose(x_image, [1, 2, 3, 0]), (8, 8))
    tf.summary.image("images", image_grid, max_outputs=1)

    z_mean, z_stddev = recognition(x_image)

    samples = tf.random_normal([BATCH_SIZE, LATENT_VAR_NUM], 0, 1, dtype=tf.float32)
    guessed_z = tf.add(tf.multiply(samples, z_stddev), z_mean)

    generated_images = generation(guessed_z)

    generated_flat = tf.reshape(generated_images, [BATCH_SIZE, 28 * 28 * 1])
    tf.summary.histogram('generated_flat', generated_flat)

    generation_loss = -tf.reduce_sum(
        x * tf.log(1e-8 + generated_flat) + (1 - x) * tf.log(1e-8 + 1 - generated_flat), 1)
    tf.summary.histogram('generation_loss', generation_loss)

    latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1, 1)

    cost = tf.reduce_mean(generation_loss + latent_loss)
    tf.summary.scalar('loss', cost)

    global_step = tf.Variable(0, trainable=False)
    lr = learning_rate(global_step)

    train_step = tf.train.AdamOptimizer(lr).minimize(cost)

    sess = tf.InteractiveSession()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train', sess.graph)

    sess.run(tf.global_variables_initializer())

    for i in range(EPOCH):
        batch = mnist.train.next_batch(BATCH_SIZE)
        if i % 100 == 0:
            print("step %d" % i)
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0]})
        train_writer.add_summary(summary, i)

    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
