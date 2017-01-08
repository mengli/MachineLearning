"""A Variational Autoencoders for CIFAR-10.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cifar
import tensorflow as tf

EPOCH = 5000
BATCH_SIZE = 64
LATENT_VAR_NUM = 256


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, name=name, stddev=0.02, dtype=tf.float32)
    return tf.Variable(initial, name)


def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape, name=name, dtype=tf.float32)
    return tf.Variable(initial, name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


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
    ch = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, ch]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, ch]))

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
        conv1 = conv_layer("conv_layer1", input_images, 5, 3, 16)
        conv2 = conv_layer("conv_layer2", conv1, 5, 16, 32)

        conv2_flat = tf.reshape(conv2, [-1, 8 * 8 * 32])
        tf.summary.histogram("conv2_flat", conv2_flat)

        z_mean = fc_layer("w_mean", conv2_flat, 8 * 8 * 32, LATENT_VAR_NUM)
        z_stddev = fc_layer("w_stddev", conv2_flat, 8 * 8 * 32, LATENT_VAR_NUM)

        tf.summary.histogram("z_mean", z_mean)
        tf.summary.histogram("z_stddev", z_stddev)

        return z_mean, z_stddev

# decoder
def generation(z):
    with tf.variable_scope("generation"):
        z_develop = fc_layer('z_matrix', z, LATENT_VAR_NUM, 8 * 8 * 32)
        z_matrix = tf.nn.relu(tf.reshape(z_develop, [BATCH_SIZE, 8, 8, 32]))
        tf.summary.histogram('z_matrix', z_matrix)
        h1 = tf.nn.relu(conv_transpose("g_h1", z_matrix, 32, 16, [BATCH_SIZE, 16, 16, 16]))
        tf.summary.histogram('h1', h1)
        h2 = tf.nn.sigmoid(conv_transpose("g_h2", h1, 16, 3, [BATCH_SIZE, 32, 32, 3]))
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
    cifar10 = cifar.Cifar()
    cifar10.ReadDataSets(one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 3072])

    x_image = tf.transpose(tf.reshape(x, [-1, 3, 32, 32]), [0, 2, 3, 1])
    image_flat = tf.reshape(x_image, [BATCH_SIZE, 32 * 32 * 3])

    image_grid = put_kernels_on_grid(tf.transpose(tf.reshape(x, [-1, 3, 32, 32]), [2, 3, 1, 0]), (8, 8))
    tf.summary.image("images", image_grid, max_outputs=1)

    z_mean, z_stddev = recognition(x_image)

    samples = tf.random_normal([BATCH_SIZE, LATENT_VAR_NUM], 0, 1, dtype=tf.float32)
    guessed_z = tf.add(tf.multiply(samples, z_stddev), z_mean)

    generated_images = generation(guessed_z)

    generated_flat = tf.reshape(generated_images, [BATCH_SIZE, 32 * 32 * 3])
    tf.summary.histogram('generated_flat', generated_flat)

    generation_loss = -tf.reduce_sum(
        image_flat * tf.log(1e-8 + generated_flat) + (1 - image_flat) * tf.log(1e-8 + 1 - generated_flat), 1)
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
        batch = cifar10.train.next_batch(BATCH_SIZE)
        if i % 100 == 0:
            print("step %d" % i)
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0]})
        train_writer.add_summary(summary, i)

    print("Done")


if __name__ == '__main__':
    tf.app.run(main=main)
