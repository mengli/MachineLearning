from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

VGG16_NPY_PATH = 'vgg16.npy'
WD = 5e-4

data_dict = np.load(VGG16_NPY_PATH, encoding='latin1').item()


def activation_summary(var):
    tensor_name = var.op.name
    tf.summary.histogram(tensor_name + '/activations', var)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(var))


def variable_summaries(var):
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar(name + '/sttdev', stddev)
            tf.summary.scalar(name + '/max', tf.reduce_max(var))
            tf.summary.scalar(name + '/min', tf.reduce_min(var))
            tf.summary.histogram(name, var)


def load_conv_filter(name):
    init = tf.constant_initializer(value=data_dict[name][0],
                                   dtype=tf.float32)
    shape = data_dict[name][0].shape
    var = tf.get_variable(name=name + "_weight", initializer=init, shape=shape)
    if not tf.get_variable_scope().reuse:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), WD, name=name + '_weight_decay')
        tf.add_to_collection('losses', weight_decay)
    variable_summaries(var)
    return var


def get_conv_filter(name, shape):
    init = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    var = tf.get_variable(name=name + "_weight", initializer=init)
    weight_decay = tf.multiply(tf.nn.l2_loss(var), WD, name=name + '_weight_decay')
    tf.add_to_collection('losses', weight_decay)
    variable_summaries(var)
    return var


def load_conv_bias(name):
    bias_wights = data_dict[name][1]
    shape = data_dict[name][1].shape
    init = tf.constant_initializer(value=bias_wights,
                                   dtype=tf.float32)
    var = tf.get_variable(name=name + "_bias", initializer=init, shape=shape)
    variable_summaries(var)
    return var


def get_conv_bias(name, shape):
    init = tf.constant(0.0, shape=shape)
    var = tf.get_variable(name=name + "_bias", initializer=init)
    variable_summaries(var)
    return var


def conv2d(bottom, weight):
    return tf.nn.conv2d(bottom, weight, strides=[1, 1, 1, 1], padding='SAME')


def batch_norm_layer(bottom, is_training, scope):
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(bottom,
                                                        is_training=True,
                                                        center=False,
                                                        scope=scope+"_bn"),
                   lambda: tf.contrib.layers.batch_norm(bottom,
                                                        is_training=False,
                                                        center=False,
                                                        scope=scope+"_bn",
                                                        reuse=True))


def conv_layer_with_bn(bottom=None, is_training=True, shape=None, name=None):
    with tf.variable_scope(name) as scope:
        if shape:
            weight = get_conv_filter(name, shape)
            bias = get_conv_bias(name, [shape[3]])
        else:
            weight = load_conv_filter(name)
            bias = load_conv_bias(name)
        conv = tf.nn.bias_add(conv2d(bottom, weight), bias)
        conv = batch_norm_layer(conv, is_training, scope.name)
        conv = tf.nn.relu(conv, name="relu")
        activation_summary(conv)
        return conv


def max_pool_with_argmax(bottom):
    with tf.name_scope('max_pool_arg_max'):
        with tf.device('/gpu:0'):
            _, indices = tf.nn.max_pool_with_argmax(
                bottom,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME')
        indices = tf.stop_gradient(indices)
        bottom = tf.nn.max_pool(bottom,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')
        return bottom, indices


def max_unpool_with_argmax(bottom, mask, output_shape=None):
    with tf.name_scope('max_unpool_with_argmax'):
        ksize = [1, 2, 2, 1]
        input_shape = bottom.get_shape().as_list()
        #  calculation new shape
        if output_shape is None:
            output_shape = (input_shape[0],
                            input_shape[1] * ksize[1],
                            input_shape[2] * ksize[2],
                            input_shape[3])
        # calculation indices for batch, height, width and feature maps
        one_like_mask = tf.ones_like(mask)
        batch_range = tf.reshape(tf.range(output_shape[0],
                                          dtype=tf.int64),
                                 shape=[input_shape[0], 1, 1, 1])
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
        feature_range = tf.range(output_shape[3], dtype=tf.int64)
        f = one_like_mask * feature_range
        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(bottom)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(bottom, [updates_size])
        return tf.scatter_nd(indices, values, output_shape)


def inference(images, is_training, num_classes):
    training = tf.equal(is_training, tf.constant(True))
    conv1_1 = conv_layer_with_bn(bottom=images, is_training=training, name="conv1_1")
    conv1_2 = conv_layer_with_bn(bottom=conv1_1, is_training=training, name="conv1_2")
    pool1, pool1_indices = max_pool_with_argmax(conv1_2)

    print("pool1: ", pool1.shape)

    conv2_1 = conv_layer_with_bn(bottom=pool1, is_training=training, name="conv2_1")
    conv2_2 = conv_layer_with_bn(bottom=conv2_1, is_training=training, name="conv2_2")
    pool2, pool2_indices = max_pool_with_argmax(conv2_2)

    print("pool2: ", pool2.shape)

    conv3_1 = conv_layer_with_bn(bottom=pool2, is_training=training, name="conv3_1")
    conv3_2 = conv_layer_with_bn(bottom=conv3_1, is_training=training, name="conv3_2")
    conv3_3 = conv_layer_with_bn(bottom=conv3_2, is_training=training, name="conv3_3")
    pool3, pool3_indices = max_pool_with_argmax(conv3_3)

    print("pool3: ", pool3.shape)

    conv4_1 = conv_layer_with_bn(bottom=pool3, is_training=training, name="conv4_1")
    conv4_2 = conv_layer_with_bn(bottom=conv4_1, is_training=training, name="conv4_2")
    conv4_3 = conv_layer_with_bn(bottom=conv4_2, is_training=training, name="conv4_3")
    pool4, pool4_indices = max_pool_with_argmax(conv4_3)

    print("pool4: ", pool4.shape)

    conv5_1 = conv_layer_with_bn(bottom=pool4, is_training=training, name="conv5_1")
    conv5_2 = conv_layer_with_bn(bottom=conv5_1, is_training=training, name="conv5_2")
    conv5_3 = conv_layer_with_bn(bottom=conv5_2, is_training=training, name="conv5_3")
    pool5, pool5_indices = max_pool_with_argmax(conv5_3)

    print("pool5: ", pool5.shape)

    # End of encoders
    # start of decoders

    up_sample_5 = max_unpool_with_argmax(pool5,
                                         pool5_indices,
                                         output_shape=conv5_3.shape)
    up_conv5 = conv_layer_with_bn(bottom=up_sample_5,
                                  shape=[3, 3, 512, 512],
                                  is_training=training,
                                  name="up_conv5")

    print("up_conv5: ", up_conv5.shape)

    up_sample_4 = max_unpool_with_argmax(up_conv5,
                                         pool4_indices,
                                         output_shape=conv4_3.shape)
    up_conv4 = conv_layer_with_bn(bottom=up_sample_4,
                                  shape=[3, 3, 512, 256],
                                  is_training=training,
                                  name="up_conv4")

    print("up_conv4: ", up_conv4.shape)

    up_sample_3 = max_unpool_with_argmax(up_conv4,
                                         pool3_indices,
                                         output_shape=conv3_3.shape)
    up_conv3 = conv_layer_with_bn(bottom=up_sample_3,
                                  shape=[3, 3, 256, 128],
                                  is_training=training,
                                  name="up_conv3")

    print("up_conv3: ", up_conv3.shape)

    up_sample_2 = max_unpool_with_argmax(up_conv3,
                                         pool2_indices,
                                         output_shape=conv2_2.shape)
    up_conv2 = conv_layer_with_bn(bottom=up_sample_2,
                                  shape=[3, 3, 128, 64],
                                  is_training=training,
                                  name="up_conv2")

    print("up_conv2: ", up_conv2.shape)

    up_sample_1 = max_unpool_with_argmax(up_conv2,
                                         pool1_indices,
                                         output_shape=conv1_2.shape)
    logits = conv_layer_with_bn(bottom=up_sample_1,
                                shape=[3, 3, 64, num_classes],
                                is_training=training,
                                name="up_conv1")

    print("logits: ", logits.shape)
    tf.add_to_collection("logits", logits)

    return logits
