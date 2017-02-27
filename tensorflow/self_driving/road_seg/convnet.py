"""A full convolutional neural network for road segmentation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

EPOCH = 1000
BATCH_SIZE = 1
N_cl = 2


def weight_variable_with_decay(shape, wd):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01, dtype=tf.float32), 'weights')


def bias_variable(shape):
    return tf.Variable(tf.constant(0.0, shape=shape, dtype=tf.float32), 'biases')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def deconv2d(x, W):
    x_shape = tf.shape(x)
    output_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    return tf.pack(output_list)


def unpool_2x2(x, raveled_argmax, out_shape):
    argmax = unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
    output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

    height = tf.shape(output)[0]
    width = tf.shape(output)[1]
    channels = tf.shape(output)[2]

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

    t2 = tf.squeeze(argmax)
    t2 = tf.pack((t2[0], t2[1]), axis=0)
    t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

    t = tf.concat(3, [t2, t1])
    indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

    x1 = tf.squeeze(x)
    x1 = tf.reshape(x1, [-1, channels])
    x1 = tf.transpose(x1, perm=[1, 0])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
    return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)


def conv_layer(layer_name, input, filter_w, filter_h, in_ch, out_dim):
    with tf.name_scope(layer_name):
        # Initialize weights and bias
        W_conv = weight_variable_with_decay([filter_w, filter_h, in_ch, out_dim], 0.004)
        b_conv = bias_variable([out_dim])

        # Log weights and bias
        tf.summary.histogram("weights", W_conv)
        tf.summary.histogram("biases", b_conv)

        return tf.nn.relu(tf.nn.bias_add(conv2d(input, W_conv), b_conv))


def deconv_layer(layer_name, input, filter_w, filter_h, in_ch, out_dim):
    with tf.name_scope(layer_name):
        # Initialize weights and bias
        W_conv = weight_variable_with_decay([filter_w, filter_h, in_ch, out_dim], 0.004)
        b_conv = bias_variable([out_dim])

        # Log weights and bias
        tf.summary.histogram("weights", W_conv)
        tf.summary.histogram("biases", b_conv)

        return tf.nn.bias_add(deconv2d(input, W_conv), b_conv)


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
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='x_entropy'))


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
    kitti_data = kitti.Kitti()
    kitti_data.ReadDataSets()

    # Create the model
    x = tf.placeholder(tf.float32, [None, 3, 1280, 384])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 1242, 375, N_cl])

    x_image = tf.transpose(x, [0, 2, 3, 1])

    tf.summary.image("images", x_image, max_outputs=1)

    conv1_1 = conv_layer("conv_layer1_1", x_image, 3, 3, 3, 64)
    conv1_2 = conv_layer("conv_layer1_2", conv1_1, 3, 3, 64, 64)
    pool1, pool_1_argmax = max_pool_2x2(conv1_2)  # 640 x 192

    conv2_1 = conv_layer("conv_layer2_1", pool1, 3, 3, 64, 128)
    conv2_2 = conv_layer("conv_layer2_2", conv2_1, 3, 3, 128, 128)
    pool2, pool_2_argmax = max_pool_2x2(conv2_2)  # 320 x 96

    conv3_1 = conv_layer("conv_layer3_1", pool2, 3, 3, 128, 256)
    conv3_2 = conv_layer("conv_layer3_2", conv3_1, 3, 3, 256, 256)
    conv3_3 = conv_layer("conv_layer3_2", conv3_2, 3, 3, 256, 256)
    pool3, pool_3_argmax = max_pool_2x2(conv3_3)  # 160 x 48

    conv4_1 = conv_layer("conv_layer4_1", pool3, 3, 3, 256, 512)
    conv4_2 = conv_layer("conv_layer4_2", conv4_1, 3, 3, 512, 512)
    conv4_3 = conv_layer("conv_layer4_2", conv4_2, 3, 3, 512, 512)
    pool4, pool_4_argmax = max_pool_2x2(conv4_3)  # 80 x 24

    conv5_1 = conv_layer("conv_layer5_1", pool4, 3, 3, 512, 512)
    conv5_2 = conv_layer("conv_layer5_2", conv5_1, 3, 3, 512, 512)
    conv5_3 = conv_layer("conv_layer5_2", conv5_2, 3, 3, 512, 512)
    pool5, pool_5_argmax = max_pool_2x2(conv5_3)  # 40 x 12

    fc_conv6 = conv_layer("fc_conv_layer6", pool5, 40, 12, 512, 4096)
    fc_conv7 = conv_layer("fc_conv_layer7", fc_conv6, 1, 1, 4096, 4096)

    fc7_deconv = deconv_layer("fc7_deconv", fc_conv7, 40, 12, 4096, 512)

    unpool5 = unpool_2x2(fc7_deconv, pool_5_argmax, tf.shape(conv5_3))
    deconv5_3 = deconv_layer("deconv_layer5_3", unpool5, 3, 3, 512, 512)
    deconv5_2 = deconv_layer("deconv_layer5_2", deconv5_3, 3, 3, 512, 512)
    deconv5_1 = deconv_layer("deconv_layer5_1", deconv5_2, 3, 3, 512, 512)

    unpool4 = unpool_2x2(deconv5_1, pool_4_argmax, tf.shape(conv4_3))
    deconv4_3 = deconv_layer("deconv_layer4_3", unpool4, 3, 3, 512, 512)
    deconv4_2 = deconv_layer("deconv_layer4_2", deconv4_3, 3, 3, 512, 512)
    deconv4_1 = deconv_layer("deconv_layer4_1", deconv4_2, 3, 3, 512, 256)

    unpool3 = unpool_2x2(deconv4_1, pool_3_argmax, tf.shape(conv3_3))
    deconv3_3 = deconv_layer("deconv_layer3_3", unpool3, 3, 3, 256, 256)
    deconv3_2 = deconv_layer("deconv_layer3_2", deconv3_3, 3, 3, 256, 256)
    deconv3_1 = deconv_layer("deconv_layer3_1", deconv3_2, 3, 3, 256, 128)

    unpool2 = unpool_2x2(deconv3_1, pool_2_argmax, tf.shape(conv2_2))
    deconv2_2 = deconv_layer("deconv_layer2_2", unpool2, 3, 3, 128, 128)
    deconv2_1 = deconv_layer("deconv_layer2_1", deconv2_2, 3, 3, 128, 64)

    unpool1 = unpool_2x2(deconv2_1, pool_1_argmax, tf.shape(conv1_2))
    deconv1_2 = deconv_layer("deconv_layer1_2", unpool1, 3, 3, 64, 64)
    deconv1_1 = deconv_layer("deconv_layer1_1", deconv1_2, 3, 3, 64, 32)

    score_1 = deconv_layer("score_1", deconv1_1, 3, 3, 32, N_cl)

    logits = tf.reshape(score_1, [-1, N_cl])
    labels = tf.reshape(y_, [-1, N_cl])
    total_loss = loss(logits, labels)

    global_step = tf.Variable(0, trainable=False)
    lr = learning_rate(global_step)
    optimizer = tf.train.AdamOptimizer(lr).minimize(total_loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.InteractiveSession(config=config)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train', sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(EPOCH):
        batch = kitti_data.train.next_batch(BATCH_SIZE)
        if i % 100 == 0:
            test_batch = kitti_data.train.next_batch(BATCH_SIZE)
            test_accuracy = accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1]})
            print("step %d, test accuracy %g" % (i, test_accuracy))
            saver.save(sess, './checkpoints/roadseg_model', global_step=i)
        summary, _ = sess.run([merged, optimizer], feed_dict={x: batch[0], y_: batch[1]})
        train_writer.add_summary(summary, i)

    test_batch = kitti_data.train.next_batch(BATCH_SIZE)
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: test_batch[0], y_: test_batch[1]}))


if __name__ == '__main__':
    tf.app.run(main=main)
