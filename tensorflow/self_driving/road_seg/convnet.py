"""A full convolutional neural network for road segmentation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils import kitti
from self_driving.road_seg import fcn8_vgg
import scipy as scp
import scipy.misc
import matplotlib as mpl
import matplotlib.cm

EPOCH = 500
N_cl = 2
UU_TRAIN_SET_SIZE = 98


def _compute_cross_entropy_mean(labels, softmax):
    cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax), [1, 1]), reduction_indices=[1])
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return cross_entropy_mean


def loss(logits, labels):
    with tf.name_scope('loss'):
        labels = tf.to_float(tf.reshape(labels, (-1, 2)))
        logits = tf.reshape(logits, (-1, 2))
        epsilon = 1e-9
        softmax = tf.nn.softmax(logits) + epsilon

        cross_entropy_mean = _compute_cross_entropy_mean(labels, softmax)

        enc_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        dec_loss = tf.add_n(tf.get_collection('dec_losses'), name='total_loss')
        fc_loss = tf.add_n(tf.get_collection('fc_wlosses'), name='total_loss')
        weight_loss = enc_loss + dec_loss + fc_loss

        total_loss = cross_entropy_mean + weight_loss

        losses = {}
        losses['total_loss'] = total_loss
        losses['xentropy'] = cross_entropy_mean
        losses['weight_loss'] = weight_loss

    return losses


def f1_score(logits, labels):
    labels = tf.to_float(tf.reshape(labels, (-1, 2)))[:, 1]
    logits = tf.reshape(logits, (-1, 2))
    epsilon = 1e-9
    softmax = (tf.nn.softmax(logits) + epsilon)[:, 1]

    true_positives = tf.reduce_sum(softmax * labels)
    false_positives = tf.reduce_sum((1 - softmax) * labels)

    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / tf.reduce_sum(labels)

    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score


def learning_rate(global_step):
    starter_learning_rate = 1e-5
    learning_rate_1 = tf.train.exponential_decay(
        starter_learning_rate, global_step, EPOCH * 0.2, 0.1, staircase=True)
    learning_rate_2 = tf.train.exponential_decay(
        learning_rate_1, global_step, EPOCH * 0.4, 0.5, staircase=True)
    decayed_learning_rate = tf.train.exponential_decay(
        learning_rate_2, global_step, EPOCH * 0.6, 0.8, staircase=True)
    tf.summary.scalar('learning_rate', decayed_learning_rate)
    return decayed_learning_rate


def color_image(image, num_classes=20):
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))


def main(_):
    kitti_data = kitti.Kitti()

    x_image = tf.placeholder(tf.float32, [1, None, None, 3])
    y_ = tf.placeholder(tf.float32, [1, None, None, N_cl])

    tf.summary.image("images", x_image, max_outputs=1)

    vgg_fcn = fcn8_vgg.FCN8VGG(vgg16_npy_path="data/vgg16.npy")
    vgg_fcn.build(x_image, debug=True, num_classes=N_cl)

    losses = loss(vgg_fcn.upscore32, y_)
    f1 = f1_score(vgg_fcn.upscore32, y_)
    total_loss = losses['total_loss']
    tf.summary.scalar("Loss", total_loss)
    tf.summary.scalar("F1 Score", f1)

    global_step = tf.Variable(0, trainable=False)
    lr = learning_rate(global_step)
    optimizer = tf.train.AdamOptimizer(lr)
    grads_and_vars = optimizer.compute_gradients(total_loss)

    grads, tvars = zip(*grads_and_vars)
    clipped_grads, norm = tf.clip_by_global_norm(grads, 1.0)
    grads_and_vars = zip(clipped_grads, tvars)

    train_step = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=0.2)
    sess = tf.InteractiveSession()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train', sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(EPOCH):
        print("step %d" % i)
        t_img, t_label = kitti_data.next_batch()
        if i % 5 == 0:
            saver.save(sess, './checkpoints/roadseg_model', global_step=i)
        up_s, summary, _ = sess.run([vgg_fcn.pred_up, merged, train_step], feed_dict={x_image: t_img, y_: t_label})
        up_color = color_image(up_s[0], 2)
        scp.misc.imsave('output/decision_%d.png' % (i % UU_TRAIN_SET_SIZE), up_color)
        merge_decision = t_img[0]
        merge_green = up_s[0] * merge_decision[..., 1] + (1 - up_s[0]) * 255
        merge_decision[..., 1] = merge_green
        scp.misc.imsave('merge/decision_%d.png' % (i % UU_TRAIN_SET_SIZE), merge_decision)
        train_writer.add_summary(summary, i)


if __name__ == '__main__':
    tf.app.run(main=main)
