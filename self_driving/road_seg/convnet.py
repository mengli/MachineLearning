"""A full convolutional neural network for road segmentation.

nohup python -u -m self_driving.road_seg.convnet > self_driving/road_seg/output.txt 2>&1 &

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
from utils import kitti
from self_driving.road_seg import fcn8_vgg
import scipy as scp
import scipy.misc
import matplotlib as mpl
import matplotlib.cm

EPOCH = 5000
N_cl = 2
UU_TRAIN_SET_SIZE = 98 - 9
UU_TEST_SET_SIZE = 9


def _compute_cross_entropy_mean(labels, softmax):
    cross_entropy = -tf.reduce_sum(
        tf.multiply(labels * tf.log(softmax), [1, 1]), reduction_indices=[1])
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
    true_labels = tf.to_float(tf.reshape(labels, (-1, 2)))[:, 1]
    pred = tf.to_float(tf.reshape(logits, [-1]))

    true_positives = tf.reduce_sum(pred * true_labels)
    false_positives = tf.reduce_sum(pred * (1 - true_labels))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / tf.reduce_sum(labels)

    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score, precision, recall


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


def save_output(index, training_image, prediction, label):
    prediction_label = 1 - prediction[0]
    output_image = copy.copy(training_image)
    # Save prediction
    up_color = color_image(prediction[0], 2)
    scp.misc.imsave('output/decision_%d.png' % index, up_color)
    # Merge true positive with training images' green channel
    true_positive = prediction_label * label[..., 0][0]
    merge_green = (1 - true_positive) * training_image[..., 1] + true_positive * 255
    output_image[..., 1] = merge_green
    # Merge false positive with training images' red channel
    false_positive = prediction_label * label[..., 1][0]
    merge_red = (1 - false_positive) * training_image[..., 0] + false_positive * 255
    output_image[..., 0] = merge_red
    # Merge false negative with training images' blue channel
    false_negative = (1 - prediction_label) * label[..., 0][0]
    merge_blue = (1 - false_negative) * training_image[..., 2] + false_negative * 255
    output_image[..., 2] = merge_blue
    # Save images
    scp.misc.imsave('merge/decision_%d.png' % index, output_image)


def main(_):
    kitti_data = kitti.Kitti()

    x_image = tf.placeholder(tf.float32, [1, None, None, 3])
    y_ = tf.placeholder(tf.float32, [1, None, None, N_cl])

    tf.summary.image("images", x_image, max_outputs=1)

    vgg_fcn = fcn8_vgg.FCN8VGG(vgg16_npy_path="data/vgg16.npy")
    vgg_fcn.build(x_image, debug=True, num_classes=N_cl)

    losses = loss(vgg_fcn.upscore32, y_)
    f1, precision, recall = f1_score(vgg_fcn.pred_up, y_)
    total_loss = losses['total_loss']
    tf.summary.scalar("Loss", total_loss)
    tf.summary.scalar("F1 Score", f1)
    tf.summary.scalar("Precision", precision)
    tf.summary.scalar("Recall", recall)

    global_step = tf.Variable(0, trainable=False)
    lr = learning_rate(global_step)
    optimizer = tf.train.AdamOptimizer(lr)
    grads_and_vars = optimizer.compute_gradients(total_loss)

    grads, tvars = zip(*grads_and_vars)
    clipped_grads, norm = tf.clip_by_global_norm(grads, 1.0)
    grads_and_vars = zip(clipped_grads, tvars)

    train_step = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    sess = tf.InteractiveSession()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train', sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(EPOCH):
        print("step %d" % i)
        t_img, t_label = kitti_data.next_batch(i % UU_TRAIN_SET_SIZE)
        pred, _ = sess.run([vgg_fcn.pred_up, train_step],
                           feed_dict={x_image: t_img, y_: t_label})
        if i % 5 == 0:
            for test_index in range(UU_TEST_SET_SIZE):
                test_img, test_label = kitti_data.next_batch(test_index + UU_TRAIN_SET_SIZE)
                pred, summary = sess.run([vgg_fcn.pred_up, merged],
                                         feed_dict={x_image: test_img, y_: test_label})
                save_output(test_index + UU_TRAIN_SET_SIZE, test_img[0], pred, test_label)
                train_writer.add_summary(summary, i)


if __name__ == '__main__':
    tf.app.run(main=main)