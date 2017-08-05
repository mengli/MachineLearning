"""Evaluate SegNet.

nohup python -u -m self_driving.segnet.evaluate_kitti > self_driving/segnet/output.txt 2>&1 &

"""

import os
import tensorflow as tf
from utils import kitti_segnet
from scipy import misc

LOG_DIR = 'backup/segnet_kitti'
EPOCH = 237
BATCH_SIZE = 1
IMAGE_HEIGHT = 375
IMAGE_WIDTH = 1242
NUM_CLASSES = 2

test_dir = "/usr/local/google/home/limeng/Downloads/kitti/data_road/testing/test.txt"

colors = [
    [255, 0, 255],
    [255, 0,   0],
]

def color_mask(tensor, color):
    return tf.reduce_all(tf.equal(tensor, color), 3)


def one_hot(labels):
    color_tensors = tf.unstack(colors)
    channel_tensors = list(map(lambda color: color_mask(labels, color), color_tensors))
    one_hot_labels = tf.cast(tf.stack(channel_tensors, 3), 'float32')
    return one_hot_labels


def rgb(logits):
    softmax = tf.nn.softmax(logits)
    argmax = tf.argmax(softmax, 3)
    color_map = tf.constant(colors, dtype=tf.float32)
    n = color_map.get_shape().as_list()[0]
    one_hot = tf.one_hot(argmax, n, dtype=tf.float32)
    one_hot_matrix = tf.reshape(one_hot, [-1, n])
    rgb_matrix = tf.matmul(one_hot_matrix, color_map)
    rgb_tensor = tf.reshape(rgb_matrix, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    return tf.cast(rgb_tensor, tf.float32)


def main(_):
    test_image_filenames, test_label_filenames = kitti_segnet.get_filename_list(test_dir)
    index = 0

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            config = tf.ConfigProto()
            config.gpu_options.allocator_type = 'BFC'
            sess = tf.InteractiveSession(config=config)

            images, labels = kitti_segnet.CamVidInputs(test_image_filenames,
                                                       test_label_filenames,
                                                       BATCH_SIZE,
                                                       shuffle=False)

            saver = tf.train.import_meta_graph(os.path.join(LOG_DIR, "segnet.ckpt.meta"))
            saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))

            graph = tf.get_default_graph()
            train_data = graph.get_tensor_by_name("train_data:0")
            train_label = graph.get_tensor_by_name("train_labels:0")
            is_training = graph.get_tensor_by_name("is_training:0")
            logits = tf.get_collection("logits")[0]

            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(EPOCH):
                image_batch, label_batch = sess.run([images, labels])
                feed_dict = {
                    train_data: image_batch,
                    train_label: label_batch,
                    is_training: True
                }
                prediction = rgb(logits)
                pred = sess.run([prediction], feed_dict)[0]
                for batch in range(BATCH_SIZE):
                    misc.imsave('output/segnet_kitti/decision_%d.png' % index, pred[batch])
                    misc.imsave('output/segnet_kitti/train_%d.png' % index, image_batch[batch])
                    index += 1

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.app.run(main=main)
