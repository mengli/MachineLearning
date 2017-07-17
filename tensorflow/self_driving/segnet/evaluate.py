"""Evaluate SegNet.

nohup python -u -m self_driving.segnet.evaluate > self_driving/segnet/output.txt 2>&1 &

"""

import os
import tensorflow as tf
from utils import camvid

LOG_DIR = 'save'
BATCH_SIZE = 8
EPOCH = 29
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
IMAGE_CHANNEL = 3
NUM_CLASSES = 12

test_dir = "/usr/local/google/home/limeng/Downloads/camvid/data/test.txt"


def main(_):
    test_image_filenames, test_label_filenames = camvid.get_filename_list(test_dir)

    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.InteractiveSession(config = config)

        images, labels = camvid.CamVidInputs(test_image_filenames,
                                             test_label_filenames,
                                             BATCH_SIZE,
                                             shuffle=False)

        saver = tf.train.import_meta_graph(os.path.join(LOG_DIR, "model.ckpt.meta"))
        saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))

        graph = tf.get_default_graph()
        train_data = graph.get_tensor_by_name("train_data:0")
        train_label = graph.get_tensor_by_name("train_labels:0")
        is_training = graph.get_tensor_by_name("is_training:0")
        logits = graph.get_tensor_by_name("up_conv1/relu:0")
        diff = graph.get_tensor_by_name("cross_entropy_mean:0")

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(EPOCH):
            print i
            image_batch ,label_batch = sess.run([images, labels])
            feed_dict = {
                train_data: image_batch,
                train_label: label_batch,
                is_training: False
            }
            classification, classification_diff = sess.run([logits, diff], feed_dict)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run(main=main)