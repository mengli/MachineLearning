"""Evaluate SegNet.

nohup python -u -m self_driving.segnet.evaluate > self_driving/segnet/output.txt 2>&1 &

"""

import os
import tensorflow as tf
from utils import camvid
from scipy import misc
import matplotlib.cm as cm
from matplotlib.colors import Normalize

LOG_DIR = 'save'
BATCH_SIZE = 8
EPOCH = 29
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
IMAGE_CHANNEL = 3
NUM_CLASSES = 12

test_dir = "/usr/local/google/home/limeng/Downloads/camvid/data/test.txt"


def color_image(image, num_classes):
    cmap = cm.get_cmap('Set1')
    norm = Normalize(vmin=0., vmax=num_classes)
    return cmap(norm(image))


def main(_):
    test_image_filenames, test_label_filenames = camvid.get_filename_list(test_dir)

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            config = tf.ConfigProto()
            config.gpu_options.allocator_type = 'BFC'
            sess = tf.InteractiveSession(config = config)

            images, labels = camvid.CamVidInputs(test_image_filenames,
                                                 test_label_filenames,
                                                 BATCH_SIZE,
                                                 shuffle=False)

            saver = tf.train.import_meta_graph(os.path.join(LOG_DIR, "segnet.ckpt.meta"))
            saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))

            graph = tf.get_default_graph()
            train_data = graph.get_tensor_by_name("train_data:0")
            train_label = graph.get_tensor_by_name("train_labels:0")
            logits = tf.get_collection("logits")[0]

            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(EPOCH):
                image_batch ,label_batch = sess.run([images, labels])
                feed_dict = {
                    train_data: image_batch,
                    train_label: label_batch
                }
                prediction = sess.run([logits], feed_dict)
                pred = tf.argmax(prediction[0], dimension=3).eval()
                for batch in range(BATCH_SIZE):
                    up_color = color_image(pred[batch], NUM_CLASSES)
                    misc.imsave('output/decision_%d_%d.png' % (i, batch), up_color)
                    misc.imsave('output/train_%d_%d.png' % (i, batch), image_batch[batch])

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.app.run(main=main)
