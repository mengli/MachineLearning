from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test

import camvid
import tensorflow as tf

image_dir = "/usr/local/google/home/limeng/Downloads/camvid/data/train.txt"


class CamvidTest(test.TestCase):

    def testGetFileNameList(self):
        image_filenames, label_filenames = camvid.get_filename_list(image_dir)
        self.assertEqual(len(image_filenames), 367)

    def testCamVidInputs(self):
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        with self.test_session(use_gpu=True, config = config) as sess:
            image_filenames, label_filenames = camvid.get_filename_list(image_dir)
            images, labels = camvid.CamVidInputs(image_filenames, label_filenames, 32)
            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            images_batch, labels_batch = sess.run([images, labels])
            self.assertEqual(images.get_shape(), [32, 360, 480, 3])
            self.assertEqual(labels.get_shape(), [32, 360, 480, 1])
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    test.main()
