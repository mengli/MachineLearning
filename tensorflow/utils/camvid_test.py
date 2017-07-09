from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test

import camvid

image_dir = "/usr/local/google/home/limeng/Downloads/camvid/data/train.txt"


class CamvidTest(test.TestCase):

    def testGetFileNameList(self):
        image_filenames, label_filenames = camvid.get_filename_list(image_dir)
        self.assertEqual(len(image_filenames), 367)

    def testCamVidInputs(self):
        image_filenames, label_filenames = camvid.get_filename_list(image_dir)
        images, labels = camvid.CamVidInputs(image_filenames, label_filenames, 32)
        self.assertEqual(images.get_shape(), [32, 360, 480, 3])
        self.assertEqual(labels.get_shape(), [32, 360, 480, 1])


if __name__ == "__main__":
    test.main()
