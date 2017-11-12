from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test
from scipy import misc
import udacity_data

IMG_TRAIN = "/usr/local/google/home/limeng/Downloads/udacity/ch2_002/output/HMB_1/center/1479424215880976321.png"
IMG_VAL = "/usr/local/google/home/limeng/Downloads/udacity/test/HMB_3/center/1479425441182877835.png"

class UdacityDataTest(test.TestCase):

    def testReadData(self):
        udacity_data.read_data()
        self.assertAllEqual(len(udacity_data.train_xs), 33808)
        self.assertAllEqual(len(udacity_data.train_ys), 33808)
        self.assertAllEqual(len(udacity_data.val_xs), 5279)
        self.assertAllEqual(len(udacity_data.val_ys), 5279)
        self.assertTrue(IMG_TRAIN in udacity_data.train_xs)
        self.assertAllClose(udacity_data.train_ys[udacity_data.train_xs.index(IMG_TRAIN)], 0.0010389391)
        self.assertTrue(IMG_VAL in udacity_data.val_xs)
        self.assertAllClose(udacity_data.val_ys[udacity_data.val_xs.index(IMG_VAL)], -0.0169280299)

    def testReadData(self):
        udacity_data.read_data()
        x_out, y_out = udacity_data.load_val_batch(64)
        misc.imsave('test.png', x_out[0])


if __name__ == "__main__":
    test.main()