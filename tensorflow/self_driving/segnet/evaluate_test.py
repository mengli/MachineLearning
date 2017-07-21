from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test
from self_driving.segnet import evaluate
import tensorflow as tf


class EvaluateTest(test.TestCase):

    def testTfArgmax(self):
        '''[[[[  1.   2.]
              [  3.   4.]
              [  5.   6.]]
             [[  8.   7.]
              [  9.  10.]
              [ 11.  12.]]
             [[ 13.  14.]
              [ 16.  15.]
              [ 17.  18.]]]]'''
        tensor_input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 7.0, 9.0,
                        10.0, 11.0, 12.0, 13.0, 14.0, 16.0, 15.0, 17.0, 18.0]
        with self.test_session(use_gpu=False) as sess:
            t = constant_op.constant(tensor_input, shape=[1, 3, 3, 2])
            argmax_op = tf.argmax(t, axis=3)
            argmax = sess.run([argmax_op])
            self.assertAllEqual(argmax, [[[[1, 1, 1], [0, 1, 1], [1, 0, 1]]]])


    def testColorImage(self):
        '''[[[[  0.   2.]
              [  3.   4.]
              [  5.   6.]]
             [[  8.   7.]
              [  9.  10.]
              [ 11.  12.]]
             [[ 13.  14.]
              [ 16.  15.]
              [ 17.  18.]]]]'''
        tensor_input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 7.0, 9.0,
                        10.0, 11.0, 12.0, 13.0, 14.0, 16.0, 15.0, 17.0, 18.0]
        with self.test_session(use_gpu=False):
            t = constant_op.constant(tensor_input, shape=[3, 3, 1, 2])
            argmax_op = tf.argmax(t, dimension=3)
            up_color = evaluate.color_image(argmax_op.eval(), 1.)
            self.assertAllClose(up_color, [[[[0.60000002, 0.60000002, 0.60000002, 1.]],
                                            [[0.60000002, 0.60000002, 0.60000002, 1.]],
                                            [[0.60000002, 0.60000002, 0.60000002, 1.]]],
                                           [[[0.89411765, 0.10196079, 0.10980392, 1.]],
                                            [[0.60000002, 0.60000002, 0.60000002, 1.]],
                                            [[0.60000002, 0.60000002, 0.60000002, 1.]]],
                                           [[[0.60000002, 0.60000002, 0.60000002, 1.]],
                                            [[0.89411765, 0.10196079, 0.10980392, 1.]],
                                            [[0.60000002, 0.60000002, 0.60000002, 1.]]]])


if __name__ == "__main__":
    test.main()