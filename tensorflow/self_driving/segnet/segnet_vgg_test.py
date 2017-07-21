from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test
from self_driving.segnet import segnet_vgg
import tensorflow as tf
import numpy as np

NUM_CLASSES = 11

class PoolingTest(test.TestCase):

    def testMaxPoolingWithArgmax(self):
        # MaxPoolWithArgMax is implemented only on CUDA.
        if not test.is_gpu_available(cuda_only=True):
            return
        '''[[[[  1.   2.]
              [  3.   4.]
              [  5.   6.]]
             [[  7.   8.]
              [  9.  10.]
              [ 11.  12.]]
             [[ 13.  14.]
              [ 15.  16.]
              [ 17.  18.]]]]'''
        tensor_input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                        10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
        with self.test_session(use_gpu=True) as sess:
            t = constant_op.constant(tensor_input, shape=[1, 3, 3, 2])
            out_op, argmax_op = segnet_vgg.max_pool_with_argmax(t)
            out, argmax = sess.run([out_op, argmax_op])
            self.assertShapeEqual(out, out_op)
            self.assertShapeEqual(argmax, argmax_op)
            '''[[[9, 10]
                 [11, 12]]
                [[15, 16]
                 [17, 18]]]'''
            self.assertAllClose(out.ravel(), [9., 10., 11., 12., 15., 16., 17., 18.])
            self.assertAllEqual(argmax.ravel(), [8, 9, 10, 11, 14, 15, 16, 17])

    def testMaxUnpoolingWithArgmax(self):
        '''[[[[  1.   2.]
              [  3.   4.]
              [  5.   6.]]
             [[  7.   8.]
              [  9.  10.]
              [ 11.  12.]]
             [[ 13.  14.]
              [ 15.  16.]
              [ 17.  18.]]]]'''
        tensor_input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                        10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
        with self.test_session(use_gpu=True) as sess:
            t = constant_op.constant(tensor_input, shape=[1, 3, 3, 2])
            out_op, argmax_op = segnet_vgg.max_pool_with_argmax(t)
            out_op = segnet_vgg.max_unpool_with_argmax(out_op,
                                                       argmax_op,
                                                       output_shape=np.int64([1, 3, 3, 2]))
            out = sess.run([out_op])
            self.assertAllClose(out, [[[[[  0.,   0.],
                                         [  0.,   0.],
                                         [  0.,   0.]],
                                        [[  0.,   0.],
                                         [  9.,  10.],
                                         [ 11.,  12.]],
                                        [[  0.,   0.],
                                         [ 15.,  16.],
                                         [ 17.,  18.]]]]])

    def testGetBias(self):
        with self.test_session(use_gpu=True) as sess:
            bias = segnet_vgg.get_bias("conv1_1")
            sess.run(tf.global_variables_initializer())
            self.assertEqual(bias.get_shape(), [64,])
            self.assertAllClose(tf.reduce_sum(bias).eval(), 32.08903503417969)

    def testGetConvFilter(self):
        with self.test_session(use_gpu=True) as sess:
            weight = segnet_vgg.get_conv_filter("conv1_1")
            sess.run(tf.global_variables_initializer())
            self.assertEqual(weight.get_shape(), [3, 3, 3, 64])
            self.assertAllClose(tf.reduce_sum(weight).eval(), -4.212705612182617)

    def testConvLayerWithBn(self):
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        tensor_input = tf.ones([10, 495, 289, 3], tf.float32)
        with self.test_session(use_gpu=True, config = config) as sess:
            conv_op = segnet_vgg.conv_layer_with_bn(tensor_input, tf.constant(True), "conv1_1")
            sess.run(tf.global_variables_initializer())
            conv_out = sess.run([conv_op])
            self.assertEqual(np.array(conv_out).shape, (1, 10, 495, 289, 64))

    def testDeconvLayerWithBn(self):
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        tensor_input = tf.ones([10, 495, 289, 3], tf.float32)
        with self.test_session(use_gpu=True, config = config) as sess:
            conv_op = segnet_vgg.deconv_layer_with_bn(tensor_input,
                                                      [3, 3, 3, 128],
                                                      tf.constant(True), "conv1_1")
            sess.run(tf.global_variables_initializer())
            conv_out = sess.run([conv_op])
            self.assertEqual(np.array(conv_out).shape, (1, 10, 495, 289, 128))

    def testInference(self):
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        train_data = tf.ones([10, 495, 289, 3], tf.float32)
        with self.test_session(use_gpu=True, config = config) as sess:
            model_op = segnet_vgg.inference(train_data)
            sess.run(tf.global_variables_initializer())
            model_out = sess.run([model_op])
            self.assertEqual(np.array(model_out).shape, (1, 10, 495, 289, NUM_CLASSES))


if __name__ == "__main__":
    test.main()
