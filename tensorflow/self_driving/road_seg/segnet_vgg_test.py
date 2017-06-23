from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test

import segnet_vgg


class PoolingTest(test.TestCase):

    def testMaxPoolingWithArgmax(self):
        # MaxPoolWithArgMax is implemented only on CUDA.
        if not test.is_gpu_available(cuda_only=True):
            return
        tensor_input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        with self.test_session(use_gpu=True) as sess:
            t = constant_op.constant(tensor_input, shape=[1, 3, 3, 1])
            out_op, argmax_op = segnet_vgg.max_pool_with_argmax(t)
            out, argmax = sess.run([out_op, argmax_op])
            self.assertShapeEqual(out, out_op)
            self.assertShapeEqual(argmax, argmax_op)
            self.assertAllClose(out.ravel(), [5.0, 6.0, 8.0, 9.0])
            self.assertAllEqual(argmax.ravel(), [4, 5, 7, 8])


if __name__ == "__main__":
    test.main()
