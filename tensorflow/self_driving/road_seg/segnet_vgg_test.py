from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class PoolingTest(test.TestCase):

    def testMaxPoolingWithArgmax(self):
        # MaxPoolWithArgMax is implemented only on CUDA.
        if not test.is_gpu_available(cuda_only=True):
            return
        tensor_input = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        with self.test_session(use_gpu=True) as sess:
            t = constant_op.constant(tensor_input, shape=[1, 3, 3, 1])
            out_op, argmax_op = nn_ops.max_pool_with_argmax(
                t,
                ksize=[1, 2, 2, 1],
                strides=[1, 1, 1, 1],
                Targmax=dtypes.int64,
                padding="VALID")
            out, argmax = sess.run([out_op, argmax_op])
            self.assertShapeEqual(out, out_op)
            self.assertShapeEqual(argmax, argmax_op)
            self.assertAllClose(out.ravel(), [1.0, 1.0, 1.0, 1.0])
            self.assertAllEqual(argmax.ravel(), [0, 1, 3, 5])


if __name__ == "__main__":
    test.main()
