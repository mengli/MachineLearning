from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test
import my_image

class MyImageTest(test.TestCase):

    def testReadData(self):
        myImageDataGenerator = my_image.MyImageDataGenerator()
        generator = myImageDataGenerator.flow("udacity_train.txt",
                                              [224, 224, 3],
                                              shuffle=False,
                                              save_to_dir='test')
        images, labels = generator.next()
        self.assertAllEqual(images.shape, [32, 224, 224, 3])
        self.assertAllEqual(labels.shape, [32])
        self.assertAllClose(labels[0], 0.0490969472)


if __name__ == "__main__":
    test.main()