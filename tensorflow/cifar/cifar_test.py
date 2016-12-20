import unittest
import cifar


class CifarTest(unittest.TestCase):
    def setUp(self):
        self._cifar = cifar.Cifar()

    def testReadDataSets(self):
        self._cifar.ReadDataSets()
        self.assertEqual(len(self._cifar.train.images), 50000)
        self.assertEqual(len(self._cifar.train.labels), 50000)
        self.assertEqual(len(self._cifar.test.images), 10000)
        self.assertEqual(len(self._cifar.test.labels), 10000)

    def testReadDataSetsOneHotEnabled(self):
        self._cifar.ReadDataSets(one_hot=True)

        self.assertEqual(len(self._cifar.train.images), 50000)
        self.assertEqual(len(self._cifar.train.images[0]), 3072)
        self.assertEqual(len(self._cifar.train.labels[0]), 10)
        self.assertEqual(1, self._cifar.train.labels[0][6])

        self.assertEqual(len(self._cifar.test.images), 10000)
        self.assertEqual(len(self._cifar.test.labels[0]), 10)
        self.assertEqual(1, self._cifar.test.labels[0][3])

if __name__ == '__main__':
    unittest.main()
