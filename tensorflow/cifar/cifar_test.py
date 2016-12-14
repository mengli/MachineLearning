import unittest
import cifar


class CifarTest(unittest.TestCase):
    def setUp(self):
        self._cifar = cifar.Cifar()

    def testReadDataSets(self):
        self._cifar.ReadDataSets("cifar-10-batches-py")
        self.assertEqual(len(self._cifar.train.images), 50000)


if __name__ == '__main__':
    unittest.main()
