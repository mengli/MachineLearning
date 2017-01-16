"""Load data from SVHN dataset
"""

import os.path
import dataset
import numpy
import scipy.io

FLAGS = None

SVHN_TRAIN_FILE_NAME = 'train_32x32.mat'
SVHN_TEST_FILE_NAME = 'train_32x32.mat'
SVHN_DATA = 'X'
SVHN_LABEL = 'y'


class SVHN(object):
    def __init__(self):
        self.train = dataset.DataSet()
        self.test = dataset.DataSet()

    def ReadDataSets(self, data_dir=".", one_hot=False):
        file_path = os.path.join(data_dir, SVHN_TRAIN_FILE_NAME)
        if not os.path.isfile(file_path):
            print("SVHN dataset not found.")
            return

        read_input = scipy.io.loadmat('train_32x32.mat')
        self.train.images = read_input[SVHN_DATA]
        self.train.labels = read_input[SVHN_LABEL]

        read_input = scipy.io.loadmat('test_32x32.mat')
        self.test.images = read_input[SVHN_DATA]
        self.test.labels = read_input[SVHN_LABEL]

        self.train.images = numpy.swapaxes(self.train.images, 0, 3)
        self.train.images = numpy.swapaxes(self.train.images, 1, 2)
        self.train.images = numpy.swapaxes(self.train.images, 2, 3)

        self.train.images = self.train.images.reshape((73257, -1))

        self.test.images = numpy.swapaxes(self.test.images, 0, 3)
        self.test.images = numpy.swapaxes(self.test.images, 1, 2)
        self.test.images = numpy.swapaxes(self.test.images, 2, 3)

        self.test.images = self.test.images.reshape((26032, -1))

        self.train.images = self.train.images / numpy.float32(255)
        self.test.images = self.test.images / numpy.float32(255)

        if one_hot:
            train_labels = numpy.zeros((73257, 10), dtype=numpy.float32)
            test_labels = numpy.zeros((26032, 10), dtype=numpy.float32)

            for i in range(73257):
                train_labels[i, self.train.labels[i] - 1] = 1.
            self.train.labels = train_labels

            for j in range(26032):
                test_labels[j, self.test.labels[j] - 1] = 1.
            self.test.labels = test_labels
