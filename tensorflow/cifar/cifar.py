"""Load data from CIFAR-10 dataset

The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5,
as well as test_batch. Each of these files is a Python "pickled" object
produced with cPickle. Here is a Python routine which will open such a file
and return a dictionary:
"""

import pickle
import os.path
import dataset
import urllib2
import tarfile
import numpy

FLAGS = None

CIFAR10_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR10_FILE_NAME = 'cifar-10-python.tar.gz'
CIFAR10_TRAIN_PREFIX = 'cifar-10-batches-py/data_batch_'
CIFAR10_TEST = 'cifar-10-batches-py/test_batch'
CIFAR10_DATA = 'data'
CIFAR10_LABEL = 'labels'


class Cifar(object):
    def __init__(self):
        self.train = dataset.DataSet()
        self.test = dataset.DataSet()

    def ReadDataSets(self, data_dir=".", one_hot=False):
        file_path = os.path.join(data_dir, CIFAR10_FILE_NAME)
        if not os.path.isfile(file_path):
            _DownloadCifar10(data_dir)

        UnzipTarGzFile(file_path)

        # Read training samples from data_batch_1, data_batch_2,..., data_batch_6
        batch1 = Unpickle(os.path.join(data_dir, CIFAR10_TRAIN_PREFIX + "1"))
        batch2 = Unpickle(os.path.join(data_dir, CIFAR10_TRAIN_PREFIX + "2"))
        batch3 = Unpickle(os.path.join(data_dir, CIFAR10_TRAIN_PREFIX + "3"))
        batch4 = Unpickle(os.path.join(data_dir, CIFAR10_TRAIN_PREFIX + "4"))
        batch5 = Unpickle(os.path.join(data_dir, CIFAR10_TRAIN_PREFIX + "5"))

        batch1_data = batch1[CIFAR10_DATA]
        batch2_data = batch2[CIFAR10_DATA]
        batch3_data = batch3[CIFAR10_DATA]
        batch4_data = batch4[CIFAR10_DATA]
        batch5_data = batch5[CIFAR10_DATA]

        batch1_labels = batch1[CIFAR10_LABEL]
        batch2_labels = batch2[CIFAR10_LABEL]
        batch3_labels = batch3[CIFAR10_LABEL]
        batch4_labels = batch4[CIFAR10_LABEL]
        batch5_labels = batch5[CIFAR10_LABEL]

        self.train.images = numpy.concatenate(
            (batch1_data, batch2_data, batch3_data, batch4_data, batch5_data))
        self.train.labels = numpy.concatenate(
            (batch1_labels, batch2_labels, batch3_labels, batch4_labels, batch5_labels))

        test_data = Unpickle(os.path.join(data_dir, CIFAR10_TEST))
        self.test.images = test_data[CIFAR10_DATA]
        self.test.labels = test_data[CIFAR10_LABEL]

        if one_hot:
            train_labels = numpy.zeros((50000, 10), dtype=numpy.float32)
            test_labels = numpy.zeros((10000, 10), dtype=numpy.float32)

            for i in range(50000):
                train_labels[i, self.train.labels[i]] = 1.
            self.train.labels = train_labels

            for j in range(10000):
                test_labels[j, self.test.labels[j]] = 1.
            self.test.labels = test_labels


def _DownloadCifar10(data_dir):
    _EnsureDir(data_dir)
    cifar10_zip_file = urllib2.urlopen(CIFAR10_DOWNLOAD_URL)
    with open(os.path.join(data_dir, CIFAR10_FILE_NAME), 'wb') as output:
        output.write(cifar10_zip_file.read())


def UnzipTarGzFile(file_path):
    with tarfile.open(file_path) as tar:
        tar.extractall()
        tar.close()


def _EnsureDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def Unpickle(file_path):
    with open(file_path, mode='rb') as file:
        dict = pickle.load(file)
    return dict
