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

        xs = []
        ys = []
        for j in range(5):
          d = Unpickle(os.path.join(data_dir, CIFAR10_TRAIN_PREFIX + `j+1`))
          x = d[CIFAR10_DATA]
          y = d[CIFAR10_LABEL]
          xs.append(x)
          ys.append(y)

        d = Unpickle(os.path.join(data_dir, CIFAR10_TEST))
        xs.append(d[CIFAR10_DATA])
        ys.append(d[CIFAR10_LABEL])

        x = numpy.concatenate(xs) / numpy.float32(255)
        y = numpy.concatenate(ys)
        x = numpy.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
        x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

        # subtract per-pixel mean
        pixel_mean = numpy.mean(x[0:50000],axis=0)
        x -= pixel_mean

        # create mirrored images
        self.train.images = x[0:50000,:,:,:]
        self.train.labels = y[0:50000]

        self.test.images = x[50000:,:,:,:]
        self.test.labels = y[50000:]

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
