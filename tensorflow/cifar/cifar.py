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

FLAGS = None

CIFAR10_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR10_FILE_NAME = 'cifar-10-python.tar.gz'
CIFAR10_BATCH_PREFIX = 'cifar-10-batches-py/data_batch_'
CIFAR10_DATA = 'data'
CIFAR10_LABEL = 'labels'


class Cifar(object):
    def __init__(self):
        self.train = dataset.DataSet()
        self.test = dataset.DataSet()

    def ReadDataSets(self, data_dir="."):
        file_path = os.path.join(data_dir, CIFAR10_FILE_NAME)
        if not os.path.isfile(file_path):
            _DownloadCifar10(data_dir)

        UnzipTarGzFile(file_path)

        # Read training samples from data_batch_1, data_batch_2,..., data_batch_6
        for num in range(1, 6):
            batch = Unpickle(os.path.join(data_dir,
                                          CIFAR10_BATCH_PREFIX + str(num)))
            self.train.images.extend(batch[CIFAR10_DATA])
            self.train.labels.extend(batch[CIFAR10_LABEL])


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
