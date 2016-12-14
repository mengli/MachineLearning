"""Load data from CIFAR-10 dataset

The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5,
as well as test_batch. Each of these files is a Python "pickled" object
produced with cPickle. Here is a Python routine which will open such a file
and return a dictionary:
"""

import pickle
import argparse
import os.path
import numpy

import dataset


FLAGS = None


class Cifar(object):

  def __init__(self):
    self.train = dataset.DataSet()
    self.test = dataset.DataSet()
  
  def read_data_sets(self, data_dir):
    # Read training samples from data_batch_1, data_batch_2,..., data_batch_6
    for num in range(1, 6):
      batch = unpickle(os.path.join(data_dir,
                                    'data_batch_' + str(num)))
      self.train.images.extend(batch['data'])
      self.train.labels.extend(batch['labels'])

def unpickle(file_path):
  with open(file_path, mode='rb') as file:
    dict = pickle.load(file)
  return dict


def main():
  cifar = Cifar()
  cifar.read_data_sets(FLAGS.data_dir)
  print cifar.train.next_batch(2)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir',
                      type=str,
                      default='/Users/limeng/Downloads/cifar-10-batches-py',
                      help='cifar-10 data set file location')

  FLAGS, unparsed = parser.parse_known_args()
  main()
