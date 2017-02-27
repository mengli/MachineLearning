import os
import cv2

import tensorflow as tf


def getGroundTruth(fileNameGT):
    """
    Returns the ground truth maps for roadArea and the validArea
    :param fileNameGT:
    """
    # Read GT
    assert os.path.isfile(fileNameGT), 'Cannot find: %s' % fileNameGT
    full_gt = cv2.imread(fileNameGT, cv2.IMREAD_UNCHANGED)
    # attention: OpenCV reads in as BGR, so first channel has Blue / road GT
    roadArea = full_gt[:, :, 0] > 0
    validArea = full_gt[:, :, 2] > 0

    return roadArea, validArea


class Cifar(object):
    def __init__(self):
        self.train = dataset.DataSet()
        self.test = dataset.DataSet()

    def ReadDataSets(self, data_dir=".", one_hot=False, raw=False):
        file_path = os.path.join(data_dir, CIFAR10_FILE_NAME)
        if not os.path.isfile(file_path):
            _DownloadCifar10(data_dir)

        UnzipTarGzFile(file_path)

        xs = []
        ys = []
        for j in range(5):
            d = Unpickle(os.path.join(data_dir, CIFAR10_TRAIN_PREFIX + `j + 1`))
            x = d[CIFAR10_DATA]
            y = d[CIFAR10_LABEL]
            xs.append(x)
            ys.append(y)

        d = Unpickle(os.path.join(data_dir, CIFAR10_TEST))
        xs.append(d[CIFAR10_DATA])
        ys.append(d[CIFAR10_LABEL])

        x = numpy.concatenate(xs) / numpy.float32(255)
        y = numpy.concatenate(ys)
        if not raw:
            x = numpy.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
            x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0, 3, 1, 2)

        # subtract per-pixel mean
        pixel_mean = numpy.mean(x[0:50000], axis=0)
        x -= pixel_mean

        # create mirrored images
        if not raw:
            self.train.images = x[0:50000, :, :, :]
        else:
            self.train.images = x[0:50000]
        self.train.labels = y[0:50000]

        if not raw:
            self.test.images = x[50000:, :, :, :]
        else:
            self.train.images = x[0:50000]
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


def main(_):
    roadArea, validArea = getGroundTruth(
        "/Users/limeng/Downloads/kitti/data_road/training/gt_image_2/um_lane_000000.png")
    print "roadArea"
    print roadArea
    print "validArea"
    print validArea


if __name__ == '__main__':
    tf.app.run(main=main)
