import os
import numpy
import tensorflow as tf
import scipy as scp
import scipy.misc

KITTI_TRAIN_DIR_PREFIX = '/usr/local/google/home/limeng/Downloads/kitti/data_road/training/image_2/'
KITTI_GT_DIR_PREFIX = '/usr/local/google/home/limeng/Downloads/kitti/data_road/training/gt_image_2/'

UM_TRAIN_TEMPLATE = "um_0000%02d.png"
UU_TRAIN_TEMPLATE = "uu_0000%02d.png"
UMM_TRAIN_TEMPLATE = "umm_0000%02d.png"

UU_GT_ROAD_TEMPLATE = "uu_road_0000%02d.png"
UM_GT_LANE_TEMPLATE = "um_lane_0000%02d.png"
UM_GT_ROAD_TEMPLATE = "um_road_0000%02d.png"
UMM_GT_ROAD_TEMPLATE = "umm_road_0000%02d.png"


class Kitti(object):
    def __init__(self):
        self._images = []
        self._labels = []
        self._file_count = 0
        self._read_datasets()

    def _read_datasets(self,
                       train_data_dir=KITTI_TRAIN_DIR_PREFIX,
                       gt_data_dir=KITTI_GT_DIR_PREFIX,
                       cat='uu'):
        assert os.path.isdir(train_data_dir), 'Cannot find: %s' % train_data_dir

        self._file_count = 98
        train_file_temp = UU_TRAIN_TEMPLATE
        gt_file_temp = UU_GT_ROAD_TEMPLATE
        if cat == 'um':
            self._file_count = 95
            train_file_temp = UM_TRAIN_TEMPLATE
            gt_file_temp = UM_GT_ROAD_TEMPLATE
        elif cat == 'umm':
            self._file_count = 96
            train_file_temp = UMM_TRAIN_TEMPLATE
            gt_file_temp = UMM_GT_ROAD_TEMPLATE

        for i in range(0, self._file_count):
            train_file_name = train_data_dir + train_file_temp % i
            gt_file_name = gt_data_dir + gt_file_temp % i
            print(train_file_name)
            x = get_training_data(train_file_name)
            y = get_ground_truth(gt_file_name)

            self._images.append(x)
            self._labels.append(y)

    def next_batch(self, batch_id):
        return self._images[batch_id], self._labels[batch_id]


def get_training_data(file_name):
    assert os.path.isfile(file_name), 'Cannot find: %s' % file_name
    training_data = scp.misc.imread(file_name, mode='RGB')
    return numpy.expand_dims(training_data, axis=0)


def get_ground_truth(fileNameGT):
    assert os.path.isfile(fileNameGT), 'Cannot find: %s' % fileNameGT
    full_gt = scp.misc.imread(fileNameGT, mode='RGB')
    roadArea = (full_gt[:, :, 2] > 0)
    notRoadArea = (full_gt[:, :, 2] == 0)
    gt_data = numpy.dstack((roadArea, notRoadArea))
    return numpy.expand_dims(gt_data, axis=0)


def main(_):
    kitti = Kitti()
    for i in range(0, 20):
        img, label = kitti.next_batch()
        print "images"
        print img.shape
        print "labels"
        print label.shape


if __name__ == '__main__':
    tf.app.run(main=main)
