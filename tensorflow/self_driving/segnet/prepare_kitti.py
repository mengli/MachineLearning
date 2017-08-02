import os
import numpy as np
from scipy import misc

data_image_dir = "/usr/local/google/home/limeng/Downloads/kitti/data_road/training/data_image_2"
image_dir = "/usr/local/google/home/limeng/Downloads/kitti/data_road/training/image_2"
data_label_dir = "/usr/local/google/home/limeng/Downloads/kitti/data_road/training/data_label_2"
label_output_dir = "/usr/local/google/home/limeng/Downloads/kitti/data_road/training/gt_image_2"


IMAGE_HEIGHT = 375
IMAGE_WIDTH = 1242
IMAGE_DEPTH = 3


#   R   G   B
# 255   0 255 road
#   0   0 255 road
# 255   0   0 valid
#   0   0   0 invalid
color2index = {
    (255, 0, 255) : 0,
    (0,   0, 255) : 0,
    (255, 0,   0) : 1,
    (0,   0,   0) : 1,
}


def im2index(im):
    height, width, ch = im.shape
    assert ch == IMAGE_DEPTH
    if height != IMAGE_HEIGHT or width != IMAGE_WIDTH:
        print("Size: (%d, %d, %d) cannot be used." % (height, width, ch))
        return None
    m_lable = np.zeros((height, width), dtype=np.uint8)
    for w in range(width):
        for h in range(height):
            r, g, b = im[h, w, :]
            m_lable[h, w] = color2index[(r, g, b)]
    return m_lable


def convert_to_label_data(file_name):
    assert os.path.isfile(file_name), 'Cannot find: %s' % file_name
    return im2index(misc.imread(file_name, mode='RGB'))


def main():
    for file in os.listdir(data_image_dir):
        if file.endswith(".png"):
            print("Try to copy %s" % file)
            im = misc.imread(os.path.join(data_image_dir, file), mode='RGB')
            height, width, ch = im.shape
            assert ch == IMAGE_DEPTH
            if height == IMAGE_HEIGHT and width == IMAGE_WIDTH and ch == IMAGE_DEPTH:
                misc.imsave(os.path.join(image_dir, file), im)
            else:
                print("Size: (%d, %d, %d) cannot be used." % (height, width, ch))

    for file in os.listdir(data_label_dir):
        if file.endswith(".png"):
            print("Try to converting %s" % file)
            gt_label = convert_to_label_data(os.path.join(data_label_dir, file))
            if gt_label is not None:
                misc.imsave(os.path.join(label_output_dir, file), gt_label)


if __name__ == '__main__':
    main()
