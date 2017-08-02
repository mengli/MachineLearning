import os
from scipy import misc

data_test_dir = "/usr/local/google/home/limeng/Downloads/kitti/data_road/testing/data_image_2"
test_dir = "/usr/local/google/home/limeng/Downloads/kitti/data_road/testing/image_2"


IMAGE_HEIGHT = 375
IMAGE_WIDTH = 1242
IMAGE_DEPTH = 3


def main():
    for file in os.listdir(data_test_dir):
        if file.endswith(".png"):
            print("Try to copy %s" % file)
            im = misc.imread(os.path.join(data_test_dir, file), mode='RGB')
            height, width, ch = im.shape
            assert ch == IMAGE_DEPTH
            if height == IMAGE_HEIGHT and width == IMAGE_WIDTH and ch == IMAGE_DEPTH:
                misc.imsave(os.path.join(test_dir, file), im)
            else:
                print("Size: (%d, %d, %d) cannot be used." % (height, width, ch))


if __name__ == '__main__':
    main()
