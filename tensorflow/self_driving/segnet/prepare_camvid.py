import os
import numpy as np
from scipy import misc

data_image_dir = "/usr/local/google/home/limeng/Downloads/camvid/LabeledApproved_full"
image_dir = "/usr/local/google/home/limeng/Downloads/camvid/LabeledApproved_full/image_2"


IMAGE_HEIGHT = 720
IMAGE_WIDTH = 960
IMAGE_DEPTH = 3


color2index = {
    (64, 128, 64) : 0, # Animal
    (192, 0, 128) : 1, # Archway
    (0, 128, 192) : 2, # Bicyclist
    (0, 128, 64)  : 3, # Bridge
    (128, 0, 0)   : 4, # Building
    (64, 0, 128)  : 5, # Car
    (64, 0, 192)  : 6, # CartLuggagePram
    (192, 128, 64) : 7, # Child
    (192, 192, 128) : 8, # Column_Pole
    (64, 64, 128) :9, # Fence
    (128, 0, 192) : 10, # LaneMkgsDriv
    (192, 0, 64) : 11, # LaneMkgsNonDriv
    (128, 128, 64) : 12, # Misc_Text
    (192, 0, 192) : 13, # MotorcycleScooter
    (128, 64, 64) : 14, # OtherMoving
    (64, 192, 128) : 15, # ParkingBlock
    (64, 64, 0) : 16, # Pedestrian
    (128, 64, 128) : 17, # Road
    (128, 128, 192) : 18, # RoadShoulder
    (0, 0, 192) : 19, # Sidewalk
    (192, 128, 128) : 20, # SignSymbol
    (128, 128, 128) : 21, # Sky
    (64, 128, 192) : 22, # SUVPickupTruck
    (0, 0, 64) : 23, # TrafficCone
    (0, 64, 64) : 24, # TrafficLight
    (192, 64, 128) : 25, # Train
    (128, 128, 0) : 26, # Tree
    (192, 128, 192) : 27, # Truck_Bus
    (64, 0, 64) : 28, # Tunnel
    (192, 192, 0) : 29, # VegetationMisc
    (0, 0, 0) : 30, # Void
    (64, 192, 0) : 31, # Wall
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
            if (r, g, b) in color2index:
                m_lable[h, w] = color2index[(r, g, b)]
            else:
                m_lable[h, w] = 30
    return m_lable


def convert_to_label_data(file_name):
    assert os.path.isfile(file_name), 'Cannot find: %s' % file_name
    return im2index(misc.imread(file_name, mode='RGB'))


def main():
    for file in os.listdir(data_image_dir):
        if file.endswith(".png"):
            print("Try to converting %s" % file)
            gt_label = convert_to_label_data(os.path.join(data_image_dir, file))
            if gt_label is not None:
                misc.imsave(os.path.join(image_dir, file), gt_label)


if __name__ == '__main__':
    main()
