import numpy as np
import os.path
from keras import backend as K
from keras.preprocessing import image

# dimensions of our images.
img_width, img_height = 455, 256

train_samples_dir = '../train_data/'
test_samples_dir = '../test_data/'

train_label = '../train_data/train_data_label.txt'
test_label = '../test_data/test_data_label.txt'

num_train_samples = 40000
num_test_samples = 5406


def load_data():
    x_train = load_image_set(num_train_samples, train_samples_dir)
    y_train = load_label_set(num_train_samples, train_label)

    x_test = load_image_set(num_test_samples, test_samples_dir)
    y_test = load_label_set(num_test_samples, test_label)

    return (x_train, y_train), (x_test, y_test)


def load_image_set(image_set_size, image_set_dir):
    img_set = np.zeros((image_set_size, img_width, img_height, 3), dtype='uint8')

    for i in range(0, image_set_size):
        img_path = os.path.join(image_set_dir, str('%d.jpg' % i))
        if os.path.isfile(img_path):
            img = image.load_img(img_path, target_size=(img_width, img_height))
            img_set[i, :, :, :] = image.img_to_array(img)

    if K.image_data_format() == 'channels_first':
        img_set = img_set.transpose(0, 3, 1, 2)

    return img_set


def load_label_set(label_set_size, label_set_dir):
    label_set = np.zeros(label_set_size, dtype='float64')

    lines = [line.rstrip('\n') for line in open(label_set_dir)]

    i = 0
    for line in lines:
        label_set[i] = float(line.split(' ')[1])
        i += 1

    return label_set


def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    print x_train.shape
    print y_train.shape
    print y_train
    print x_test.shape
    print y_test.shape
    print y_test


if __name__ == '__main__':
    main()

