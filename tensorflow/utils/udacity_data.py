import scipy.misc
import random
import pandas as pd
import tensorflow as tf


TRAIN_IMG_PREFIX = "/usr/local/google/home/limeng/Downloads/udacity/ch2_002/output/HMB_%s/"
TRAIN_CSV = "/usr/local/google/home/limeng/Downloads/udacity/ch2_002/output/HMB_%s/interpolated.csv"
VAL_IMG_PREFIX = "/usr/local/google/home/limeng/Downloads/udacity/test/HMB_3/"
VAL_CSV = "/usr/local/google/home/limeng/Downloads/udacity/test/HMB_3/interpolated.csv"


def read_csv(csv_file_name, img_prefix):
    x_out = []
    data_csv = pd.read_csv(csv_file_name)
    data = data_csv[[x.startswith("center") for x in data_csv["filename"]]]
    for file_name in data["filename"]:
        x_out.append(img_prefix + file_name)
    return x_out, data["angle"]


def read_data():
    train_xs = []
    train_ys = []
    val_xs = []
    val_ys = []

    # Read train set
    for idx in range(1, 7):
        if idx == 3:
            continue
        x_out, y_out = read_csv(TRAIN_CSV % idx, TRAIN_IMG_PREFIX % idx)
        train_xs.extend(x_out)
        train_ys.extend(y_out)
    # Read val set
    val_xs, val_ys = read_csv(VAL_CSV, VAL_IMG_PREFIX)

    #shuffle train set
    c = list(zip(train_xs, train_ys))
    random.shuffle(c)
    train_xs, train_ys = zip(*c)
    #shuffle val set
    c = list(zip(val_xs, val_ys))
    random.shuffle(c)
    val_xs, val_ys = zip(*c)

    print(len(train_xs))
    print(len(train_ys))
    print(len(val_xs))
    print(len(val_ys))


def main(_):
    read_data()

if __name__ == '__main__':
    tf.app.run(main=main)