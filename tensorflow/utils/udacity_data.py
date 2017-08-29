import scipy.misc
import random
import pandas as pd
import tensorflow as tf

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

train_xs = []
train_ys = []
val_xs = []
val_ys = []

TRAIN_IMG_PREFIX = "/usr/local/google/home/limeng/Downloads/udacity/ch2_002/output/HMB_%s/"
TRAIN_CSV = "/usr/local/google/home/limeng/Downloads/udacity/ch2_002/output/HMB_%s/interpolated.csv"
VAL_IMG_PREFIX = "/usr/local/google/home/limeng/Downloads/udacity/test/HMB_3/"
VAL_CSV = "/usr/local/google/home/limeng/Downloads/udacity/test/HMB_3/interpolated.csv"

NUM_TRAIN_IMAGES = 33808
NUM_VAL_IMAGES = 5279


def read_csv(csv_file_name, img_prefix):
    x_out = []
    data_csv = pd.read_csv(csv_file_name)
    data = data_csv[[x.startswith("center") for x in data_csv["filename"]]]
    for file_name in data["filename"]:
        x_out.append(img_prefix + file_name)
    return x_out, data["angle"]


def read_data(shuffe=True):
    global train_xs
    global train_ys
    global val_xs
    global val_ys

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
    if shuffe:
        random.shuffle(c)
    train_xs, train_ys = zip(*c)
    #shuffle val set
    c = list(zip(val_xs, val_ys))
    if shuffe:
        random.shuffle(c)
    val_xs, val_ys = zip(*c)


def load_train_batch(batch_size):
    global train_batch_pointer
    global train_xs
    global train_ys

    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(
            scipy.misc.imresize(
            scipy.misc.imread(
                train_xs[(train_batch_pointer + i) % NUM_TRAIN_IMAGES], mode="RGB"), [66, 200]) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % NUM_TRAIN_IMAGES]])
    train_batch_pointer += batch_size
    return x_out, y_out


def load_val_batch(batch_size):
    global val_batch_pointer
    global val_xs
    global val_ys

    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(
            scipy.misc.imresize(
            scipy.misc.imread(
                val_xs[(val_batch_pointer + i) % NUM_VAL_IMAGES], mode="RGB"), [66, 200]) / 255.0)
        #print(val_ys)
        y_out.append([val_ys[(val_batch_pointer + i) % NUM_VAL_IMAGES]])
    val_batch_pointer += batch_size
    return x_out, y_out


def main(_):
    read_data()

if __name__ == '__main__':
    tf.app.run(main=main)