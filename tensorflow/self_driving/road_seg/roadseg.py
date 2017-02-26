import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.convolutional import Convolution2D, Deconvolution2D, MaxPooling2D, ZeroPadding2D


N_cl = 2
C = 32


def get_model():
    # KITTI data set.
    main_input = Input(shape=(None, 3, 1242, 375), dtype='float32', name='kitti_data')

    conv1_1 = ZeroPadding2D((10, 10))(main_input)
    conv1_1 = Convolution2D(64, 3, 3, activation='relu')(conv1_1)  # 1260 * 393 * 64
    conv1_2 = ZeroPadding2D((1, 1))(conv1_1)
    conv1_2 = Convolution2D(64, 3, 3, activation='relu')(conv1_2)  # 1260 * 393 * 64
    pool1 = ZeroPadding2D((0, 1))(conv1_2)  # 1260 * 394 * 64
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(pool1)  # 630 * 197 * 64

    conv2_1 = ZeroPadding2D((1, 1))(pool1)
    conv2_1 = Convolution2D(128, 3, 3, activation='relu')(conv2_1) # 630 * 197 * 128
    conv2_2 = ZeroPadding2D((1, 1))(conv2_1)
    conv2_2 = Convolution2D(128, 3, 3, activation='relu')(conv2_2) # 630 * 197 * 128
    pool2 = ZeroPadding2D((0, 1))(conv2_2)  # 630 * 198 * 128
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(pool2) # 315 * 99 * 128

    conv3_1 = ZeroPadding2D((1, 1))(pool2)
    conv3_1 = Convolution2D(256, 3, 3, activation='relu')(conv3_1) # 315 * 99 * 256
    conv3_2 = ZeroPadding2D((1, 1))(conv3_1)
    conv3_2 = Convolution2D(256, 3, 3, activation='relu')(conv3_2) # 315 * 99 * 256
    conv3_3 = ZeroPadding2D((1, 1))(conv3_2)
    conv3_3 = Convolution2D(256, 3, 3, activation='relu')(conv3_3) # 315 * 99 * 256
    pool3 = ZeroPadding2D((1, 1))(conv3_3)  # 316 * 100 * 256
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(pool3) # 158 * 50 * 256

    conv4_1 = ZeroPadding2D((1, 1))(pool3)
    conv4_1 = Convolution2D(512, 3, 3, activation='relu')(conv4_1) # 158 * 50 * 512
    conv4_2 = ZeroPadding2D((1, 1))(conv4_1)
    conv4_2 = Convolution2D(512, 3, 3, activation='relu')(conv4_2) # 158 * 50 * 512
    conv4_3 = ZeroPadding2D((1, 1))(conv4_2)
    conv4_3 = Convolution2D(512, 3, 3, activation='relu')(conv4_3) # 158 * 50 * 512
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_3) # 79 * 25 * 512

    conv5_1 = ZeroPadding2D((1, 1))(pool4)
    conv5_1 = Convolution2D(512, 3, 3, activation='relu')(conv5_1) # 79 * 25 * 512
    conv5_2 = ZeroPadding2D((1, 1))(conv5_1)
    conv5_2 = Convolution2D(512, 3, 3, activation='relu')(conv5_2) # 79 * 25 * 512
    conv5_3 = ZeroPadding2D((1, 1))(conv5_2)
    conv5_3 = Convolution2D(512, 3, 3, activation='relu')(conv5_3) # 79 * 25 * 512
    pool5 = ZeroPadding2D((1, 1))(conv5_3)  # 80 * 26 * 512
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(pool5) # 40 * 13 * 512

    # FC_conv1
    fc6 = ZeroPadding2D((1, 1))(pool5)
    fc6 = Convolution2D(1024, 3, 3, activation='relu')(fc6)  # 40 * 13 * 1024
    fc6 = Dropout(0.5)(fc6)
    # FC_conv2
    fc7 = Convolution2D(1024, 1, 1, activation='relu')(fc6)  # 40 * 13 * 1024
    fc7 = Dropout(0.5)(fc7)

    score_fc7 = Convolution2D(N_cl, 1, 1, activation='relu')(fc7) # 40 * 13 * N_cl
    score_fc7_up = Deconvolution2D(N_cl, 3, 3, output_shape=(None, N_cl, 80, 26))(score_fc7)

    # scale pool4 skip for compatibility
    scale_pool4 = tf.mul(pool4, 0.01)
    scale_pool4 = ZeroPadding2D((1, 1))(scale_pool4) # 80 * 26 * 512
    score_pool4 = Convolution2D(N_cl, 1, 1, activation='relu')(scale_pool4) # 80 * 26 * N_cl
    fuse_pool4 = tf.add(score_fc7_up, score_pool4)
    score_pool4_up = Deconvolution2D(N_cl, 3, 3, output_shape=(None, N_cl, 158, 50))(fuse_pool4)

    # scale pool3 skip for compatibility
    scale_pool3 = tf.mul(pool3, 0.0001)
    score_pool3 = Convolution2D(N_cl, 1, 1, activation='relu')(scale_pool3)  # 158 * 50 * N_cl
    fuse_pool3 = tf.add(score_pool4_up, score_pool3)
    score = Deconvolution2D(N_cl, 3, 3, output_shape=(None, N_cl, 1242, 375))(fuse_pool3)

    model = Model(input=main_input, output=score)

    return model
