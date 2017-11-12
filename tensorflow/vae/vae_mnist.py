#!/usr/bin/python
# -*- coding: utf-8 -*-

"""A Variational Autoencoders for MNIST.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, Dense, Lambda, Conv2D, Conv2DTranspose, \
    Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras import metrics
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

EPOCH = 5
INPUT_DIM = 784
BATCH_SIZE = 64
HIDDEN_VAR_DIM = 7 * 7 * 32
LATENT_VAR_DIM = 2

# input image dimensions

(img_rows, img_cols, img_chns) = (28, 28, 1)

if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
    output_shape = (BATCH_SIZE, 32, 7, 7)
else:
    original_img_size = (img_rows, img_cols, img_chns)
    output_shape = (BATCH_SIZE, 7, 7, 32)


def sampling(args):
    (z_mean, z_var) = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0],
                              LATENT_VAR_DIM), mean=0., stddev=1.)
    return z_mean + z_var * epsilon


def encode(x):
    input_reshape = Reshape(original_img_size)(x)
    conv1 = Conv2D(16, 5, strides=(2, 2), padding='same',
                   activation='relu')(input_reshape)
    conv2 = Conv2D(32, 5, strides=(2, 2), padding='same',
                   activation='relu')(conv1)
    hidden = Flatten()(conv2)
    z_mean = Dense(LATENT_VAR_DIM, activation='relu')(hidden)
    z_var = Dense(LATENT_VAR_DIM, activation='relu')(hidden)
    return (z_mean, z_var)


def decode(z):
    hidden = Dense(HIDDEN_VAR_DIM, activation='relu')(z)
    hidden_reshape = Reshape(output_shape[1:])(hidden)
    deconv1 = Conv2DTranspose(16, 5, strides=(2, 2), padding='same',
                              activation='relu')(hidden_reshape)
    deconv2 = Conv2DTranspose(1, 5, strides=(2, 2), padding='same',
                              activation='sigmoid')(deconv1)
    return Flatten()(deconv2)


def main(_):
    x = Input(shape=(INPUT_DIM, ))
    (z_mean, z_var) = encode(x)
    z = Lambda(sampling)([z_mean, z_var])
    x_decoded = decode(z)
    model = Model(inputs=x, outputs=x_decoded)

    def vae_loss(y_true, y_pred):
        generation_loss = img_rows * img_cols \
            * metrics.binary_crossentropy(x, x_decoded)
        kl_loss = 0.5 * tf.reduce_sum(K.square(z_mean)
                + K.square(z_var) - K.log(K.square(z_var + 1e-8)) - 1,
                axis=1)
        return tf.reduce_mean(generation_loss + kl_loss)

    model.compile(optimizer='rmsprop', loss=vae_loss)

    # train the VAE on MNIST digits

    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train),
                              np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    print(model.summary())

    model.fit(
        x_train,
        y_train,
        shuffle=True,
        epochs=EPOCH,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        )

    generator = K.function([model.layers[8].input],
                           [model.layers[12].output])

    # display a 2D manifold of the digits

    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))

    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian

    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for (i, yi) in enumerate(grid_x):
        for (j, xi) in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            z_sample = np.tile(z_sample,
                               BATCH_SIZE).reshape(BATCH_SIZE, 2)
            x_decoded = generator([z_sample])[0]
            digit = x_decoded[0].reshape(digit_size, digit_size)

            figure[i * digit_size:(i + 1) * digit_size, j * digit_size:
                   (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


if __name__ == '__main__':
    tf.app.run(main=main)
