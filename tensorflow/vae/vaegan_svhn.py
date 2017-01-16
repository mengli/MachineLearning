import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Reshape, Activation, Convolution2D, Deconvolution2D, LeakyReLU, Flatten, \
    BatchNormalization as BN
from keras.models import Sequential, Model
from keras import initializations
from functools import partial
from svhn import SVHN

learning_rate = .0002
beta1 = .5
z_dim = 512
normal = partial(initializations.normal, scale=.02)
EPOCH = 10000
batch_size=64


def mean_normal(shape, mean=1., scale=0.02, name=None):
    return K.variable(np.random.normal(loc=mean, scale=scale, size=shape), name=name)


def generator(batch_size, gf_dim, ch, rows, cols):
    model = Sequential()

    model.add(
        Dense(gf_dim * 8 * rows[0] * cols[0], batch_input_shape=(batch_size, z_dim), name="g_h0_lin", init=normal))
    model.add(Reshape((rows[0], cols[0], gf_dim * 8)))
    model.add(BN(mode=2, axis=3, name="g_bn0", gamma_init=mean_normal, epsilon=1e-5))
    model.add(Activation("relu"))

    model.add(Deconvolution2D(gf_dim * 4, 5, 5, output_shape=(batch_size, rows[1], cols[1], gf_dim * 4), subsample=(2, 2),
                              name="g_h1", border_mode="same", init=normal))
    model.add(BN(mode=2, axis=3, name="g_bn1", gamma_init=mean_normal, epsilon=1e-5))
    model.add(Activation("relu"))

    model.add(Deconvolution2D(gf_dim * 2, 5, 5, output_shape=(batch_size, rows[2], cols[2], gf_dim * 2), subsample=(2, 2),
                              name="g_h2", border_mode="same", init=normal))
    model.add(BN(mode=2, axis=3, name="g_bn2", gamma_init=mean_normal, epsilon=1e-5))
    model.add(Activation("relu"))

    model.add(Deconvolution2D(ch, 5, 5, output_shape=(batch_size, rows[3], cols[3], ch), subsample=(2, 2), name="g_h3",
                              border_mode="same", init=normal))
    model.add(Activation("tanh"))

    return model


def encoder(batch_size, df_dim, ch, rows, cols):
    model = Sequential()
    X = Input(batch_shape=(batch_size, rows[-1], cols[-1], ch))
    model = Convolution2D(df_dim, 5, 5, subsample=(2, 2), border_mode="same",
                          name="e_h0_conv", dim_ordering="tf", init=normal)(X)
    model = LeakyReLU(.2)(model)

    model = Convolution2D(df_dim * 2, 5, 5, subsample=(2, 2), border_mode="same",
                          name="e_h1_conv", dim_ordering="tf")(model)
    model = BN(mode=2, axis=3, name="e_bn1", gamma_init=mean_normal, epsilon=1e-5)(model)
    model = LeakyReLU(.2)(model)

    model = Convolution2D(df_dim * 4, 5, 5, subsample=(2, 2), name="e_h2_conv", border_mode="same",
                          dim_ordering="tf", init=normal)(model)
    model = BN(mode=2, axis=3, name="e_bn2", gamma_init=mean_normal, epsilon=1e-5)(model)
    model = LeakyReLU(.2)(model)
    model = Flatten()(model)

    mean = Dense(z_dim, name="e_h3_lin", init=normal)(model)
    logsigma = Dense(z_dim, name="e_h4_lin", activation="tanh", init=normal)(model)
    meansigma = Model([X], [mean, logsigma])
    return meansigma


def discriminator(batch_size, df_dim, ch, rows, cols):
    X = Input(batch_shape=(batch_size, rows[-1], cols[-1], ch))
    model = Convolution2D(df_dim, 5, 5, subsample=(2, 2), border_mode="same",
                          name="d_h0_conv", dim_ordering="tf", init=normal)(X)
    model = LeakyReLU(.2)(model)

    model = Convolution2D(df_dim * 2, 5, 5, subsample=(2, 2), border_mode="same",
                          name="d_h1_conv", dim_ordering="tf", init=normal)(model)
    model = BN(mode=2, axis=3, name="d_bn1", gamma_init=mean_normal, epsilon=1e-5)(model)
    model = LeakyReLU(.2)(model)

    model = Convolution2D(df_dim * 4, 5, 5, subsample=(2, 2), border_mode="same",
                          name="d_h2_conv", dim_ordering="tf", init=normal)(model)

    dec = BN(mode=2, axis=3, name="d_bn3", gamma_init=mean_normal, epsilon=1e-5)(model)
    dec = LeakyReLU(.2)(dec)
    dec = Flatten()(dec)
    dec = Dense(1, name="d_h3_lin", init=normal)(dec)

    output = Model([X], [dec, model])

    return output


def get_model(sess, image_shape=(32, 32, 3), gf_dim=64, df_dim=64, batch_size=64,
              name="autoencoder"):
    K.set_session(sess)
    with tf.variable_scope(name):
        # sizes
        ch = image_shape[2]
        rows = [4, 8, 16, 32]
        cols = [4, 8, 16, 32]

        # nets
        G = generator(batch_size, gf_dim, ch, rows, cols)
        G.compile("sgd", "mse")
        g_vars = G.trainable_weights
        print "G.shape: ", G.output_shape

        E = encoder(batch_size, df_dim, ch, rows, cols)
        E.compile("sgd", "mse")
        e_vars = E.trainable_weights
        print "E.shape: ", E.output_shape

        D = discriminator(batch_size, df_dim, ch, rows, cols)
        D.compile("sgd", "mse")
        d_vars = D.trainable_weights
        print "D.shape: ", D.output_shape

        Z2 = Input(batch_shape=(batch_size, z_dim), name='more_noise')
        Z = G.input
        Img = D.input
        image_grid = put_kernels_on_grid(tf.transpose(Img, [1, 2, 3, 0]), (8, 8))
        sum_img = tf.summary.image("Img", image_grid, max_outputs=1)
        G_train = G(Z)
        E_mean, E_logsigma = E(Img)
        G_dec = G(E_mean + Z2 * E_logsigma)
        D_fake, F_fake = D(G_train)
        D_dec_fake, F_dec_fake = D(G_dec)
        D_legit, F_legit = D(Img)

        # costs
        recon_vs_gan = 1e-6
        like_loss = tf.reduce_mean(tf.square(F_legit - F_dec_fake)) / 2.
        kl_loss = tf.reduce_mean(-E_logsigma + .5 * (-1 + tf.exp(2. * E_logsigma) + tf.square(E_mean)))

        d_loss_legit = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_legit, tf.ones_like(D_legit)))
        d_loss_fake1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake, tf.zeros_like(D_fake)))
        d_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_dec_fake, tf.zeros_like(D_dec_fake)))
        d_loss_fake = d_loss_fake1 + d_loss_fake2
        d_loss = d_loss_legit + d_loss_fake

        g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_fake, tf.ones_like(D_fake)))
        g_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_dec_fake, tf.ones_like(D_dec_fake)))
        g_loss = g_loss1 + g_loss2 + recon_vs_gan * like_loss
        e_loss = kl_loss + like_loss

        # optimizers
        print "Generator variables:"
        for v in g_vars:
            print v.name
        print "Discriminator variables:"
        for v in d_vars:
            print v.name
        print "Encoder variables:"
        for v in e_vars:
            print v.name

        e_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(e_loss, var_list=e_vars)
        d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
        sess.run(tf.global_variables_initializer())

    # summaries
    sum_d_loss_legit = tf.summary.scalar("d_loss_legit", d_loss_legit)
    sum_d_loss_fake = tf.summary.scalar("d_loss_fake", d_loss_fake)
    sum_d_loss = tf.summary.scalar("d_loss", d_loss)
    sum_g_loss = tf.summary.scalar("g_loss", g_loss)
    sum_e_loss = tf.summary.scalar("e_loss", e_loss)
    sum_e_mean = tf.summary.histogram("e_mean", E_mean)
    sum_e_sigma = tf.summary.histogram("e_sigma", tf.exp(E_logsigma))
    sum_Z = tf.summary.histogram("Z", Z)
    image_grid = put_kernels_on_grid(tf.transpose(G_train, [1, 2, 3, 0]), (8, 8))
    sum_gen = tf.summary.image("G", image_grid, max_outputs=1)
    image_grid = put_kernels_on_grid(tf.transpose(G_dec, [1, 2, 3, 0]), (8, 8))
    sum_dec = tf.summary.image("E", image_grid, max_outputs=1)

    g_sum = tf.summary.merge([sum_Z, sum_gen, sum_d_loss_fake, sum_g_loss, sum_img])
    e_sum = tf.summary.merge([sum_dec, sum_e_loss, sum_e_mean, sum_e_sigma])
    d_sum = tf.summary.merge([sum_d_loss_legit, sum_d_loss])
    writer = tf.summary.FileWriter("train", sess.graph)

    # functions
    def train_d(images, z, counter, sess=sess):
        z2 = np.random.normal(0., 1., z.shape)
        outputs = [d_loss, d_loss_fake, d_loss_legit, d_sum, d_optim]
        images = np.transpose(np.reshape(images, (-1, 3, 32, 32)), (0, 2, 3, 1))
        with tf.control_dependencies(outputs):
            updates = [tf.assign(p, new_p) for (p, new_p) in D.updates]
        outs = sess.run(outputs + updates, feed_dict={Img: images, Z: z, Z2: z2, K.learning_phase(): 1})
        dl, dlf, dll, sums = outs[:4]
        writer.add_summary(sums, counter)
        return dl, dlf, dll

    def train_g(images, z, counter, sess=sess):
        # generator
        z2 = np.random.normal(0., 1., z.shape)
        outputs = [g_loss, G_train, g_sum, g_optim]
        images = np.transpose(np.reshape(images, (-1, 3, 32, 32)), (0, 2, 3, 1))
        with tf.control_dependencies(outputs):
            updates = [tf.assign(p, new_p) for (p, new_p) in G.updates]
        outs = sess.run(outputs + updates, feed_dict={Img: images, Z: z, Z2: z2, K.learning_phase(): 1})
        gl, samples, sums = outs[:3]
        writer.add_summary(sums, counter)
        # encoder
        outputs = [e_loss, G_dec, e_sum, e_optim]
        with tf.control_dependencies(outputs):
            updates = [tf.assign(p, new_p) for (p, new_p) in E.updates]
        outs = sess.run(outputs + updates, feed_dict={Img: images, Z: z, Z2: z2, K.learning_phase(): 1})
        gl, samples, sums = outs[:3]
        writer.add_summary(sums, counter)

        return gl, samples, images

    def sampler(z, x):
        code = E.predict(x, batch_size=batch_size)[0]
        out = G.predict(code, batch_size=batch_size)
        return out, x

    return train_g, train_d, sampler, [G, D, E]

def put_kernels_on_grid(kernel, (grid_Y, grid_X), pad=1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
    '''
    # pad X and Y
    x1 = tf.pad(kernel, tf.constant([[pad, 0], [pad, 0], [0, 0], [0, 0]]))

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + pad
    X = kernel.get_shape()[1] + pad
    ch = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, ch]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, ch]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 1]
    x_min = tf.reduce_min(x7)
    x_max = tf.reduce_max(x7)
    x8 = (x7 - x_min) / (x_max - x_min)

    return x8


def fetch_next_batch(svhn):
    z = np.random.normal(0., 1., (batch_size, z_dim))  # normal dist for GAN
    x = svhn.train.next_batch(batch_size)
    return z, x[0]


def train_model(name, g_train, d_train, sampler):
    """
    Main training loop.
    modified from Keras fit_generator
    """
    epoch = 0
    svhn = SVHN()
    svhn.ReadDataSets(one_hot=True)

    while epoch < EPOCH:
        z, x = fetch_next_batch(svhn)
        d_losses = d_train(x, z, epoch)
        z, x = fetch_next_batch(svhn)
        g_loss, samples, xs = g_train(x, z, epoch)
        if epoch % 2 == 0:
            print "epoch: ", epoch
        epoch += 1


if __name__ == "__main__":
    with tf.Session() as sess:
        g_train, d_train, sampler, extras = get_model(sess=sess)

        train_model("autoencoder", g_train, d_train, sampler)
